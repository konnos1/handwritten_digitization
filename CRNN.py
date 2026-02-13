import os
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import evaluate
from google.colab import drive
import random
import matplotlib.pyplot as plt
import json

# ============================================================================
# 1. SETUP & REPRODUCIBILITY
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

drive.mount('/content/drive', force_remount=True)

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 512
BATCH_SIZE = 16
ACCUM_STEPS = 4  # Effective Batch = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_OUTPUT_DIR = '/content/drive/MyDrive/Thesis_OCR/CRNN_Final'

if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

print(f"Device: {DEVICE} | Mixed Precision: ON | Effective Batch: {BATCH_SIZE * ACCUM_STEPS}")

scaler = torch.amp.GradScaler('cuda')

# ============================================================================
# 2. DATA UTILS & COLLATOR
# ============================================================================
def build_unified_vocabulary(csv_paths):
    print("Building Unified Vocabulary...")
    all_chars = set()
    base_vocab = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίόύώϊϋΐΰς0123456789.,-!?()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    all_chars.update(base_vocab)

    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            text = "".join(df['text'].astype(str).tolist())
            all_chars.update(text)

    chars = sorted(list(all_chars))
    char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
    idx2char = {idx + 1: char for idx, char in enumerate(chars)}

    print(f"Vocab Size: {len(chars) + 1} (incl. blank)")
    return char2idx, idx2char, len(chars) + 1

class CRNNDataset(Dataset):
    def __init__(self, df, root_dir, char2idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.char2idx = char2idx

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['file_name'])
        text = str(row['text'])

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT))

        if self.transform:
            image = self.transform(image)

        encoded = []
        for c in text:
            if c in self.char2idx:
                encoded.append(self.char2idx[c])

        return image, torch.tensor(encoded, dtype=torch.long), text


def collate_fn(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return images, targets_padded, target_lengths, texts


def compute_dataset_stats(df, root_dir):
    """Computes Mean/Std for a single dataset"""
    print(f"Computing Stats for {len(df)} images...")
    basic_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])

    subset_df = df.sample(min(1000, len(df)), random_state=42)
    dummy_c2i = {'a': 1}
    ds = CRNNDataset(subset_df, root_dir, dummy_c2i, transform=basic_transform)


    loader = DataLoader(ds, batch_size=32, num_workers=2, shuffle=False, collate_fn=collate_fn)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_batches = 0

    for images, _, _, _ in loader:
        mean += images.mean([0, 2, 3])
        std += images.std([0, 2, 3])
        total_batches += 1

    if total_batches == 0:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    mean /= total_batches
    std /= total_batches
    return mean.tolist(), std.tolist()

def precompute_all_stats(stages):
    stats_cache = {}
    for stage in stages:
        csv_path = os.path.join(stage['dir'], 'train.csv')
        json_path = os.path.join(BASE_OUTPUT_DIR, f"stats_{stage['name']}.json")

        if os.path.exists(json_path):
            print(f"Loading cached stats for {stage['name']}...")
            with open(json_path, 'r') as f:
                data = json.load(f)
                stats_cache[stage['name']] = (data['mean'], data['std'])
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            mean, std = compute_dataset_stats(df, stage['dir'])
            stats_cache[stage['name']] = (mean, std)
            with open(json_path, 'w') as f:
                json.dump({'mean': mean, 'std': std}, f)
        else:
            print(f"Dataset missing for {stage['name']}, skipping stats.")
            stats_cache[stage['name']] = ([0.5]*3, [0.5]*3)

    return stats_cache

# ============================================================================
# 3. TRANSFORMS & MODEL
# ============================================================================
class DeslantTransform:
    def __call__(self, img):
        angle = np.random.uniform(-15, 15)
        return transforms.functional.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=angle)

def get_transforms(stage_type, mean, std):
    base = [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    if stage_type == "printed":
        aug = []
    elif stage_type == "handwritten":
        aug = [
            transforms.RandomApply([DeslantTransform()], p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02))
        ]
    else: # real
        aug = [
            transforms.RandomApply([DeslantTransform()], p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ]
    return transforms.Compose(aug + base)

class ResearchCRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64); self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128); self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256); self.pool4 = nn.MaxPool2d((2, 1))
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1); self.bn5 = nn.BatchNorm2d(256); self.pool5 = nn.MaxPool2d((2, 1))
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1); self.bn6 = nn.BatchNorm2d(512)

        self.rnn = nn.GRU(512, hidden_size, num_layers=3, bidirectional=True, batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.leaky_relu(self.bn5(self.conv5(x))))
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x = F.adaptive_avg_pool2d(x, (1, None))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        return F.log_softmax(self.classifier(x), dim=2), x.size(1)

# ============================================================================
# 4. TRAINER
# ============================================================================
def decode_greedy(tokens, idx2char):
    res = []
    prev = None
    for t in tokens:
        if t != prev and t != 0: res.append(idx2char[t])
        prev = t
    return "".join(res)

class ResearchTrainer:
    def __init__(self, model, char2idx, idx2char):
        self.model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")
        self.history = {}

    def train_epoch(self, loader, optimizer, accum_steps):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc="Train", leave=False)
        optimizer.zero_grad()

        for i, (images, targets, target_lengths, _) in enumerate(pbar):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            with torch.amp.autocast('cuda'):
                log_probs, seq_len = self.model(images)
                log_probs = log_probs.permute(1, 0, 2)
                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(DEVICE)
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            pbar.set_postfix({'loss': loss.item() * accum_steps})

        if len(loader) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return total_loss / len(loader)

    def eval_model(self, loader):
        self.model.eval()
        val_loss = 0
        preds, truths = [], []

        with torch.no_grad():
            for images, targets, target_lengths, texts in loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                target_lengths = target_lengths.to(DEVICE)

                log_probs, seq_len = self.model(images)
                log_probs_perm = log_probs.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long).to(DEVICE)
                loss = self.criterion(log_probs_perm, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                _, max_idx = torch.max(log_probs, dim=2)
                for k in range(len(texts)):
                    preds.append(decode_greedy(max_idx[k].cpu().numpy(), self.idx2char))
                    truths.append(texts[k])

        cer = self.cer_metric.compute(predictions=preds, references=truths)
        wer = self.wer_metric.compute(predictions=preds, references=truths)
        return val_loss / len(loader), cer, wer, preds, truths

    def error_analysis(self, preds, truths, config_name):
        errors = []
        for p, t in zip(preds, truths):
            sample_cer = self.cer_metric.compute(predictions=[p], references=[t])
            errors.append({'gt': t, 'pred': p, 'cer': sample_cer})
        errors = sorted(errors, key=lambda x: x['cer'], reverse=True)[:50]
        with open(os.path.join(BASE_OUTPUT_DIR, f"errors_{config_name}.txt"), 'w', encoding='utf-8') as f:
            for e in errors:
                f.write(f"CER: {e['cer']:.2f} | GT: {e['gt']} | PRED: {e['pred']}\n")
        print(f"Worst errors saved to errors_{config_name}.txt")

    def run_stage(self, config, stats_cache):
        print(f"\n STAGE: {config['name']} | Epochs: {config['epochs']}")
        self.history = {'train_loss': [], 'val_loss': [], 'cer': [], 'wer': []}

        df = pd.read_csv(os.path.join(config['dir'], 'train.csv'))
        mean, std = stats_cache[config['name']]

        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        if config['type'] == 'real':
            train_df = pd.concat([train_df] * 5).reset_index(drop=True)

        transforms_train = get_transforms(config['type'], mean, std)
        transforms_eval = get_transforms("printed", mean, std)

        train_ds = CRNNDataset(train_df, config['dir'], self.char2idx, transforms_train)
        val_ds = CRNNDataset(val_df, config['dir'], self.char2idx, transforms_eval)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

        if config['freeze_cnn']:
            for param in self.model.parameters(): param.requires_grad = False
            for param in self.model.rnn.parameters(): param.requires_grad = True
            for param in self.model.classifier.parameters(): param.requires_grad = True
        else:
            for param in self.model.parameters(): param.requires_grad = True

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        warmup_epochs = 3
        best_cer = float('inf')
        patience = 0

        for epoch in range(config['epochs']):
            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups: pg['lr'] = config['lr'] * lr_scale

            train_loss = self.train_epoch(train_loader, optimizer, ACCUM_STEPS)
            val_loss, cer, wer, preds, truths = self.eval_model(val_loader)

            if epoch >= warmup_epochs:
                scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['cer'].append(cer)
            self.history['wer'].append(wer)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch+1} | LR:{current_lr:.6f} | L:{train_loss:.4f} | V:{val_loss:.4f} | CER:{cer:.4f}")

            # Print Random Sample for monitoring
            if preds:
                rand_idx = random.randint(0, len(preds)-1)
                print(f"   Sample: GT='{truths[rand_idx][:50]}' | PRED='{preds[rand_idx][:50]}'")

            if cer < best_cer:
                best_cer = cer
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'history': self.history
                }, os.path.join(BASE_OUTPUT_DIR, f"best_{config['name']}.pth"))
                patience = 0
            else:
                patience += 1
                if patience >= 7:
                    print("Early Stopping!")
                    break

        print("Running Final Test Evaluation (Greedy)...")
        test_ds = CRNNDataset(test_df, config['dir'], self.char2idx, transforms_eval)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        ckpt = torch.load(os.path.join(BASE_OUTPUT_DIR, f"best_{config['name']}.pth"))
        self.model.load_state_dict(ckpt['model_state'])
        _, test_cer, test_wer, test_preds, test_truths = self.eval_model(test_loader)

        print(f"TEST RESULTS: CER={test_cer:.4f} | WER={test_wer:.4f}")
        self.error_analysis(test_preds, test_truths, config['name'])

        plt.figure(figsize=(15, 5))
        plt.subplot(131); plt.plot(self.history['train_loss'], label='Train'); plt.plot(self.history['val_loss'], label='Val'); plt.title('Loss'); plt.legend(); plt.xlabel('Epoch')
        plt.subplot(132); plt.plot(self.history['cer']); plt.title('CER'); plt.xlabel('Epoch')
        plt.subplot(133); plt.plot(self.history['wer']); plt.title('WER'); plt.xlabel('Epoch')
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, f"metrics_{config['name']}.png"))
        plt.close()

# ============================================================================
# 5. EXECUTION CONFIG
# ============================================================================
STAGES = [
    {
        'name': '1_Printed', 'type': 'printed',
        'zip': '/content/drive/MyDrive/Thesis_OCR/datasets/dataset_printed1.zip',
        'dir': '/content/data_printed',
        'epochs': 20, 'lr': 1e-3, 'freeze_cnn': False
    },
  {
        'name': '2_Handwritten_Easy',
        'type': 'handwritten',
        'zip': '/content/drive/MyDrive/Thesis_OCR/datasets/dataset_easy1.zip',
        'dir': '/content/data_hw_easy',
        'epochs': 15,
        'lr': 5e-4,
        'freeze_cnn': False
    },
    {
        'name': '3_Real_Hard',
        'type': 'real',
        'zip': '/content/drive/MyDrive/Thesis_OCR/datasets/dataset_hard_real1.zip',
        'dir': '/content/data_real',
        'epochs': 20,
        'lr': 1e-4,
        'freeze_cnn': False
    }
]

# 1. Prepare Data & Vocab
csv_paths = []
for stage in STAGES:
    if os.path.exists(stage['zip']):
        if not os.path.exists(stage['dir']):
            with zipfile.ZipFile(stage['zip'], 'r') as z: z.extractall(stage['dir'])
        csv_paths.append(os.path.join(stage['dir'], 'train.csv'))

# 2. Build Vocab & Stats
char2idx, idx2char, vocab_size = build_unified_vocabulary(csv_paths)
stats_cache = precompute_all_stats(STAGES)

# 3. Init Model & Trainer
model = ResearchCRNN(vocab_size=vocab_size, hidden_size=128).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f" Model initialized: {total_params:,} parameters")

trainer = ResearchTrainer(model, char2idx, idx2char)

# 4. Run Stages
for stage in STAGES:
    trainer.run_stage(stage, stats_cache)

def save_project_metadata(output_dir, char2idx, idx2char, model_params):
    metadata = {
        'char2idx': char2idx,
        'idx2char': idx2char,
        'vocab_size': len(char2idx) + 1,
        'model_params': model_params
    }

    save_path = os.path.join(output_dir, 'metadata.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Metadata saved to: {save_path}")

save_project_metadata(
    BASE_OUTPUT_DIR,
    char2idx,
    idx2char,
    {'hidden_size': 128, 'img_height': 128, 'img_width': 512}
)
