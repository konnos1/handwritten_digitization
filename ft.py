import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    from jiwer import cer, wer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

# ======================================================
# CONFIG
# ======================================================
MODEL_DIR = 'CRNN_models' 
LOCAL_DATA_DIR = os.path.join('combined_training', 'lines')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_3_Real_Hard.pth') 
OUTPUT_NAME = 'fine_tuned_full_unfreeze.pth' 
PLOT_NAME = 'training_plot_full_unfreeze.png'

IMG_HEIGHT = 128
IMG_WIDTH = 512
BATCH_SIZE = 8       
EPOCHS = 40        
LEARNING_RATE = 1e-4 

DEVICE = torch.device('cpu') 

# ======================================================
# MODEL
# ======================================================
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

 ======================================================
# UTILS & TRANSFORMS
# ======================================================
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
            
        if self.transform: image = self.transform(image)
        encoded = [self.char2idx[c] for c in text if c in self.char2idx]
        return image, torch.tensor(encoded, dtype=torch.long), text

def collate_fn(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return images, targets_padded, target_lengths, texts

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            # Τυχαίες αλλαγές φωτεινότητας/αντίθεσης
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            # Ελαφριά περιστροφή και κλίση
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def decode_greedy(tokens, idx2char):
    res = []
    prev = None
    for t in tokens:
        if t != prev and t != 0:
            res.append(idx2char.get(t, ''))
        prev = t
    return "".join(res)

def evaluate_epoch(model, loader, idx2char):
    model.eval()
    preds = []
    truths = []
    
    with torch.no_grad():
        for images, targets, target_lengths, texts in loader:
            images = images.to(DEVICE)
            log_probs, seq_len = model(images)
            _, max_idx = torch.max(log_probs, dim=2)
            
            for i in range(len(texts)):
                pred_text = decode_greedy(max_idx[i].cpu().numpy(), idx2char)
                preds.append(pred_text)
                truths.append(texts[i])
    
    epoch_cer = cer(truths, preds) if HAS_JIWER else 0.0
    epoch_wer = wer(truths, preds) if HAS_JIWER else 0.0
    return preds, truths, epoch_cer, epoch_wer

# ======================================================
# MAIN
# ======================================================
if __name__ == '__main__':
    print("Loading Metadata...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    char2idx = meta['char2idx']
    idx2char = {int(v): k for k, v in char2idx.items()}
    vocab_size = meta['vocab_size']
    
    csv_path = os.path.join(LOCAL_DATA_DIR, 'train_541.csv')
    if not os.path.exists(csv_path): raise FileNotFoundError(f"Δεν βρέθηκε το {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Local Samples: {len(df)}")

    # 90% Train, 10% Validation
    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42, shuffle=True)

    print(f"Training Samples: {len(train_df)}")
    print(f"Validation Samples: {len(val_df)}")
    
    train_dataset = CRNNDataset(train_df, LOCAL_DATA_DIR, char2idx, transform=get_transforms(train=True))
    val_dataset = CRNNDataset(val_df, LOCAL_DATA_DIR, char2idx, transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = ResearchCRNN(vocab_size=vocab_size, hidden_size=128).to(DEVICE)
    
    print(f"Loading Weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    # ==========================================================
    # PARTIAL UNFREEZE
    # ==========================================================
    print("Configuring Layers: Unfreezing RNN + Classifier + Last CNN Block (Conv6)")
    for name, param in model.named_parameters():
        if 'rnn' in name or 'classifier' in name:
            param.requires_grad = True # εκπαιδεύουμε το RNN
        elif 'conv6' in name or 'bn6' in name:
            param.requires_grad = True # Ξεπαγώνουμε το τελευταίο CNN layer
        else:
            param.requires_grad = False # Τα Conv1-Conv5 μένουν παγωμένα
            
    # Έλεγχος
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    history = {'loss': [], 'cer': [], 'wer': []}

    print(f"\n Starting Enhanced Fine-Tuning for {EPOCHS} epochs...")
    
    best_cer = float('inf')

    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for images, targets, target_lengths, _ in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)
            
            optimizer.zero_grad()
            log_probs, seq_len = model(images)
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long).to(DEVICE)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        print("Evaluating...")
        preds, truths, epoch_cer, epoch_wer = evaluate_epoch(model, val_loader, idx2char)
        
        # Ενημέρωση Scheduler
        scheduler.step(avg_loss)
        

        # Αποθήκευση μόνο αν βελτιωθεί το CER
        if epoch_cer < best_cer:
            best_cer = epoch_cer
            print(f"New Best Model Saved! (CER: {best_cer:.4f})")
            torch.save({'model_state': model.state_dict(), 'vocab_size': vocab_size}, OUTPUT_NAME)
            patience = 0  # Μηδενισμός αν βρούμε καλύτερο
        else:
            patience += 1
            print(f"Patience: {patience}/10") # Ενημέρωση

        # ΕΛΕΓΧΟΣ EARLY STOPPING
        if patience >= 10: # Αν περάσουν 10 εποχές χωρίς βελτίωση
            print("Early Stopping triggered! Stopping training.")
            break

        print(f"Ep {epoch+1} | Loss: {avg_loss:.4f} | CER: {epoch_cer:.4f} | WER: {epoch_wer:.4f}")
        
        history['loss'].append(avg_loss)
        history['cer'].append(epoch_cer)
        history['wer'].append(epoch_wer)

        idx = random.randint(0, len(preds)-1)
        print(f"Sample: GT='{truths[idx]}' | PRED='{preds[idx]}'")

        # Αποθήκευση μόνο αν βελτιωθεί το CER
        if epoch_cer < best_cer:
            best_cer = epoch_cer
            print(f"New Best Model Saved! (CER: {best_cer:.4f})")
            torch.save({'model_state': model.state_dict(), 'vocab_size': vocab_size}, OUTPUT_NAME)

    # --- PLOTTING ---
    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.plot(history['loss'], label='Loss'); plt.legend(); plt.title('Loss')
    plt.subplot(132); plt.plot(history['cer'], label='CER', color='orange'); plt.legend(); plt.title('CER')
    plt.subplot(133); plt.plot(history['wer'], label='WER', color='green'); plt.legend(); plt.title('WER')
    plt.savefig(PLOT_NAME)
    print(f"Plot saved as {PLOT_NAME}")
