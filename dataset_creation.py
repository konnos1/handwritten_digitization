import os
import zipfile
import shutil
import random
import glob
import numpy as np
import pandas as pd
import unicodedata
import math
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps, ImageFilter, ImageEnhance
from tqdm.auto import tqdm
from google.colab import drive
import requests

# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"Seed: {SEED}")

# =============================================================================
# 1. DRIVE & PATHS
# =============================================================================
drive.mount('/content/drive')

ZIP_ALPHABETS = '/content/drive/MyDrive/Greek Alphabets Dataset.zip'
ZIP_ARCHIVE = '/content/drive/MyDrive/archive.zip'

# Φάκελοι προσωρινής αποθήκευσης στο Colab
DIR_PRINTED = 'dataset_printed'
DIR_EASY = 'dataset_easy'
DIR_HARD = 'dataset_hard'

# Τελικά Paths στο Drive
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/Thesis_OCR/datasets'
ZIP_PATH_PRINTED = os.path.join(DRIVE_OUTPUT_DIR, 'dataset_printed1.zip')
ZIP_PATH_EASY = os.path.join(DRIVE_OUTPUT_DIR, 'dataset_easy1.zip')
ZIP_PATH_HARD = os.path.join(DRIVE_OUTPUT_DIR, 'dataset_hard_real1.zip')

TEMP_DIR = 'temp_input_files'
TEMP_ACCENTS_DIR = 'temp_accented_chars'

# Cleanup & Creation
for d in [TEMP_DIR, TEMP_ACCENTS_DIR, DIR_PRINTED, DIR_EASY, DIR_HARD]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Δημιουργία υποφακέλων images
os.makedirs(os.path.join(DIR_PRINTED, 'images'), exist_ok=True)
os.makedirs(os.path.join(DIR_EASY, 'images'), exist_ok=True)
os.makedirs(os.path.join(DIR_HARD, 'images'), exist_ok=True)

# =============================================================================
# 2. GREEK FONT SETUP
# =============================================================================
print("Εγκατάσταση & Validation Ελληνικών Γραμματοσειρών...")

# Install Greek fonts
os.system('apt-get update -qq 2>&1 > /dev/null')
os.system('apt-get install -y fonts-dejavu fonts-liberation fonts-noto fonts-freefont-ttf 2>&1 > /dev/null')

# Find all system fonts
SYSTEM_FONTS_DIR = "/usr/share/fonts/truetype/"
all_fonts = glob.glob(SYSTEM_FONTS_DIR + "**/*.ttf", recursive=True)

def test_greek_support_strict(font_path):
    """
    Ελέγχει αν η γραμματοσειρά τυπώνει 'Tofu' αντί για γράμματα.
    """
    try:
        font = ImageFont.truetype(font_path, 40)

        # Helper για γρήγορο render
        def render_char(char):
            img = Image.new('L', (50, 50), color=255)
            draw = ImageDraw.Draw(img)
            # Κεντράρισμα για να μην έχουμε θέματα μετατόπισης
            bbox = font.getbbox(char)
            if not bbox: return None # Αν δεν επιστρέφει καν bbox, είναι άκυρη
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text(((50-w)/2, (50-h)/2), char, font=font, fill=0)
            return np.array(img)

        # 1. Βασικός έλεγχος: Υπάρχει μελάνι στο 'α;
        img_alpha = render_char("α")
        if img_alpha is None: return False
        if np.sum(img_alpha < 250) < 10: return False # Κενή εικόνα

        # 2. Έλεγχος Tofu #1: Μοιάζει το 'α' με ανύπαρκτο χαρακτήρα;
        # Χρησιμοποιούμε έναν χαρακτήρα που σίγουρα δεν υπάρχει (Private Use Area)
        # Αν η γραμματοσειρά δεν έχει 'α', θα τυπώσει το ίδιο κουτάκι και στα δύο.
        img_garbage = render_char(chr(0xE000))
        if img_garbage is not None:
            # Αν είναι ακριβώς ίδια τα pixels, σημαίνει ότι και το 'α' τυπώθηκε ως κουτάκι
            if np.array_equal(img_alpha, img_garbage):
                return False

        # 3. Έλεγχος Tofu #2: Μοιάζει το 'α' με το 'ω';
        # Αν η γραμματοσειρά τυπώνει κουτάκια, το κουτάκι του 'α' θα είναι ίδιο με του 'ω'.
        img_omega = render_char("ω")
        if img_omega is not None:
            if np.array_equal(img_alpha, img_omega):
                return False

        return True

    except Exception as e:
        return False

print("Testing fonts for REAL Greek support")

# Test fonts in batches (faster)
greek_fonts = []
for font_path in tqdm(all_fonts, desc="Validating fonts"):
    if test_greek_support_strict(font_path):
        greek_fonts.append(font_path)

    # Stop after finding 20 good fonts (enough variety)
    if len(greek_fonts) >= 20:
        break

if not greek_fonts:
    print("No valid Greek fonts found! Using fallback strategy...")
    # Fallback: Known good fonts
    fallback_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    ]
    greek_fonts = [f for f in fallback_paths if os.path.exists(f)]

print(f"Found {len(greek_fonts)} VERIFIED Greek fonts")
print(f"Examples: {[os.path.basename(f) for f in greek_fonts[:5]]}")


# =============================================================================
# 3. EXTRACT HANDWRITING DATASETS
# =============================================================================
print("Αποσυμπίεση...")
try:
    with zipfile.ZipFile(ZIP_ALPHABETS, 'r') as z:
        z.extractall(os.path.join(TEMP_DIR, 'Dataset1'))
    with zipfile.ZipFile(ZIP_ARCHIVE, 'r') as z:
        z.extractall(os.path.join(TEMP_DIR, 'Dataset2'))
except FileNotFoundError as e:
    print(f"{e}")
    exit()

# =============================================================================
# 4. CHARACTER DATABASE MAPPING
# =============================================================================
char_db = {}

map_ds1 = {
    "alpha": "α", "beta": "β", "chi": "χ", "delta": "δ", "epsilon": "ε", "eta": "η",
    "gamma": "γ", "kappa": "κ", "lamda": "λ", "lota": "ι", "mu": "μ", "nu": "ν",
    "omega": "ω", "omicro": "ο", "phi": "φ", "pi": "π", "psi": "ψ", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "theta": "θ", "upsilo": "υ", "xi": "ξ", "zeta": "ζ"
}

map_ds2 = {
    # ΜΙΚΡΑ
    "LETT_SML_NORM.ALPHA": "α", "LETT_SML_NORM.BETA": "β", "LETT_SML_NORM.GAMMA": "γ", "LETT_SML_NORM.DELTA": "δ",
    "LETT_SML_NORM.EPSILON": "ε", "LETT_SML_NORM.ZETA": "ζ", "LETT_SML_NORM.HETA": "η", "LETT_SML_NORM.THETA": "θ",
    "LETT_SML_NORM.IOTA": "ι", "LETT_SML_NORM.KAPA": "κ", "LETT_SML_NORM.LAMDA": "λ", "LETT_SML_NORM.MI": "μ",
    "LETT_SML_NORM.NI": "ν", "LETT_SML_NORM.KSI": "ξ", "LETT_SML_NORM.OMIKRON": "ο", "LETT_SML_NORM.PIΙ": "π",
    "LETT_SML_NORM.PSI": "ψ", "LETT_SML_NORM.RO": "ρ", "LETT_SML_NORM.SIGMA": "σ", "LETT_SML_NORM.TAU": "τ",
    "LETT_SML_NORM.YPSILON": "υ", "LETT_SML_NORM.FI": "φ", "LETT_SML_NORM.XI": "χ", "LETT_SML_NORM.OMEGA": "ω",
    # SUFFIXES
    "LETT_SML_SUFF.ALPHA": "α", "LETT_SML_SUFF.BETA": "β", "LETT_SML_SUFF.GAMMA": "γ", "LETT_SML_SUFF.DELTA": "δ",
    "LETT_SML_SUFF.EPSILON": "ε", "LETT_SML_SUFF.ZETA": "ζ", "LETT_SML_SUFF.HETA": "η", "LETT_SML_SUFF.THETA": "θ",
    "LETT_SML_SUFF.IOTA": "ι", "LETT_SML_SUFF.KAPA": "κ", "LETT_SML_SUFF.LAMDA": "λ", "LETT_SML_SUFF.MI": "μ",
    "LETT_SML_SUFF.NI": "ν", "LETT_SML_SUFF.KSI": "ξ", "LETT_SML_SUFF.OMIKRON": "ο", "LETT_SML_SUFF.PII": "π",
    "LETT_SML_SUFF.PSI": "ψ", "LETT_SML_SUFF.RO": "ρ", "LETT_SML_SUFF.SIGMA": "σ", "LETT_SML_SUFF.TAU": "τ",
    "LETT_SML_SUFF.YPSILON": "υ", "LETT_SML_SUFF.FI": "φ", "LETT_SML_SUFF.xI": "χ", "LETT_SML_SUFF.OMEGA": "ω",
    # ΚΕΦΑΛΑΙΑ
    "LETT_CAP_NORM.ALPHA": "Α", "LETT_CAP_SUFF.ALPHA": "Α", "LETT_CAP_NORM.BETA": "Β", "LETT_CAP_SUFF.BETA": "Β",
    "LETT_CAP_NORM.GAMMA": "Γ", "LETT_CAP_SUFF.GAMMA": "Γ", "LETT_CAP_NORM.DELTA": "Δ", "LETT_CAP_SUFF.DELTA": "Δ",
    "LETT_CAP_NORM.EPSILON": "Ε", "LETT_CAP_SUFF.EPSILON": "Ε", "LETT_CAP_NORM.ZETA": "Ζ", "LETT_CAP_SUFF.ZETA": "Ζ",
    "LETT_CAP_NORM.HETA": "Η", "LETT_CAP_SUFF.HETA": "Η", "LETT_CAP_NORM.THETA": "Θ", "LETT_CAP_SUFF.THETA": "Θ",
    "LETT_CAP_NORM.IOTA": "Ι", "LETT_CAP_SUFF.IOTA": "Ι", "LETT_CAP_NORM.KAPA": "Κ", "LETT_CAP_SUFF.KAPA": "Κ",
    "LETT_CAP_NORM.LAMDA": "Λ", "LETT_CAP_SUFF.LAMDA": "Λ", "LETT_CAP_NORM.MI": "Μ", "LETT_CAP_SUFF.MI": "Μ",
    "LETT_CAP_NORM.NI": "Ν", "LETT_CAP_SUFF.NI": "Ν", "LETT_CAP_NORM.KSI": "Ξ", "LETT_CAP_SUFF.KSI": "Ξ",
    "LETT_CAP_NORM.OMIKRON": "Ο", "LETT_CAP_SUFF.OMIKRON": "Ο", "LETT_CAP_NORM.PΙI": "Π", "LETT_CAP_SUFF.PII": "Π",
    "LETT_CAP_NORM.PSI": "Ψ", "LETT_CAP_SUFF.PSI": "Ψ", "LETT_CAP_NORM.RO": "Ρ", "LETT_CAP_SUFF.RO": "Ρ",
    "LETT_CAP_NORM.SIGMA": "Σ", "LETT_CAP_SUFF.SIGMA": "Σ", "LETT_CAP_NORM.TAU": "Τ", "LETT_CAP_SUFF.TAU": "Τ",
    "LETT_CAP_NORM.YPSILON": "Υ", "LETT_CAP_SUFF.YPSILON": "Υ", "LETT_CAP_NORM.FI": "Φ", "LETT_CAP_SUFF.FI": "Φ",
    "LETT_CAP_NORM.XI": "Χ", "LETT_CAP_SUFF.XI": "Χ", "LETT_CAP_NORM.OMEGA": "Ω", "LETT_CAP_SUFF.OMEGA": "Ω"
}

# Scan datasets
ds1_root = os.path.join(TEMP_DIR, 'Dataset1', 'Greek Alphabets Dataset',
                        'Working Dataset_split (70-15-15)')
if os.path.exists(ds1_root):
    for root, dirs, files in os.walk(ds1_root):
        folder = os.path.basename(root).lower()
        if folder in map_ds1:
            char = map_ds1[folder]
            if char not in char_db: char_db[char] = []
            for f in files:
                if f.endswith(('jpg', 'png', 'jpeg')):
                    char_db[char].append(os.path.join(root, f))

ds2_root = os.path.join(TEMP_DIR, 'Dataset2', 'Query')
if os.path.exists(ds2_root):
    for root, dirs, files in os.walk(ds2_root):
        folder = os.path.basename(root)
        if folder in map_ds2:
            char = map_ds2[folder]
            if char not in char_db: char_db[char] = []
            for f in files:
                if f.lower().endswith(('bmp', 'jpg', 'png')):
                    char_db[char].append(os.path.join(root, f))

print(f"Characters loaded: {len(char_db)}")


# =============================================================================
# 4.5. SYNTHETIC FINAL SIGMA GENERATOR (ς) 
# =============================================================================
def crop_char_img_helper(img):
    """Tight crop helper specifically for this section"""
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img

def apply_elastic_transform(image, alpha=1000, sigma=30):
    """
    Applies elastic distortion to make printed text look wobbly/handwritten.
    """
    try:
        from scipy.ndimage import map_coordinates, gaussian_filter

        img_arr = np.array(image)
        shape = img_arr.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        if len(shape) == 3: # RGB
            distorted = np.zeros_like(img_arr)
            for i in range(3):
                distorted[:,:,i] = map_coordinates(img_arr[:,:,i], indices, order=1, mode='reflect').reshape(shape[:2])
        else:
            distorted = map_coordinates(img_arr, indices, order=1, mode='reflect').reshape(shape)

        return Image.fromarray(distorted)
    except ImportError:
        return image.rotate(random.uniform(-5, 5))

def create_synthetic_final_sigmas(database, fonts, output_folder, count=300):
    print("Generating synthetic handwritten 'ς' (Final Sigma)...")

    # Αν υπάρχουν ήδη αρκετά αληθινά, δεν φτιάχνουμε ψεύτικα
    if 'ς' in database and len(database['ς']) > 10:
        print("Real 'ς' already exists in database. Skipping generation.")
        return database

    os.makedirs(output_folder, exist_ok=True)
    database['ς'] = []

    if not fonts:
        print("No fonts available to generate 'ς'.")
        return database

    created = 0
    pbar = tqdm(total=count, desc="Synthesizing ς")

    while created < count:
        font_path = random.choice(fonts)
        font_size = random.randint(50, 80)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            continue

        # 1. Render 'ς'
        text = "ς"
        bbox = font.getbbox(text)
        if not bbox: continue

        w = bbox[2] - bbox[0] + 40
        h = bbox[3] - bbox[1] + 40

        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20-bbox[1]), text, font=font, fill=(0, 0, 0))

        img_arr = np.array(img.convert('L'))
        if np.sum(img_arr < 250) < 50: continue

        # 2. Apply Effects
        # A. Elastic Distortion
        alpha_val = random.randint(800, 1200)
        sigma_val = random.randint(12, 18)
        img = apply_elastic_transform(img.convert('L'), alpha=alpha_val, sigma=sigma_val).convert('RGB')

        # B. Ink Bleed
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        thresh = 150
        fn = lambda x : 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1').convert('RGB')

        # C. Rotation
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor='white', expand=True)

        # D. Crop
        img = crop_char_img_helper(img)

        # 3. Save
        fname = f"synth_sigma_{created:05d}.jpg"
        save_path = os.path.join(output_folder, fname)
        img.save(save_path)

        database['ς'].append(save_path)
        if 'synthetic_meta' in globals():
            synthetic_meta[save_path] = True

        created += 1
        pbar.update(1)

    pbar.close()
    return database

# --- EXECUTE ---
TEMP_SIGMA_DIR = os.path.join(TEMP_DIR, 'synthetic_sigmas')
char_db = create_synthetic_final_sigmas(char_db, greek_fonts, TEMP_SIGMA_DIR, count=500)

# =============================================================================
# 5. ACCENT GENERATION
# =============================================================================
def add_strong_accent(img, is_upper=False):
    img = img.convert('RGB')
    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    if not bbox:
        return img

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    stroke_width = max(3, int(height * 0.08)) 
    stroke_len = int(width * 0.3)

    ink_color = (0, 0, 0)  # Pure black

    draw = ImageDraw.Draw(img)

    if is_upper:  # Κεφαλαία (Ά)
        # Τόνος στα αριστερά-πάνω
        start_x = max(3, left - stroke_len - 3)
        start_y = max(3, top + int(height * 0.15))
        end_x = start_x + stroke_len
        end_y = start_y - stroke_len

        if end_y < 0: end_y = 0

    else:  # Μικρά (ά)
        # Τόνος πάνω από το κέντρο
        center_x = left + (width // 2)
        start_x = center_x - int(stroke_len * 0.2)
        start_y = max(3, top - 5)  # Πιο ψηλά
        end_x = center_x + int(stroke_len * 0.5)
        end_y = start_y - stroke_len

        if end_y < 0: end_y = 0

    for i in range(2):  # 2 passes για πιο έντονο
        draw.line(
            [(start_x + i, start_y), (end_x + i, end_y)],
            fill=ink_color,
            width=stroke_width
        )

    return img

def generate_accented_chars(database, output_folder):
    """Δημιουργία τονισμένων χαρακτήρων"""

    vowels_map = {
        'α': 'ά', 'ε': 'έ', 'η': 'ή', 'ι': 'ί', 'ο': 'ό', 'υ': 'ύ', 'ω': 'ώ',
        'Α': 'Ά', 'Ε': 'Έ', 'Η': 'Ή', 'Ι': 'Ί', 'Ο': 'Ό', 'Υ': 'Ύ', 'Ω': 'Ώ'
    }

    count = 0
    synthetic_meta = {}

    for clean_char, accented_char in vowels_map.items():
        if clean_char in database:
            if accented_char not in database:
                database[accented_char] = []

            for img_path in database[clean_char]:
                try:
                    img = Image.open(img_path).convert('RGB')
                    is_upper = clean_char.isupper()

                    # Use improved accent function
                    accented_img = add_strong_accent(img, is_upper)

                    fname = f"acc_{count:05d}_{os.path.basename(img_path)}"
                    save_path = os.path.join(output_folder, fname)
                    accented_img.save(save_path)

                    database[accented_char].append(save_path)
                    synthetic_meta[save_path] = True
                    count += 1
                except Exception as e:
                    continue

    print(f"Created {count} accented characters")
    return database, synthetic_meta

char_db, synthetic_meta = generate_accented_chars(char_db, TEMP_ACCENTS_DIR)

# =============================================================================
# 6. VOCABULARY & HELPERS
# =============================================================================
def normalize_word(w):
    w = unicodedata.normalize("NFC", w)
    w = w.replace("ς ", "σ ")
    return w.strip()

FALLBACK_WORDS = ["καλημέρα", "κόσμος", "ιατρική", "εξέταση", "ασθενής", "και", "να", "το"]

real_words = []
try:
    print("Downloading dictionary...")
    url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/el/el_50k.txt"
    r = requests.get(url, timeout=15)

    if r.status_code == 200:
        lines = r.text.split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                w = normalize_word(parts[0])
                if len(w) >= 1 and all(c.isalpha() or c in "-άέήίόύώϊϋΐΰς" for c in w):
                     if any('\u0370' <= c <= '\u03ff' or '\u1f00' <= c <= '\u1fff' for c in w):
                        real_words.append(w)
        print(f"Vocab: {len(real_words)} words")
    else:
        real_words = FALLBACK_WORDS

except:
    real_words = FALLBACK_WORDS

if not real_words: real_words = FALLBACK_WORDS

def generate_greek_pseudoword():
    consonants = list("βγδζθκλμνξπρστφχψ")
    vowels = list("αεηιουω")
    accented = list("άέήίόύώ")
    length = random.randint(2, 5)
    word = ""
    has_accent = False
    for i in range(length):
        if random.random() > 0.3: word += random.choice(consonants)
        if not has_accent and random.random() > 0.6:
            word += random.choice(accented); has_accent = True
        else: word += random.choice(vowels)
        if random.random() > 0.8: word += random.choice(consonants)
    if word.endswith('σ'): word = word[:-1] + 'ς'
    if not has_accent and len(word) > 1: word = word[:-1] + 'ά'
    return word

def to_upper_with_accents(text):
    mapping = {'ά':'Ά', 'έ':'Έ', 'ή':'Ή', 'ί':'Ί', 'ό':'Ό', 'ύ':'Ύ', 'ώ':'Ώ'}
    return "".join([mapping.get(c, c.upper()) for c in text])


basic_vocab = [
    "και", "ή", "αλλά", "όμως", "ενώ", "δεν", "να", "στο", "στη", "στον", "στα",
    "του", "της", "των", "με", "σε", "από", "για", "ως", "είναι", "ήταν", "έχει", "έχουν",
    "αυτό", "αυτή", "αυτά", "εκεί", "εδώ", "πολύ", "λίγο", "αρκετά", "πιο", "λιγότερο",
    "καθόλου", "πάντα", "συχνά", "μερικές φορές", "σπάνια", "σήμερα", "χθες", "αύριο",
    "κατά τη διάρκεια", "μετά", "πριν", "όταν", "αν", "ώστε", "λόγω", "χωρίς", "μέσα", "έξω",
    "επίσης", "ακόμη", "παρόλο", "διότι", "καθώς", "μην", "θα", "προς", "κατά", "μεταξύ",
    "παρά", "τον", "την", "τα", "το", "προς", "απέναντι", "εντός", "εκτός",
    "θα είναι", "ήταν", "είχε", "είχαν", "αυτός", "αυτοί", "αυτές", "εκείνος", "εκείνη",
    "αρκετός", "ελάχιστος", "υπερβολικά", "περισσότερο", "πολύ λιγότερο", "διόλου",
    "ποτέ", "σχεδόν πάντα", "κάποτε", "συνήθως", "σπανίως", "προχθές", "μεθαύριο", "τώρα",
    "κατά τη διάρκεια", "έπειτα", "προηγουμένως", "εφόσον", "εάν", "προκειμένου",
    "εξαιτίας", "δίχως", "εντός", "απέξω"
]

behavior_vocab = [
    "ήρεμος", "ήρεμη", "ήρεμο", "ανήσυχος", "ανήσυχη", "υπερκινητικότητα", "παρορμητικότητα",
    "επιθετικότητα", "απόσυρση", "συνεργάσιμος", "συνεργάσιμη", "αντιδραστικός", "αντιδραστική",
    "αδιάφορος", "αδιάφορη", "φοβισμένος", "αγχωμένος", "χαρούμενος", "λυπημένος",
    "εκνευρισμένος", "ευέξαπτος", "συγκεντρωμένος", "αποσπασμένος", "παρατηρητικός",
    "απρόσεκτος", "ελεγχόμενη συμπεριφορά", "δυσκολία αυτορρύθμισης", "ανεκτικός", "ανυπόμονος",
    "δείχνει επιμονή", "εμφανίζει ένταση", "ήρεμα", "γαλήνιος", "γαλήνια", "ανήσυχα",
    "ταραγμένος", "ταραγμένη", "υπερδραστηριότητα", "ανυπομονησία", "επιθετική συμπεριφορά",
    "κοινωνική απομόνωση", "συνεργατικός", "συνεργατική", "αμυντικός", "αμυντική",
    "απαθής", "απαθές", "ανασφαλής", "νευρικός", "ευδιάθετος", "κατηφής", "θυμωμένος",
    "ευερέθιστος", "προσεκτικός", "διάσπαση προσοχής", "επιφυλακτικός", "αφηρημένος",
    "αυτοέλεγχος", "έλλειψη ελέγχου συμπεριφοράς", "υπομονετικός", "βιαστικός",
    "επίμονος στην προσπάθεια", "δείχνει αναστάτωση", "εσωστρεφής", "εξωστρεφής",
    "σταθερή διάθεση", "ευμετάβλητη διάθεση"
]

response_vocab = [
    "ανταποκρίνεται", "δεν ανταποκρίνεται", "ακολουθεί οδηγίες", "δυσκολεύεται",
    "επικοινωνεί", "δεν επικοινωνεί", "μιλά", "δεν μιλά", "βλεμματική επαφή",
    "δείχνει ενδιαφέρον", "συμμετέχει", "αποφεύγει", "απαντά", "δείχνει κατανόηση",
    "χρειάζεται βοήθεια", "χρειάζεται υπενθύμιση", "ανταποκρίνεται με καθυστέρηση",
    "ανταποκρίνεται λεκτικά", "ανταποκρίνεται μη λεκτικά", "παραμένει στο έργο",
    "εγκαταλείπει τη δραστηριότητα", "ακολουθεί ρουτίνα", "δυσκολεύεται στη μετάβαση",
    "απαιτεί καθοδήγηση", "ανταπόκριση θετική", "μη ανταπόκριση", "εκτελεί εντολές",
    "αντιμετωπίζει δυσκολία", "αλληλεπιδρά", "περιορισμένη επικοινωνία", "λεκτική έκφραση",
    "σιωπηλός", "οπτική επαφή", "εκδηλώνει περιέργεια", "ενεργή συμμετοχή",
    "αποστασιοποιείται", "παρέχει απαντήσεις", "καταλαβαίνει", "απαιτεί υποστήριξη",
    "χρειάζεται επανάληψη", "καθυστερημένη αντίδραση", "προφορική απάντηση",
    "χειρονομίες και νεύματα", "επιμένει στο καθήκον", "διακόπτει την εργασία",
    "τηρεί το πρόγραμμα", "δυσκολία στις αλλαγές", "χρειάζεται κατεύθυνση",
    "αυθόρμητη ανταπόκριση", "αργή επεξεργασία", "χρειάζεται οπτική υποστήριξη",
    "απαιτεί επανάληψη οδηγιών"
]

family_vocab = [
    "γονείς", "μητέρα", "πατέρας", "οικογένεια", "συνεργασία με γονείς",
    "υποστήριξη από γονείς", "σχολείο", "εκπαιδευτικός", "θεραπευτής", "λογοθεραπευτής",
    "εργοθεραπευτής", "ψυχολόγος", "οικογενειακό περιβάλλον", "συμμετοχή οικογένειας",
    "επικοινωνία με γονείς", "καθημερινή ρουτίνα", "δομή στο σπίτι", "πλαίσιο τάξης",
    "ομαδική δραστηριότητα", "ατομική παρέμβαση", "κηδεμόνες", "μαμά", "μπαμπάς",
    "οικογενειακό σύστημα", "γονεϊκή συνεργασία", "οικογενειακή υποστήριξη",
    "εκπαιδευτικό ίδρυμα", "δάσκαλος", "ειδικός θεραπευτής", "ειδικός λόγου",
    "επαγγελματίας εργοθεραπείας", "παιδοψυχολόγος", "σπιτικό περιβάλλον",
    "οικογενειακή εμπλοκή", "επαφή με οικογένεια", "ημερήσιο πρόγραμμα",
    "οργάνωση οικογένειας", "σχολικό πλαίσιο", "ομαδική εργασία",
    "εξατομικευμένη θεραπεία", "αδέλφια", "παππούς", "γιαγιά", "οικογενειακή δυναμική",
    "γονεϊκή εκπαίδευση", "σχολική κοινότητα", "νηπιαγωγός"
]

assessment_vocab = [
    "αξιολόγηση", "επανεκτίμηση", "παρατήρηση", "καταγραφή", "δεξιότητες",
    "αναπτυξιακές δεξιότητες", "δυσκολίες", "ανάγκες", "στόχοι", "παρέμβαση",
    "πρώιμη παρέμβαση", "πρόοδος", "βελτίωση", "καθυστέρηση λόγου", "κινητική ανάπτυξη",
    "κοινωνική αλληλεπίδραση", "γνωστική ανάπτυξη", "γλωσσική ανάπτυξη",
    "συναισθηματική ανάπτυξη", "λειτουργικές δεξιότητες", "αυτονομία",
    "επίπεδο λειτουργικότητας", "δείκτες προόδου", "παρατηρούμενη συμπεριφορά",
    "συμπερασματικά ευρήματα", "προτεινόμενοι στόχοι", "εξατομικευμένο πρόγραμμα",
    "διαγνωστική αξιολόγηση", "συνεχής παρακολούθηση", "κλινική παρατήρηση", "τεκμηρίωση",
    "ικανότητες", "αναπτυξιακά ορόσημα", "προκλήσεις", "θεραπευτικές ανάγκες",
    "θεραπευτικοί στόχοι", "θεραπευτική παρέμβαση", "έγκαιρη παρέμβαση", "εξέλιξη",
    "ανάπτυξη", "λογοπεδικά προβλήματα", "ψυχοκινητική ανάπτυξη", "κοινωνικές δεξιότητες",
    "νοητική ανάπτυξη", "γλωσσικές ικανότητες", "ψυχοσυναισθηματική εξέλιξη",
    "καθημερινές δεξιότητες", "ανεξαρτησία", "λειτουργική ικανότητα", "ενδείξεις προόδου",
    "συμπεριφορικά χαρακτηριστικά", "διαγνωστικά ευρήματα", "θεραπευτικοί στόχοι",
    "εξατομικευμένη θεραπεία", "διάγνωση", "θεραπευτικό σχέδιο",
    "αναπτυξιακή καθυστέρηση", "πρώιμα σημεία", "βραχυπρόθεσμοι στόχοι",
    "μακροπρόθεσμοι στόχοι", "συνολική εικόνα", "περιοδική επανεκτίμηση"
]

general_extra_vocab = [
    # Καθημερινές έννοιες
    "σπίτι", "δωμάτιο", "παράθυρο", "πόρτα", "τραπέζι", "καρέκλα", "δρόμος", "πλατεία",
    "πόλη", "χωριό", "γειτονιά", "αγορά", "μαγαζί", "γραφείο", "αίθουσα", "αυλή",
    # Χρόνος & ρυθμός
    "πρωί", "μεσημέρι", "βράδυ", "νύχτα", "στιγμή", "διάστημα", "περίοδος",
    "ρυθμός", "παύση", "συνέχεια", "αρχή", "τέλος",
    # Ρήματα
    "κινείται", "παραμένει", "αλλάζει", "εμφανίζεται", "εξαφανίζεται",
    "πλησιάζει", "απομακρύνεται", "εξελίσσεται", "παρατηρεί", "αντιλαμβάνεται",
    "δοκιμάζει", "αποφασίζει", "επιλέγει", "αναζητά", "συνεχίζει",
    # Επίθετα
    "σταθερός", "σταθερή", "σταθερό",
    "τυχαίος", "τυχαία", "τυχαίο",
    "σύντομος", "σύντομη", "σύντομο",
    "διαφορετικός", "διαφορετική", "διαφορετικό",
    "απλός", "απλή", "απλό",
    # Αφηρημένες έννοιες
    "έννοια", "διαφορά", "παράδειγμα", "περίπτωση", "διαδικασία",
    "αποτέλεσμα", "σχέση", "επίδραση", "αιτία", "παρατήρηση",
    # Σπάνια γράμματα (ξ ψ ζ)
    "εξέλιξη", "σύμπτωση", "ψυχή", "ψίθυρος", "ζώνη", "ζυγός",
    "οξυγόνο", "άξονας", "παράξενη", "σύγχυση", "συνδυασμός",
    # Συνθετικές αλλά ρεαλιστικές λέξεις OCR-wise
    "μεταβολή", "αντίδραση", "κατεύθυνση", "προσέγγιση", "αποτύπωση",
    "διαμόρφωση", "ενίσχυση", "κατανομή", "συσχέτιση", "αλληλουχία",
    # Μικρές φράσεις
    "σε εξέλιξη", "χωρίς αλλαγή", "με μικρή διαφορά",
    "σε σταθερό ρυθμό", "μετά από λίγο",
    "σε τυχαία σειρά", "κατά προσέγγιση",
    # Ουδέτερες λέξεις
    "μονάδα", "σύνολο", "στοιχείο", "τιμή", "μέγεθος",
    "πλαίσιο", "δομή", "μορφή", "τύπος", "κατηγορία"
]


full_vocab = basic_vocab + behavior_vocab + response_vocab + family_vocab + assessment_vocab + general_extra_vocab


def clean_polytonic(text):
    """Καθαρισμός πολυτονικού"""
    if not isinstance(text, str): return ""
    norm = unicodedata.normalize('NFD', text)
    result = []
    for c in norm:
        if unicodedata.category(c) != 'Mn':
            result.append(c)
        elif ord(c) in [0x0301, 0x0300, 0x0342]:
            result.append('\u0301')
    return unicodedata.normalize('NFC', "".join(result))

def remove_accents(text):
    """Αφαίρεση τόνων"""
    norm = unicodedata.normalize('NFD', text)
    return "".join(c for c in norm if unicodedata.category(c) != 'Mn')

def get_text_style(vocab_list):
    """Επιλογή στυλ κειμένου"""
    words = random.choices(vocab_list, k=random.randint(2, 4))
    phrase = " ".join(words)
    clean_phrase = clean_polytonic(phrase)

    style = random.choices(['lower', 'sentence', 'title', 'upper'],
                          weights=[40, 30, 10, 20])[0]

    if style == 'lower':
        return clean_phrase.lower()
    elif style == 'upper':
        return remove_accents(clean_phrase).upper()
    elif style == 'sentence':
        first = remove_accents(clean_phrase[0]).upper()
        return first + clean_phrase[1:].lower()
    else:  # title
        words_list = []
        for w in clean_phrase.split():
            if w:
                first = remove_accents(w[0]).upper()
                words_list.append(first + w[1:].lower())
        return " ".join(words_list)

# =============================================================================
# 7. IMAGE GENERATORS
# =============================================================================
def crop_char_img(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img

def generate_printed(text, font_path):
    font_size = random.randint(42, 58)

    try:
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(text)
    except:
        return None

    left, top, right, bottom = bbox
    w = right - left + 40
    h = bottom - top + 40

    if w < 10 or h < 10:
        return None

    img = Image.new('RGB', (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    txt_x = 20
    txt_y = 20 - top
    draw.text((txt_x, txt_y), text, font=font, fill=(0, 0, 0))

    # VALIDATE: Check if actual text was rendered
    img_array = np.array(img.convert('L'))
    non_white_pixels = np.sum(img_array < 250)

    # If < 50 pixels, it's probably boxes
    if non_white_pixels < 50:
        return None

    # Rotation
    if random.random() > 0.5:
        angle = random.uniform(-1.5, 1.5)
        img = img.rotate(angle, expand=True, fillcolor='white')

    # Blur
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(0.3))

    return img

def generate_handwritten_line(text, difficulty='easy'):
    """Fixed handwritten line generator"""
    imgs = []
    has_synth = False

    for char in text:
        if char == ' ':
            imgs.append('SPACE')
            continue

        # Find character
        candidates = char_db.get(char)
        if not candidates:
            base = remove_accents(char)
            candidates = char_db.get(base)

        if candidates:
            img_path = random.choice(candidates)
            if img_path in synthetic_meta:
                has_synth = True

            try:
                char_img = Image.open(img_path).convert('RGB')
                char_img = crop_char_img(char_img)

                target_h = 60 if difficulty == 'easy' else random.randint(55, 70)
                ratio = target_h / char_img.size[1]
                new_w = int(char_img.size[0] * ratio)
                char_img = char_img.resize((new_w, target_h),
                                          Image.Resampling.LANCZOS)

                # Hard mode transforms
                if difficulty == 'hard':
                    angle = random.uniform(-8, 8)
                    char_img = char_img.rotate(angle, expand=True,
                                              fillcolor='white')

                imgs.append(char_img)
            except Exception as e:
                continue

    if not imgs:
        return None, False

    total_w = 0
    items = []

    for item in imgs:
        if item == 'SPACE':
            gap = random.randint(30, 50) if difficulty == 'easy' else random.randint(20, 35)
            total_w += gap
            items.append(('SPACE', gap))
        else:
            overlap = 0 if difficulty == 'easy' else random.randint(0, 6)
            total_w += item.width - overlap
            items.append((item, -overlap))

    # Canvas
    h_canvas = 120
    bg_color = (255, 255, 255)
    if difficulty == 'hard':
        bg_color = tuple(random.randint(248, 255) for _ in range(3))

    final_im = Image.new('RGB', (total_w + 60, h_canvas), bg_color)

    # Paste characters
    x = 30
    for item, val in items:
        if item == 'SPACE':
            x += val
        else:
            y_jitter = 0 if difficulty == 'easy' else random.randint(-5, 5)
            y = (h_canvas - item.height) // 2 + y_jitter
            y = max(0, min(y, h_canvas - item.height))
            final_im.paste(item, (x, y))
            x += item.width + val

    # Post-processing
    if difficulty == 'hard' and random.random() > 0.5:
        final_im = final_im.filter(ImageFilter.GaussianBlur(0.5))

    return final_im, has_synth

# =============================================================================
# 8. DATASET GENERATION
# =============================================================================
print("Generating Dataset (Split into 3)...")

dataset_structure = [
    ('printed', 80000),
    ('handwritten_easy', 60000),
    ('handwritten_hard', 120000)
]

# Ξεχωριστές λίστες για τα metadata του κάθε dataset
data_printed = []
data_easy = []
data_hard = []

failures = 0

for phase_name, num_samples in dataset_structure:
    print(f"\n Phase: {phase_name} ({num_samples} samples)")

    # Επιλογή του σωστού φακέλου και λίστας αποθήκευσης
    if phase_name == 'printed':
        current_save_dir = DIR_PRINTED
        current_list = data_printed
    elif phase_name == 'handwritten_easy':
        current_save_dir = DIR_EASY
        current_list = data_easy
    else: # hard
        current_save_dir = DIR_HARD
        current_list = data_hard

    for i in tqdm(range(num_samples)):
        # --- 1. TEXT GENERATION ---
        if random.random() < 0.05:  # 10% gibberish
            chars = list(char_db.keys())
            text_label = "".join(random.choices(chars, k=random.randint(4, 8)))
        else:
            # ΕΠΙΛΟΓΗ ΛΕΞΙΚΟΥ ΑΝΑΛΟΓΑ ΜΕ ΤΗ ΦΑΣΗ
            if phase_name == 'handwritten_hard':
                num_words = random.randint(2, 5)

                # 50% πιθανότητα να πάρει από το γενικό λεξικό
                # 50% πιθανότητα να πάρει από το ιατρικό λεξιλόγιο
                if random.random() < 0.4:
                    words = random.choices(real_words, k=num_words)
                else:
                    words = random.choices(full_vocab, k=num_words)

            elif phase_name == 'printed':
                # Printed: Μικρό ελεγχόμενο λεξιλόγιο
                num_words = random.randint(3, 5)
                words = random.choices(full_vocab, k=num_words)

            else: # handwritten_easy
                # Easy: Μικρό ελεγχόμενο λεξιλόγιο
                num_words = random.randint(2, 5)
                words = random.choices(full_vocab, k=num_words)

            raw_text = " ".join(words)

            # Εφαρμογή Στυλ (Lower, Upper, Title)
            style = random.choices(['lower', 'sentence', 'title', 'upper'], weights=[40, 30, 10, 20])[0]

            if style == 'lower':
                text_label = clean_polytonic(raw_text).lower()
            elif style == 'upper':
                text_label = remove_accents(clean_polytonic(raw_text)).upper()
            elif style == 'sentence':
                tmp = clean_polytonic(raw_text)
                text_label = remove_accents(tmp[0]).upper() + tmp[1:].lower() if tmp else ""
            else: # title
                text_label = clean_polytonic(raw_text).title()

        # --- 2. IMAGE GENERATION ---
        fname = f"{phase_name}_{i:05d}.jpg"
        img = None
        used_synth = False

        try:
            if phase_name == 'printed':
                if greek_fonts:
                    fpath = random.choice(greek_fonts)
                    img = generate_printed(text_label, fpath)
            elif phase_name == 'handwritten_easy':
                img, used_synth = generate_handwritten_line(text_label, 'easy')
            else:  # hard
                img, used_synth = generate_handwritten_line(text_label, 'hard')

            if img:
                save_path = os.path.join(current_save_dir, 'images', fname)
                img.save(save_path, quality=95)

                current_list.append({
                    'file_name': f"images/{fname}",
                    'text': text_label,
                    'phase': phase_name,
                    'has_synth_accent': used_synth
                })
            else:
                failures += 1

        except Exception as e:
            failures += 1
            continue


print(f"\n Total failures: {failures}")

# =============================================================================
# 9. SAVE & ZIP (ΤΡΟΠΟΠΟΙΗΜΕΝΟ ΓΙΑ 3 ZIP)
# =============================================================================
print("\n Saving separate datasets...")

# Λίστα με τα ζεύγη (Φάκελος, Λίστα Δεδομένων, Τελικό Zip Path)
outputs = [
    (DIR_PRINTED, data_printed, ZIP_PATH_PRINTED, 'Printed'),
    (DIR_EASY, data_easy, ZIP_PATH_EASY, 'Easy Handwritten'),
    (DIR_HARD, data_hard, ZIP_PATH_HARD, 'Hard Handwritten')
]

for dir_path, data_list, zip_path, name in outputs:
    if len(data_list) > 0:
        print(f"\n Processing {name}...")

        df = pd.DataFrame(data_list)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True) 

        csv_path = os.path.join(dir_path, 'train.csv')
        df.to_csv(csv_path, index=False)

        print(f"   Samples: {len(df)}")
        print(f"   Zipping to: {zip_path}")

        base_name = os.path.splitext(zip_path)[0] 
        shutil.make_archive(base_name, 'zip', dir_path)

        # Verify
        if os.path.exists(zip_path):
            print(f"Saved successfully.")
        else:
            local_zip = base_name.split('/')[-1] + '.zip'
            if os.path.exists(local_zip):
                 shutil.move(local_zip, zip_path)
                 print(f"Moved to Drive successfully.")
    else:
        print(f"Skipping {name} (No data generated)")

print(f"\n ALL DONE!")
