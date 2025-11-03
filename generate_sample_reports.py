"""Generate synthetic report images for OCR/testing.

Creates resources/sample_reports/ with several PNG files:
 - report_clean.png
 - report_rotated.png
 - report_noise.png
 - report_partial.png

The text includes symptom names from the app so OCR+matching can be tested.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, 'resources', 'sample_reports')
os.makedirs(OUT_DIR, exist_ok=True)

# Use a default PIL font. If a TTF is available, you can set FONT_PATH
try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except Exception:
    FONT = ImageFont.load_default()

# Symptoms to include (choose names that appear in disease_prediction.py symptoms_list)
SYMPTOMS = [
    'Headache',
    'Mild fever',
    'Nausea',
    'Fatigue',
    'Loss of smell'
]

EXTRA = [
    'Patient Name: John Doe',
    'Date: 2025-10-13',
    'Notes: Blood pressure stable. See lab results attached.'
]

BASE_TEXT = "\n".join(EXTRA + ["Reported symptoms:"] + [f"- {s}" for s in SYMPTOMS])

# helper to render text to image
def text_to_image(text, size=(1200, 1600), bgcolor=(255,255,255), fill=(0,0,0)):
    img = Image.new('RGB', size, color=bgcolor)
    draw = ImageDraw.Draw(img)
    margin = 40
    offset = margin
    for line in text.split('\n'):
        draw.text((margin, offset), line, font=FONT, fill=fill)
        # Use textbbox to compute height in a Pillow-version-compatible way
        try:
            bbox = draw.textbbox((margin, offset), line, font=FONT)
            line_height = bbox[3] - bbox[1]
        except Exception:
            # Fallback to textsize for very old Pillow versions
            line_width, line_height = draw.textsize(line, font=FONT)
        offset += line_height + 8
    return img

# 1) clean report
img_clean = text_to_image(BASE_TEXT)
img_clean.save(os.path.join(OUT_DIR, 'report_clean.png'))

# 2) rotated report (simulate a tilted scan)
img_rot = img_clean.rotate(3, expand=True, fillcolor=(255,255,255))
# crop back to similar aspect
w,h = img_rot.size
crop = img_rot.crop((10,10,w-10,h-10)).resize(img_clean.size)
crop.save(os.path.join(OUT_DIR, 'report_rotated.png'))

# 3) noisy report (add speckle noise and blur)
arr = np.array(img_clean).astype(np.float32)
noise = np.random.normal(0, 12, arr.shape)
noisy = arr + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
img_noisy = Image.fromarray(noisy)
img_noisy = img_noisy.filter(ImageFilter.GaussianBlur(radius=0.6))
img_noisy.save(os.path.join(OUT_DIR, 'report_noise.png'))

# 4) partial report - only 2 symptoms (to test insufficient symptom detection)
partial_symptoms = SYMPTOMS[:2]
partial_text = "\n".join(EXTRA + ["Reported symptoms:"] + [f"- {s}" for s in partial_symptoms])
img_partial = text_to_image(partial_text)
img_partial.save(os.path.join(OUT_DIR, 'report_partial.png'))

print('Generated sample reports in', OUT_DIR)
