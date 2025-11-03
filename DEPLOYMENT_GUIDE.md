# 🚀 Deployment Guide - Disease Prediction App

## 📦 Essential Files (MUST KEEP)

### Core Application Files
```
disease_prediction.py          # Main Streamlit app (REQUIRED)
requirements.txt               # Python dependencies (REQUIRED)
readme.md                      # Documentation
```

### Model & Data Files (resources/)
```
resources/
├── dataset_curated.csv        # Training data for 71 diseases (REQUIRED)
├── mlp_model_curated.h5       # Trained neural network model (REQUIRED)
├── label_index_curated.json   # Disease label mapping (REQUIRED)
├── symptom_index_curated.json # Symptom index mapping (REQUIRED)
└── tessdata/                  # Tesseract OCR language data (REQUIRED for image analysis)
    ├── eng.traineddata        # English language model
    └── (keep other .traineddata files for multi-language support)
```

**Total size estimate:** ~50-100 MB (without tessdata), ~200-300 MB (with tessdata)

---

## 🗑️ Files You Can DELETE

### 1. Development/Testing Scripts (Safe to delete)
```
check_dataset.py
compare_datasets.py
expand_dataset.py
generate_sample_reports.py
ocr_predict.py
show_test_cases.py
switch_dataset.py
test_symptoms.py
train_curated_model.py
train_image_model.py
train_model.py
main.ipynb
```

### 2. Documentation (Safe to delete, but recommended to keep)
```
DATASET_IMPROVEMENT.md
EXPLAINABILITY.md
```

### 3. Tesseract Executables & DLLs (Not needed on cloud platforms)
```
All .exe files (tesseract.exe, ambiguous_words.exe, etc.)
All .dll files (libarchive-13.dll, libb2-1.dll, etc.)
All .html files (documentation for Tesseract tools)
```
**Why?** Cloud platforms use system-installed Tesseract, not bundled executables.

### 4. Backup/Cache Folders
```
resources/backup_20251031_000126/
resources/.ipynb_checkpoints/
resources/sample_reports/
__pycache__/
.venv/
```

### 5. Unused Model Files (if using curated dataset)
```
resources/dataset_kaggle.csv
resources/mlp_model.h5
resources/label_index.json
resources/symptom_index.json
resources/logistic_model.joblib
```

---

## 🌐 FREE Deployment Options

### Option 1: **Streamlit Community Cloud** ⭐ RECOMMENDED
**Best for:** Easy deployment, automatic updates, free hosting

#### Pros:
✅ Completely FREE forever
✅ 1 GB RAM, 1 CPU core per app
✅ Direct GitHub integration
✅ Automatic redeployment on git push
✅ HTTPS included
✅ Custom subdomain (yourapp.streamlit.app)

#### Cons:
❌ Apps sleep after inactivity (wake up in ~30 seconds)
❌ Limited to 3 apps on free tier
❌ 1 GB RAM limit (enough for your app)

#### Steps:
1. **Create GitHub repository**
   ```bash
   # In your project folder
   git init
   git add disease_prediction.py requirements.txt resources/
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/disease-prediction.git
   git push -u origin main
   ```

2. **Visit:** https://streamlit.io/cloud
3. **Sign in** with GitHub
4. **Click "New app"**
5. **Select your repository**
6. **Main file path:** `disease_prediction.py`
7. **Click "Deploy"**

**Special setup needed:**
- Add `packages.txt` for Tesseract:
  ```
  tesseract-ocr
  tesseract-ocr-eng
  ```
- Add `.streamlit/config.toml` for settings:
  ```toml
  [server]
  maxUploadSize = 10
  
  [theme]
  primaryColor = "#FF4B4B"
  ```

---

### Option 2: **Hugging Face Spaces**
**Best for:** ML apps, more resources

#### Pros:
✅ FREE tier available
✅ 2 CPU cores, 16 GB RAM (generous!)
✅ Great for ML models
✅ GPU available on paid tier

#### Cons:
❌ Slightly more complex setup
❌ Apps sleep after 48h inactivity

#### Steps:
1. Create account at https://huggingface.co
2. Create new Space → Select "Streamlit"
3. Upload files via web interface or git
4. Runs automatically

---

### Option 3: **Railway.app**
**Best for:** More control, databases

#### Pros:
✅ $5 free credits per month
✅ No sleeping apps
✅ Database support
✅ Custom domains

#### Cons:
❌ Limited free credits (~500 hours/month)
❌ Requires credit card

---

### Option 4: **Render**
**Best for:** Production-ready apps

#### Pros:
✅ FREE tier available
✅ No credit card needed
✅ Auto-deploy from GitHub

#### Cons:
❌ Apps spin down after 15 min inactivity
❌ Slow cold starts (30-60 seconds)

---

## 📋 Pre-Deployment Checklist

### 1. Update requirements.txt for cloud deployment
```bash
# Create optimized requirements.txt
streamlit>=1.0
pandas>=1.3
numpy>=1.21
tensorflow>=2.8
plotly>=5.0
scikit-learn>=1.0
Pillow>=9.0
pytesseract>=0.3
spacy>=3.0
shap>=0.41
```

### 2. Download spaCy model
Add to your code or create `setup.sh`:
```bash
#!/bin/bash
python -m spacy download en_core_web_sm
```

### 3. Create `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 10
headless = true
port = 8501

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### 4. Create `packages.txt` (for Streamlit Cloud)
```
tesseract-ocr
tesseract-ocr-eng
```

### 5. Create `.gitignore`
```
__pycache__/
.venv/
*.pyc
.ipynb_checkpoints/
resources/backup_*/
resources/sample_reports/
*.exe
*.dll
```

---

## 🎯 Minimal Deployment Structure

```
disease-prediction-app/
├── disease_prediction.py           # Main app
├── requirements.txt                # Dependencies
├── packages.txt                    # System packages (Tesseract)
├── setup.sh                        # Download spaCy model
├── readme.md                       # Documentation
├── .streamlit/
│   └── config.toml                 # App configuration
├── resources/
│   ├── dataset_curated.csv
│   ├── mlp_model_curated.h5
│   ├── label_index_curated.json
│   └── symptom_index_curated.json
└── tessdata/
    └── eng.traineddata             # English OCR model
```

**Total size:** ~150-200 MB

---

## 🚀 Quick Deploy Commands

### For Streamlit Cloud (GitHub):
```bash
# 1. Clean up project
rm -rf __pycache__ .venv resources/backup_* *.exe *.dll

# 2. Initialize git
git init
git add .
git commit -m "Ready for deployment"

# 3. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/disease-prediction.git
git push -u origin main

# 4. Go to streamlit.io/cloud and deploy!
```

### For local testing:
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run locally
streamlit run disease_prediction.py
```

---

## ⚡ Performance Tips

1. **Reduce tessdata size:**
   - Keep only `eng.traineddata` if English-only
   - Delete other language files to save ~500 MB

2. **Model optimization:**
   - Your current model (~5 MB) is already optimal
   - Consider model quantization if needed

3. **Enable caching:**
   ```python
   @st.cache_resource
   def load_model():
       return tf.keras.models.load_model('resources/mlp_model_curated.h5')
   ```

4. **Optimize imports:**
   ```python
   import streamlit as st
   # Load heavy libraries only when needed
   if st.session_state.get('uploaded_image'):
       import pytesseract
   ```

---

## 📊 Platform Comparison

| Platform | Free Tier | RAM | Sleeping | Best For |
|----------|-----------|-----|----------|----------|
| **Streamlit Cloud** | ✅ Forever | 1 GB | Yes (inactive) | **Your app** ⭐ |
| Hugging Face | ✅ Forever | 16 GB | Yes (48h) | ML-heavy apps |
| Railway | ⚠️ $5/month | 8 GB | No | Always-on apps |
| Render | ✅ Forever | 512 MB | Yes (15 min) | Simple apps |
| Heroku | ❌ Paid only | - | - | Not recommended |

---

## 🎉 Recommended: Streamlit Cloud

**Why?** Your app is perfect for Streamlit Cloud:
- ✅ Streamlit-native (obviously!)
- ✅ ~200 MB total size (well under 1 GB limit)
- ✅ Simple dependencies
- ✅ No database needed
- ✅ Auto-updates from GitHub
- ✅ FREE forever

**Deploy now:** https://streamlit.io/cloud

---

## 🆘 Troubleshooting

### Issue: "Tesseract not found"
**Solution:** Add `packages.txt` with `tesseract-ocr`

### Issue: "spaCy model not found"
**Solution:** Add `setup.sh`:
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Out of memory"
**Solution:** 
- Remove unused models
- Use Hugging Face Spaces (16 GB RAM)
- Implement lazy loading

### Issue: "App is slow"
**Solution:**
- Add `@st.cache_resource` to model loading
- Optimize image processing
- Reduce model size

---

## 📞 Support

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io
- GitHub Issues: Create issues in your repo

---

**Ready to deploy? Follow the Streamlit Cloud steps above! 🚀**
