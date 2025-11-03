# 🚀 Quick Deploy - Disease Prediction App

## ✅ Essential Files (DO NOT DELETE)

### Application
- `disease_prediction.py` - Main app
- `requirements.txt` - Dependencies
- `packages.txt` - System packages
- `setup.sh` - SpaCy model download
- `.streamlit/config.toml` - App settings

### Model & Data (resources/)
- `dataset_curated.csv` - Training data (71 diseases)
- `mlp_model_curated.h5` - Neural network model
- `label_index_curated.json` - Disease labels
- `symptom_index_curated.json` - Symptom mapping

### OCR (tessdata/)
- `eng.traineddata` - English OCR model
- Keep other `.traineddata` files if you need multi-language support

---

## 🗑️ Safe to Delete

- All `.exe` files (tesseract executables)
- All `.dll` files (Windows libraries)
- All `.html` files (Tesseract documentation)
- `__pycache__/`, `.venv/`, `resources/backup_*/`
- Development scripts: `train_*.py`, `test_*.py`, `check_*.py`, etc.
- Unused models: `mlp_model.h5`, `dataset_kaggle.csv`, `logistic_model.joblib`

---

## 🎯 Deploy to Streamlit Cloud (FREE)

### Step 1: Cleanup
```powershell
.\cleanup_for_deployment.ps1
```

### Step 2: Create GitHub Repo
```bash
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://github.com/YOUR_USERNAME/disease-prediction.git
git push -u origin main
```

### Step 3: Deploy
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `disease_prediction.py`
6. Click "Deploy"

**Done! Your app will be live at: `yourapp.streamlit.app`**

---

## 📊 Final Size
- Before cleanup: ~500+ MB
- After cleanup: ~150-200 MB
- Easily fits in Streamlit Cloud's 1 GB limit

---

## 🔧 Troubleshooting

**Tesseract not found?**
→ Check `packages.txt` exists with `tesseract-ocr`

**spaCy model error?**
→ Check `setup.sh` exists and runs `python -m spacy download en_core_web_sm`

**App crashes?**
→ Check logs in Streamlit Cloud dashboard

---

## 📚 Full Guide
See `DEPLOYMENT_GUIDE.md` for complete instructions and alternative platforms.

---

**Support:** https://discuss.streamlit.io
