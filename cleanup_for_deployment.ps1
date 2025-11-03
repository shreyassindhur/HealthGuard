# Cleanup Script for Deployment
# Run this to remove unnecessary files before deploying

Write-Host "🧹 Cleaning up project for deployment..." -ForegroundColor Cyan

# Remove development scripts
$devScripts = @(
    "check_dataset.py",
    "compare_datasets.py",
    "expand_dataset.py",
    "generate_sample_reports.py",
    "ocr_predict.py",
    "show_test_cases.py",
    "switch_dataset.py",
    "test_symptoms.py",
    "train_curated_model.py",
    "train_image_model.py",
    "train_model.py",
    "main.ipynb"
)

foreach ($file in $devScripts) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "✅ Removed: $file" -ForegroundColor Green
    }
}

# Remove all .exe files
Write-Host "`n🗑️  Removing executables..." -ForegroundColor Yellow
Get-ChildItem -Filter "*.exe" | Remove-Item -Force
Write-Host "✅ Removed all .exe files" -ForegroundColor Green

# Remove all .dll files
Write-Host "`n🗑️  Removing DLL files..." -ForegroundColor Yellow
Get-ChildItem -Filter "*.dll" | Remove-Item -Force
Write-Host "✅ Removed all .dll files" -ForegroundColor Green

# Remove all .html files (Tesseract docs)
Write-Host "`n🗑️  Removing HTML documentation..." -ForegroundColor Yellow
Get-ChildItem -Filter "*.html" | Remove-Item -Force
Write-Host "✅ Removed all .html files" -ForegroundColor Green

# Remove backup folders
Write-Host "`n🗑️  Removing backups and cache..." -ForegroundColor Yellow
if (Test-Path "resources\backup_20251031_000126") {
    Remove-Item "resources\backup_20251031_000126" -Recurse -Force
    Write-Host "✅ Removed backup folder" -ForegroundColor Green
}

if (Test-Path "resources\.ipynb_checkpoints") {
    Remove-Item "resources\.ipynb_checkpoints" -Recurse -Force
    Write-Host "✅ Removed .ipynb_checkpoints" -ForegroundColor Green
}

if (Test-Path "resources\sample_reports") {
    Remove-Item "resources\sample_reports" -Recurse -Force
    Write-Host "✅ Removed sample_reports" -ForegroundColor Green
}

if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    Write-Host "✅ Removed __pycache__" -ForegroundColor Green
}

if (Test-Path ".venv") {
    Remove-Item ".venv" -Recurse -Force
    Write-Host "✅ Removed .venv" -ForegroundColor Green
}

# Remove unused model files (keeping only curated versions)
Write-Host "`n🗑️  Removing unused models..." -ForegroundColor Yellow
$unusedModels = @(
    "resources\dataset_kaggle.csv",
    "resources\mlp_model.h5",
    "resources\label_index.json",
    "resources\symptom_index.json",
    "resources\logistic_model.joblib"
)

foreach ($file in $unusedModels) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "✅ Removed: $file" -ForegroundColor Green
    }
}

Write-Host "`n✨ Cleanup complete!" -ForegroundColor Cyan
Write-Host "`n📊 Project is now ready for deployment!" -ForegroundColor Green
Write-Host "`n📦 Essential files remaining:" -ForegroundColor Yellow
Write-Host "   - disease_prediction.py" -ForegroundColor White
Write-Host "   - requirements.txt" -ForegroundColor White
Write-Host "   - packages.txt" -ForegroundColor White
Write-Host "   - setup.sh" -ForegroundColor White
Write-Host "   - .streamlit/config.toml" -ForegroundColor White
Write-Host "   - resources/dataset_curated.csv" -ForegroundColor White
Write-Host "   - resources/mlp_model_curated.h5" -ForegroundColor White
Write-Host "   - resources/label_index_curated.json" -ForegroundColor White
Write-Host "   - resources/symptom_index_curated.json" -ForegroundColor White
Write-Host "   - tessdata/ (OCR language files)" -ForegroundColor White

Write-Host "`n🚀 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Review DEPLOYMENT_GUIDE.md" -ForegroundColor White
Write-Host "   2. Push to GitHub" -ForegroundColor White
Write-Host "   3. Deploy on Streamlit Cloud" -ForegroundColor White
