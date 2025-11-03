"""
Switch between original and curated datasets

This script backs up your current model files and switches between the
original Kaggle dataset and the curated medical dataset.
"""

import os
import shutil
from datetime import datetime

RESOURCES_DIR = 'resources'

FILES = {
    'original': {
        'model': 'mlp_model.h5',
        'dataset': 'dataset_kaggle.csv',
        'symptom_index': 'symptom_index.json',
        'label_index': 'label_index.json'
    },
    'curated': {
        'model': 'mlp_model_curated.h5',
        'dataset': 'dataset_curated.csv',
        'symptom_index': 'symptom_index_curated.json',
        'label_index': 'label_index_curated.json'
    }
}

def backup_files():
    """Create timestamped backups of current files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(RESOURCES_DIR, f'backup_{timestamp}')
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"📦 Creating backup in {backup_dir}/")
    
    for file_type, filename in FILES['original'].items():
        src = os.path.join(RESOURCES_DIR, filename)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, filename)
            shutil.copy2(src, dst)
            print(f"  ✓ Backed up {filename}")
    
    return backup_dir

def switch_to(dataset_type):
    """Switch to specified dataset type"""
    if dataset_type not in ['original', 'curated']:
        print(f"❌ Invalid dataset type: {dataset_type}")
        print("   Valid options: 'original', 'curated'")
        return False
    
    print(f"\n🔄 Switching to {dataset_type.upper()} dataset...")
    
    # Check if curated files exist
    if dataset_type == 'curated':
        curated_model = os.path.join(RESOURCES_DIR, FILES['curated']['model'])
        if not os.path.exists(curated_model):
            print(f"\n❌ Curated model not found at {curated_model}")
            print("   Run: python train_curated_model.py first")
            return False
    
    # Create backup first
    backup_dir = backup_files()
    
    # Copy files
    source_files = FILES[dataset_type]
    target_files = FILES['original']  # Always replace the "active" files
    
    print(f"\n📋 Copying {dataset_type} files to active location...")
    for file_type in ['model', 'dataset']:
        src = os.path.join(RESOURCES_DIR, source_files[file_type])
        dst = os.path.join(RESOURCES_DIR, target_files[file_type])
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {source_files[file_type]} → {target_files[file_type]}")
        else:
            print(f"  ⚠️  {source_files[file_type]} not found, skipping...")
    
    # Handle optional index files
    for file_type in ['symptom_index', 'label_index']:
        if file_type in source_files:
            src = os.path.join(RESOURCES_DIR, source_files[file_type])
            if os.path.exists(src):
                dst = os.path.join(RESOURCES_DIR, target_files[file_type])
                shutil.copy2(src, dst)
                print(f"  ✓ {source_files[file_type]} → {target_files[file_type]}")
    
    print(f"\n✅ Successfully switched to {dataset_type.upper()} dataset!")
    print(f"📂 Previous files backed up to: {backup_dir}")
    print(f"\n🚀 You can now run: streamlit run disease_prediction.py")
    
    return True

def show_status():
    """Show which dataset is currently active"""
    print("\n" + "="*80)
    print("CURRENT DATASET STATUS")
    print("="*80)
    
    model_path = os.path.join(RESOURCES_DIR, FILES['original']['model'])
    dataset_path = os.path.join(RESOURCES_DIR, FILES['original']['dataset'])
    
    if not os.path.exists(model_path):
        print("❌ No active model found")
        return
    
    # Check model size to guess which dataset
    import pandas as pd
    df = pd.read_csv(dataset_path)
    
    if len(df) > 100:
        print("📊 Currently using: ORIGINAL (Kaggle) dataset")
        print(f"   - {len(df)} disease records")
        print(f"   - {len(df['Disease'].unique())} unique diseases")
        print(f"   - Multiple entries per disease with varied symptoms")
    else:
        print("📊 Currently using: CURATED (Medical) dataset")
        print(f"   - {len(df)} diseases (one per disease)")
        print(f"   - Medically accurate primary symptoms")
        print(f"   - Consistent symptom sets")
    
    print(f"\n📂 Active files:")
    print(f"   - Model: {FILES['original']['model']}")
    print(f"   - Dataset: {FILES['original']['dataset']}")
    
    # Check if curated version exists
    curated_model = os.path.join(RESOURCES_DIR, FILES['curated']['model'])
    if os.path.exists(curated_model):
        print(f"\n✅ Curated dataset available")
        print(f"   To switch: python switch_dataset.py curated")
    else:
        print(f"\n⚠️  Curated dataset not yet created")
        print(f"   To create: python train_curated_model.py")
    
    print("="*80)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        show_status()
        print("\n💡 Usage:")
        print("   python switch_dataset.py original   # Switch to Kaggle dataset")
        print("   python switch_dataset.py curated    # Switch to curated dataset")
        print("   python switch_dataset.py status     # Show current status")
    else:
        command = sys.argv[1].lower()
        
        if command == 'status':
            show_status()
        elif command in ['original', 'curated']:
            switch_to(command)
        else:
            print(f"❌ Unknown command: {command}")
            print("   Valid commands: original, curated, status")
