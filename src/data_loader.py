import pandas as pd
import os

def load_fitlife_data():
    """Load FitLife dataset from local file"""
    try:
        if os.path.exists('data/health_fitness_dataset.csv'):
            print("✓ Found local dataset file, loading...")
            df = pd.read_csv('data/health_fitness_dataset.csv')
            print(f"✓ Dataset loaded successfully: {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            return df
        else:
            raise FileNotFoundError("Dataset not found in data/health_fitness_dataset.csv")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def get_dataset_info(df):
    """Display comprehensive dataset information"""
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    
    print(f"\n📊 Dataset Shape:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    print(f"\n📋 Column Names and Types:")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        print(f"   {i:2d}. {col:30s} - {dtype}")
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   ✓ No missing values found!")
    
    print(f"\n📈 Numerical Columns Statistics:")
    print(df.describe())
    
    print(f"\n🔤 Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"   {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"      Values: {df[col].unique().tolist()}")
    
    return df

if __name__ == "__main__":
    print("="*70)
    print("HEALTH ACTIVITY PREDICTOR - DATA LOADER")
    print("="*70)
    
    df = load_fitlife_data()
    df = get_dataset_info(df)
    
    print(f"\n🔍 Sample Data (First 5 rows):")
    print(df.head())
    
    print(f"\n✅ Data loading completed successfully!")
