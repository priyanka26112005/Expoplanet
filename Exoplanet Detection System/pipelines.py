"""
STEP 3: Complete Data Preprocessing Pipeline
Clean, transform, and prepare data for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


class ExoplanetPreprocessor:
    """
    Complete preprocessing pipeline for exoplanet detection
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = None
        self.label_mapping = None
        
    def load_data(self, filepath):
        """Load merged dataset"""
        print("📂 Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def identify_columns(self, df):
        """Automatically identify target and feature columns"""
        print("\n🔍 Identifying columns...")
        
        # Find target column
        target_candidates = [col for col in df.columns if 'disposition' in col.lower() or 'disp' in col.lower()]
        
        if not target_candidates:
            raise ValueError("❌ No target column found! Need a 'disposition' column")
        
        self.target_column = target_candidates[0]
        print(f"   🎯 Target: {self.target_column}")
        
        # Find feature columns (numerical only, exclude identifiers)
        exclude_keywords = ['id', 'name', 'disposition', 'flag', 'date', 'kepid', 'epic', 'tic']
        
        feature_cols = []
        for col in df.columns:
            # Check if numerical
            if df[col].dtype in ['int64', 'float64']:
                # Check if not an identifier
                if not any(keyword in col.lower() for keyword in exclude_keywords):
                    feature_cols.append(col)
        
        self.feature_columns = feature_cols
        print(f"   📊 Features: {len(self.feature_columns)} numerical columns")
        
        return self.target_column, self.feature_columns
    
    def clean_target(self, df):
        """Clean and encode target variable"""
        print("\n🧹 Cleaning target variable...")
        
        # Remove rows with missing target
        initial_len = len(df)
        df = df.dropna(subset=[self.target_column])
        removed = initial_len - len(df)
        
        if removed > 0:
            print(f"   Removed {removed} rows with missing target")
        
        # Show unique values
        unique_values = df[self.target_column].unique()
        print(f"   Unique classes: {list(unique_values)}")
        
        # Encode target variable
        df['target_encoded'] = self.label_encoder.fit_transform(df[self.target_column])
        
        # Create label mapping for reference
        self.label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        print(f"   Label mapping: {self.label_mapping}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in features"""
        print("\n💧 Handling missing values...")
        
        missing_before = df[self.feature_columns].isnull().sum().sum()
        print(f"   Missing values before: {missing_before:,}")
        
        # Check which columns have too many missing values
        missing_pct = (df[self.feature_columns].isnull().sum() / len(df)) * 100
        cols_to_drop = missing_pct[missing_pct > 70].index.tolist()
        
        if cols_to_drop:
            print(f"   Dropping {len(cols_to_drop)} columns with >70% missing:")
            for col in cols_to_drop:
                print(f"      - {col} ({missing_pct[col]:.1f}% missing)")
            
            self.feature_columns = [col for col in self.feature_columns if col not in cols_to_drop]
            df = df.drop(columns=cols_to_drop)
        
        # Remove rows with too many missing values
        missing_per_row = df[self.feature_columns].isnull().sum(axis=1)
        threshold = len(self.feature_columns) * 0.5  # If >50% features missing
        rows_to_drop = missing_per_row > threshold
        
        if rows_to_drop.sum() > 0:
            print(f"   Dropping {rows_to_drop.sum()} rows with >50% missing features")
            df = df[~rows_to_drop]
        
        print(f"   Remaining rows: {len(df):,}")
        print(f"   Remaining features: {len(self.feature_columns)}")
        
        return df
    
    def remove_outliers(self, df, method='iqr', threshold=3):
        """Remove extreme outliers"""
        print("\n🔍 Removing outliers...")
        
        initial_len = len(df)
        
        for col in self.feature_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # Using 3x IQR (more conservative)
                upper_bound = Q3 + 3 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        removed = initial_len - len(df)
        print(f"   Removed {removed} outlier rows ({removed/initial_len*100:.1f}%)")
        print(f"   Remaining rows: {len(df):,}")
        
        return df
    
    def engineer_features(self, df):
        """Create new features from existing ones"""
        print("\n🔧 Engineering features...")
        
        created_features = []
        
        # Transit-related features
        if 'koi_period' in df.columns and 'koi_duration' in df.columns:
            df['period_duration_ratio'] = df['koi_period'] / (df['koi_duration'] + 1e-10)
            created_features.append('period_duration_ratio')
        
        if 'pl_orbper' in df.columns and 'pl_trandur' in df.columns:
            df['period_duration_ratio'] = df['pl_orbper'] / (df['pl_trandur'] + 1e-10)
            created_features.append('period_duration_ratio')
        
        # Depth to radius ratio
        if 'koi_depth' in df.columns and 'koi_prad' in df.columns:
            df['depth_radius_ratio'] = df['koi_depth'] / (df['koi_prad'] ** 2 + 1e-10)
            created_features.append('depth_radius_ratio')
        
        # Log transformations for skewed features
        if 'koi_model_snr' in df.columns:
            df['log_snr'] = np.log1p(df['koi_model_snr'])
            created_features.append('log_snr')
        
        if 'toi_snr' in df.columns:
            df['log_snr'] = np.log1p(df['toi_snr'])
            created_features.append('log_snr')
        
        # Stellar density
        if 'koi_srad' in df.columns and 'koi_smass' in df.columns:
            df['stellar_density'] = df['koi_smass'] / (df['koi_srad'] ** 3 + 1e-10)
            created_features.append('stellar_density')
        
        if 'st_rad' in df.columns and 'st_mass' in df.columns:
            df['stellar_density'] = df['st_mass'] / (df['st_rad'] ** 3 + 1e-10)
            created_features.append('stellar_density')
        
        # Habitable zone indicator
        temp_cols = ['koi_teq', 'pl_eqt']
        for temp_col in temp_cols:
            if temp_col in df.columns:
                df['habitable_zone'] = ((df[temp_col] >= 200) & (df[temp_col] <= 350)).astype(int)
                created_features.append('habitable_zone')
                break
        
        # Update feature columns
        self.feature_columns.extend([col for col in created_features if col not in self.feature_columns])
        
        print(f"   Created {len(created_features)} new features:")
        for feat in created_features:
            print(f"      - {feat}")
        
        return df
    
    def prepare_data(self, df):
        """Prepare final X and y arrays"""
        print("\n📦 Preparing final data...")
        
        # Extract features and target
        X = df[self.feature_columns].copy()
        y = df['target_encoded'].values
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Impute remaining missing values
        print("   Imputing missing values...")
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Scale features
        print("   Scaling features...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        print(f"   Final shape: {X_scaled.shape}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Samples: {len(X_scaled):,}")
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.15, random_state=42):
        """Split data into train, validation, and test sets"""
        print("\n✂️  Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"   Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Check class distribution
        print("\n   Class distribution:")
        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(split_y, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"      {split_name}: {dist}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, output_dir='models'):
        """Save preprocessor objects for later use"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving preprocessor to '{output_dir}/'...")
        
        # Save scaler and imputer
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        joblib.dump(self.imputer, f'{output_dir}/imputer.pkl')
        joblib.dump(self.label_encoder, f'{output_dir}/label_encoder.pkl')
        
        # Convert numpy types to native Python types for JSON serialization
        label_mapping_native = {str(k): int(v) for k, v in self.label_mapping.items()}
        
        # Save configuration
        config = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'label_mapping': label_mapping_native,
            'n_features': int(len(self.feature_columns))
        }
        
        with open(f'{output_dir}/preprocessing_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("   ✅ Saved:")
        print("      - scaler.pkl")
        print("      - imputer.pkl")
        print("      - label_encoder.pkl")
        print("      - preprocessing_config.json")
    
    def full_pipeline(self, filepath, save=True):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("COMPLETE PREPROCESSING PIPELINE")
        print("="*70)
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Identify columns
        self.identify_columns(df)
        
        # 3. Clean target
        df = self.clean_target(df)
        
        # 4. Handle missing values
        df = self.handle_missing_values(df)
        
        # 5. Remove outliers
        df = self.remove_outliers(df, method='iqr')
        
        # 6. Engineer features
        df = self.engineer_features(df)
        
        # 7. Prepare data
        X, y = self.prepare_data(df)
        
        # 8. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 9. Save preprocessor
        if save:
            self.save_preprocessor()
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        
        print("\n✅ Data is ready for model training!")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Classes: {len(self.label_mapping)}")
        print(f"   Train samples: {len(X_train):,}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize preprocessor
    preprocessor = ExoplanetPreprocessor()
    
    # Run full pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.full_pipeline(
        filepath=r'D:\Exoplanet Detection System\kepler_cumulative_selected.csv',  # CHANGE THIS to your file
        save=True
    )
    
    # Save processed data for model training
    print("\n💾 Saving processed data...")
    np.save('data/X_train.npy', X_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_val.npy', y_val)
    np.save('data/y_test.npy', y_test)
    
    print("✅ Processed data saved!")
    print("\n🚀 Next: Run model training script")