"""
STEP 2: Exploratory Data Analysis (EDA)
Understand your merged dataset before building models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExoplanetEDA:
    """Comprehensive Exploratory Data Analysis for Exoplanet Dataset"""
    
    def __init__(self, filepath):
        """Load the merged dataset"""
        print("📂 Loading merged dataset...")
        self.df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(self.df)} rows and {len(self.df.columns)} columns\n")
        
    def basic_info(self):
        """Display basic dataset information"""
        print("="*70)
        print("BASIC DATASET INFORMATION")
        print("="*70)
        
        print(f"\n📊 Dataset Shape: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]:,}")
        print(f"   Columns: {self.df.shape[1]}")
        
        print(f"\n💾 Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📋 Column Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        print("\n🔍 Data Types:")
        print(self.df.dtypes.value_counts())
        
        print("\n" + "="*70 + "\n")
        
    def check_target_variable(self):
        """Analyze target variable distribution"""
        print("="*70)
        print("TARGET VARIABLE ANALYSIS")
        print("="*70)
        
        # Find target column (disposition column)
        target_cols = [col for col in self.df.columns if 'disposition' in col.lower() or 'disp' in col.lower()]
        
        if not target_cols:
            print("⚠️  No target column found! Look for columns like:")
            print("   - koi_disposition")
            print("   - tfopwg_disp")
            print("   - k2c_disp")
            return None
        
        target_col = target_cols[0]
        print(f"\n🎯 Target Column: '{target_col}'")
        
        print("\n📊 Class Distribution:")
        counts = self.df[target_col].value_counts()
        percentages = self.df[target_col].value_counts(normalize=True) * 100
        
        for label, count in counts.items():
            pct = percentages[label]
            print(f"   {label:20s}: {count:5,} ({pct:5.2f}%)")
        
        # Check for class imbalance
        max_pct = percentages.max()
        min_pct = percentages.min()
        imbalance_ratio = max_pct / min_pct
        
        print(f"\n⚖️  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 3:
            print("   ⚠️  WARNING: Significant class imbalance detected!")
            print("   💡 Consider using: class_weight='balanced' in models")
        else:
            print("   ✅ Class distribution is reasonably balanced")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar', color=['#4ade80', '#fbbf24', '#ef4444'])
        plt.title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        print("\n💾 Saved visualization: target_distribution.png")
        
        print("\n" + "="*70 + "\n")
        return target_col
    
    def check_missing_values(self):
        """Analyze missing values in dataset"""
        print("="*70)
        print("MISSING VALUES ANALYSIS")
        print("="*70)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_df) == 0:
            print("\n✅ No missing values found! Perfect dataset!")
        else:
            print(f"\n⚠️  Found missing values in {len(missing_df)} columns:\n")
            print(missing_df.to_string(index=False))
            
            # Recommendations
            print("\n💡 RECOMMENDATIONS:")
            for _, row in missing_df.iterrows():
                col = row['Column']
                pct = row['Missing_Percentage']
                
                if pct > 70:
                    print(f"   ❌ {col}: {pct:.1f}% missing → Consider DROPPING this column")
                elif pct > 40:
                    print(f"   ⚠️  {col}: {pct:.1f}% missing → Use carefully, impute or drop")
                elif pct > 10:
                    print(f"   🔸 {col}: {pct:.1f}% missing → Impute with median/mean")
                else:
                    print(f"   ✅ {col}: {pct:.1f}% missing → Safe to impute")
        
        print("\n" + "="*70 + "\n")
        return missing_df
    
    def analyze_numerical_features(self):
        """Statistical analysis of numerical features"""
        print("="*70)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*70)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n📈 Found {len(numerical_cols)} numerical features")
        
        # Basic statistics
        print("\n📊 Statistical Summary:")
        stats = self.df[numerical_cols].describe().T
        stats['missing%'] = (self.df[numerical_cols].isnull().sum() / len(self.df) * 100)
        print(stats[['mean', 'std', 'min', 'max', 'missing%']].to_string())
        
        # Check for outliers using IQR method
        print("\n🔍 Outlier Detection (using IQR method):")
        for col in numerical_cols[:10]:  # Check first 10 columns
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(self.df)) * 100
            
            if outlier_pct > 5:
                print(f"   ⚠️  {col}: {outliers} outliers ({outlier_pct:.1f}%)")
        
        print("\n" + "="*70 + "\n")
    
    def check_correlations(self, target_col):
        """Analyze feature correlations with target"""
        print("="*70)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*70)
        
        # Encode target if needed
        if self.df[target_col].dtype == 'object':
            label_map = {label: idx for idx, label in enumerate(sorted(self.df[target_col].unique()))}
            target_encoded = self.df[target_col].map(label_map)
        else:
            target_encoded = self.df[target_col]
        
        # Get numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlations with target
        correlations = {}
        for col in numerical_cols:
            if col != target_col:
                corr = self.df[col].corr(target_encoded)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        
        # Sort by correlation strength
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 Top 15 Features Correlated with Target:\n")
        for i, (feature, corr) in enumerate(sorted_corr[:15], 1):
            print(f"   {i:2d}. {feature:30s}: {corr:.4f}")
        
        # Visualize top correlations
        if len(sorted_corr) > 0:
            top_features = [f[0] for f in sorted_corr[:10]]
            top_corr_values = [f[1] for f in sorted_corr[:10]]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), top_corr_values, color='#667eea')
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Absolute Correlation with Target', fontsize=12)
            plt.title('Top 10 Features by Correlation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
            print("\n💾 Saved visualization: feature_correlations.png")
        
        print("\n" + "="*70 + "\n")
        return sorted_corr
    
    def generate_report(self):
        """Generate complete EDA report"""
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS REPORT")
        print("="*70 + "\n")
        
        # 1. Basic Info
        self.basic_info()
        
        # 2. Target Variable
        target_col = self.check_target_variable()
        
        # 3. Missing Values
        self.check_missing_values()
        
        # 4. Numerical Features
        self.analyze_numerical_features()
        
        # 5. Correlations
        if target_col:
            self.check_correlations(target_col)
        
        print("\n" + "="*70)
        print("EDA COMPLETE!")
        print("="*70)
        
        print("\n📋 SUMMARY & NEXT STEPS:")
        print("\n✅ What we learned:")
        print("   1. Dataset size and structure")
        print("   2. Target variable distribution")
        print("   3. Missing values pattern")
        print("   4. Important features")
        print("   5. Potential outliers")
        
        print("\n🚀 Next Steps:")
        print("   1. Clean the data (handle missing values)")
        print("   2. Engineer new features")
        print("   3. Scale/normalize features")
        print("   4. Split data (train/val/test)")
        print("   5. Train machine learning models")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("🚀 NASA EXOPLANET DETECTION - DATA EXPLORATION")
    print("="*70 + "\n")
    
    # Initialize EDA
    # REPLACE 'merged_dataset.csv' with your actual filename
    eda = ExoplanetEDA(r'D:\Exoplanet Detection System\kepler_cumulative_selected.csv')
    
    # Generate complete report
    eda.generate_report()
    
    print("\n💡 TIP: Review the generated plots:")
    print("   - target_distribution.png")
    print("   - feature_correlations.png")
    print("\n   These will help you understand your data better!")