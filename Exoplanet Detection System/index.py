"""
Script to download NASA Exoplanet datasets with only important columns
This saves time and reduces file size!
"""

import pandas as pd
import requests
from io import StringIO

def download_kepler_cumulative():
    """Download Kepler Cumulative table with selected columns"""
    
    print("📡 Downloading Kepler Cumulative dataset...")
    
    # Important columns for Kepler
    columns = [
        'kepoi_name',  # Identifier (for reference)
        'koi_disposition',  # TARGET VARIABLE
        
        # Transit properties
        'koi_period',
        'koi_duration',
        'koi_depth',
        'koi_prad',
        'koi_sma',
        
        # Signal quality
        'koi_model_snr',
        'koi_num_transits',
        'koi_max_mult_ev',
        'koi_max_sngle_ev',
        
        # Planet properties
        'koi_teq',
        'koi_insol',
        
        # Stellar properties
        'koi_steff',
        'koi_slogg',
        'koi_srad',
        'koi_smass',
        
        # Validation
        'koi_count',
        'koi_fpflag_nt',
        'koi_fpflag_ss',
        'koi_fpflag_co',
        'koi_fpflag_ec'
    ]
    
    # Build query URL
    columns_str = ','.join(columns)
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"select {columns_str} from cumulative"
    
    params = {
        'query': query,
        'format': 'csv'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        
        # Load into DataFrame
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Downloaded {len(df)} rows with {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        
        # Save to CSV
        df.to_csv('kepler_cumulative_selected.csv', index=False)
        print(f"💾 Saved to: kepler_cumulative_selected.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative")
        return None


def download_k2_candidates():
    """Download K2 Planets and Candidates table"""
    
    print("\n📡 Downloading K2 Planets and Candidates...")
    
    columns = [
        'epic_name',  # Identifier
        'k2c_disp',  # TARGET VARIABLE
        
        # Transit properties
        'pl_orbper',
        'pl_trandur',
        'pl_trandep',
        'pl_rade',
        'pl_orbsmax',
        
        # Planet properties
        'pl_eqt',
        'pl_insol',
        'pl_dens',
        
        # Stellar properties
        'st_teff',
        'st_logg',
        'st_rad',
        'st_mass'
    ]
    
    columns_str = ','.join(columns)
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"select {columns_str} from k2pandc"
    
    params = {
        'query': query,
        'format': 'csv'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Downloaded {len(df)} rows with {len(df.columns)} columns")
        
        df.to_csv('k2_candidates_selected.csv', index=False)
        print(f"💾 Saved to: k2_candidates_selected.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc")
        return None


def download_tess_toi():
    """Download TESS Objects of Interest"""
    
    print("\n📡 Downloading TESS TOI dataset...")
    
    columns = [
        'toi',  # Identifier
        'tfopwg_disp',  # TARGET VARIABLE
        
        # Transit properties
        'pl_orbper',
        'pl_trandur',
        'pl_trandep',
        'pl_rade',
        
        # Signal quality
        'toi_snr',
        
        # Planet properties
        'pl_eqt',
        'pl_insol',
        'pl_bmasse',
        
        # Stellar properties
        'st_teff',
        'st_logg',
        'st_rad',
        'st_mass',
        'st_tmag'
    ]
    
    columns_str = ','.join(columns)
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"select {columns_str} from toi"
    
    params = {
        'query': query,
        'format': 'csv'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Downloaded {len(df)} rows with {len(df.columns)} columns")
        
        df.to_csv('tess_toi_selected.csv', index=False)
        print(f"💾 Saved to: tess_toi_selected.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI")
        return None


def download_all_datasets():
    """Download all datasets with important columns"""
    
    print("="*60)
    print("NASA EXOPLANET ARCHIVE - DATASET DOWNLOADER")
    print("="*60)
    
    datasets = {}
    
    # Download Kepler
    kepler_df = download_kepler_cumulative()
    if kepler_df is not None:
        datasets['kepler'] = kepler_df
    
    # Download K2
    k2_df = download_k2_candidates()
    if k2_df is not None:
        datasets['k2'] = k2_df
    
    # Download TESS
    tess_df = download_tess_toi()
    if tess_df is not None:
        datasets['tess'] = tess_df
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"{name.upper()}: {len(df)} rows, {len(df.columns)} columns")
    
    print("\n✅ All datasets downloaded successfully!")
    print("\nNext steps:")
    print("1. Check the CSV files in your directory")
    print("2. Run preprocessing script")
    print("3. Train ML models")
    
    return datasets


# Manual download instructions
def print_manual_download_instructions():
    """Print instructions for manual download"""
    
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 KEPLER CUMULATIVE TABLE:")
    print("1. Go to: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative")
    print("2. Click 'Download Table' button (top right)")
    print("3. Select 'Download Table' → 'All columns' or 'Select columns'")
    print("4. Choose these columns:")
    print("   - koi_disposition (TARGET)")
    print("   - koi_period, koi_duration, koi_depth, koi_prad")
    print("   - koi_model_snr, koi_num_transits")
    print("   - koi_teq, koi_insol")
    print("   - koi_steff, koi_slogg, koi_srad")
    print("5. Format: CSV")
    print("6. Save as: kepler_cumulative.csv")
    
    print("\n📋 K2 PLANETS AND CANDIDATES:")
    print("1. Go to: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc")
    print("2. Repeat same process with K2 columns")
    print("3. Save as: k2_candidates.csv")
    
    print("\n📋 TESS TOI:")
    print("1. Go to: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI")
    print("2. Repeat same process with TOI columns")
    print("3. Save as: tess_toi.csv")
    
    print("\n" + "="*60)


# Main execution
if __name__ == "__main__":
    
    # Try automatic download
    datasets = download_all_datasets()
    
    # If automatic download fails, print manual instructions
    if not datasets:
        print_manual_download_instructions()
    
    # Show data preview