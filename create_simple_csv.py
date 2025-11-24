import os
import glob
import pandas as pd

def create_simple_master(output_dir='output'):
    print("=== 🧹 Creating Clean Master Dataset (Existing Data Only) ===\n")

    # 1. ALIAS MAP 
    # We rename the columns to match what the Colab code expects.
    alias_map = {
        "Date": "Date(dd:mm:yyyy)",
        "Time": "Time(hh:mm:ss)",
        "Source": "Source Label",
        "AOD_Extinction-Total[500nm]": "AOD_500",
        # We rename SSA just in case it exists, but we won't force it if missing
        "Single_Scattering_Albedo[440nm]": "SSA_440",
        "Single_Scattering_Albedo[675nm]": "SSA_675",
        "Single_Scattering_Albedo[870nm]": "SSA_870",
        "Single_Scattering_Albedo[1020nm]": "SSA_1020",
    }

    # 2. LOCATE FILES
    files = glob.glob(os.path.join(output_dir, "*", "*_Processed.csv"))
    
    if not files:
        print(f"❌ No processed files found in {output_dir}")
        return

    print(f"Found {len(files)} regional files. Processing...")
    
    dfs = []
    
    for filepath in files:
        try:
            site_name = os.path.basename(os.path.dirname(filepath))
            
            # Read Data
            df = pd.read_csv(filepath)
            
            # Add Site Column
            df['Site'] = site_name
            
            # Rename columns
            df = df.rename(columns=alias_map)
            
            # We do NOT add empty columns. We just take what is there.
            dfs.append(df)
            print(f"  ✅ Added: {site_name}")
            
        except Exception as e:
            print(f"  ⚠️ Error processing {filepath}: {e}")

    # 3. CONCATENATE AND SAVE
    if dfs:
        # Concatenate creates a master DF with all columns found in any file
        master_df = pd.concat(dfs, ignore_index=True)
        
        # Verify Key Columns Exist (Just a warning check)
        key_cols = ['AOD_500', 'AE', 'FMF']
        missing = [c for c in key_cols if c not in master_df.columns]
        
        output_filename = "South_America_Master_Clean.csv"
        master_df.to_csv(output_filename, index=False)
        
        print("\n" + "="*50)
        print(f"Done! Final file saved as: {output_filename}")
        print(f"Total Rows: {len(master_df)}")
        if missing:
            print(f"⚠️ WARNING: The following critical columns are MISSING: {missing}")
            print("The clustering code will NOT work without them.")
        else:
            print("✅ All critical clustering columns (AOD, AE, FMF) are present.")
        print("="*50)
    else:
        print("No valid dataframes were created.")

if __name__ == "__main__":
    create_simple_master()