import os
import glob
import pandas as pd

def diagnose_csv_headers(output_dir='output'):
    print("=== 🚑 DATA DOCTOR: Diagnosing Region CSV Headers ===\n")

    # 1. THE STRICT REQUIREMENTS (Target Headers)
    required_headers = {
        "Site", "Date(dd:mm:yyyy)", "Time(hh:mm:ss)", "FMF", 
        "AOD_Extinction-Total[440nm]", "AOD_Extinction-Total[675nm]", "AOD_Extinction-Total[870nm]", "AOD_Extinction-Total[1020nm]",
        "AOD_Extinction-Fine[440nm]", "AOD_Extinction-Fine[675nm]", "AOD_Extinction-Fine[870nm]", "AOD_Extinction-Fine[1020nm]",
        "AOD_Extinction-Coarse[440nm]", "AOD_Extinction-Coarse[675nm]", "AOD_Extinction-Coarse[870nm]", "AOD_Extinction-Coarse[1020nm]",
        "AE", "Coincident_AOD440nm", 
        "Surface_Albedo[440m]", "Surface_Albedo[675m]", "Surface_Albedo[870m]", "Surface_Albedo[1020m]",
        "AOD_Coincident_Input[440nm]", "AOD_Coincident_Input[675nm]", "AOD_Coincident_Input[870nm]", "AOD_Coincident_Input[1020nm]",
        "Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD",
        "Refractive_Index-Real_Part[440nm]", "Refractive_Index-Real_Part[675nm]", "Refractive_Index-Real_Part[870nm]", "Refractive_Index-Real_Part[1020nm]",
        "Refractive_Index-Imaginary_Part[440nm]", "Refractive_Index-Imaginary_Part[675nm]", "Refractive_Index-Imaginary_Part[870nm]", "Refractive_Index-Imaginary_Part[1020nm]",
        "SSA_440", "Single_Scattering_Albedo[675nm]", "SSA_870", "Single_Scattering_Albedo[1020nm]",
        "Absorption_AOD[440nm]", "Absorption_AOD[675nm]", "Absorption_AOD[870nm]", "Absorption_AOD[1020nm]",
        "Absorption_Angstrom_Exponent_440-870nm", 
        "Label", "Season", "AOD_500", "Source Label"
    }

    # 2. THE ALIAS MAP
    # These are columns that exist in your files but need renaming to match the target.
    # The script will simulate this rename to see if the file WOULD be valid.
    alias_map = {
        "Date": "Date(dd:mm:yyyy)",
        "Time": "Time(hh:mm:ss)",
        "Source": "Source Label",
        "AOD_Extinction-Total[500nm]": "AOD_500",
        "Single_Scattering_Albedo[440nm]": "SSA_440",
        "Single_Scattering_Albedo[870nm]": "SSA_870",
        # Mapping nm to m typo handling if necessary
        "Surface_Albedo[440nm]": "Surface_Albedo[440m]",
        "Surface_Albedo[675nm]": "Surface_Albedo[675m]",
        "Surface_Albedo[870nm]": "Surface_Albedo[870m]",
        "Surface_Albedo[1020nm]": "Surface_Albedo[1020m]",
    }

    # 3. SCAN FILES
    files = glob.glob(os.path.join(output_dir, "*", "*_Processed.csv"))
    
    if not files:
        print("No Processed CSVs found.")
        return

    good_files = []
    bad_files = []

    print(f"Scanning {len(files)} regions...\n")

    for filepath in files:
        site_name = os.path.basename(os.path.dirname(filepath))
        
        try:
            # Read ONLY the header (nrows=0) for speed
            df_head = pd.read_csv(filepath, nrows=0)
            
            # 1. Add "Site" (simulated)
            current_cols = set(df_head.columns)
            current_cols.add("Site") 

            # 2. Apply Alias Map (Rename)
            renamed_cols = set()
            for col in current_cols:
                if col in alias_map:
                    renamed_cols.add(alias_map[col])
                else:
                    renamed_cols.add(col)
            
            # 3. Calculate Differences
            # What is in Required that is NOT in Renamed?
            missing = required_headers - renamed_cols
            
            if len(missing) == 0:
                good_files.append(site_name)
            else:
                bad_files.append((site_name, missing))

        except Exception as e:
            print(f"CRITICAL ERROR reading {site_name}: {e}")

    # 4. REPORT
    print("-" * 50)
    print(f"✅ READY TO MERGE: {len(good_files)} regions")
    print(f"❌ DISCREPANCIES : {len(bad_files)} regions")
    print("-" * 50)

    if bad_files:
        print("\n--- DETAILED ERROR REPORT ---")
        for site, missing_set in bad_files:
            print(f"\n📍 Region: {site}")
            print(f"   Missing {len(missing_set)} Columns:")
            # Sort for readability
            for col in sorted(list(missing_set)):
                print(f"    - {col}")
            
            # Heuristic Diagnosis
            if "FMF" in missing_set:
                print("     (Hint: FMF calculation might have failed or Fine Mode columns were missing)")
            if "Source Label" in missing_set:
                print("     (Hint: Is the column named 'Source' in the csv?)")
            if any("Refractive" in s for s in missing_set):
                print("     (Hint: Inversion data (RIN) might be missing for this site)")

    else:
        print("\n🎉 PERFECT! All regions have matching headers.")
        print("You can run the collation script now.")

if __name__ == "__main__":
    diagnose_csv_headers()