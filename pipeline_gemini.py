import os
import glob
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Setup plotting to save to file instead of display
plt.switch_backend('Agg')

class AerosolPipeline:
    def __init__(self, input_dir='raw_data', output_dir='output'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def read_aeronet_file(self, filepath):
        """
        User-provided function to read AERONET files, finding headers dynamically
        and prefixing columns to avoid duplicates.
        """
        try:
            with open(filepath, 'r', encoding='latin1') as f:
                lines = f.readlines()
            
            header_line = None
            for i, line in enumerate(lines):
                if "date" in line.lower() and "time" in line.lower():
                    header_line = i
                    break
            
            if header_line is None:
                return None # Skip files without valid headers

            df = pd.read_csv(filepath, skiprows=header_line, sep=",", engine="python")

            # Standardize column names
            df.columns = [c.strip() for c in df.columns]
            
            # Find Date and Time columns safely
            date_cols = [c for c in df.columns if c.lower().startswith("date")]
            time_cols = [c for c in df.columns if c.lower().startswith("time")]
            
            if not date_cols or not time_cols:
                return None

            df = df.rename(columns={date_cols[0]: "Date", time_cols[0]: "Time"})

            # Keep only Date, Time, and measurement columns
            measure_cols = [c for c in df.columns if c not in ["Date", "Time"]]

            # Prefix measurement columns with file type (e.g., AOD_, SSA_)
            file_prefix = os.path.splitext(os.path.basename(filepath))[1][1:].upper()
            
            # Select and rename
            df = df[["Date", "Time"] + measure_cols]
            df = df.rename(columns={col: f"{file_prefix}_{col}" for col in measure_cols})

            return df
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            return None

    def process_zip_site(self, zip_path):
        """
        Extracts zip, merges files using user logic, and runs analysis.
        """
        site_name = os.path.splitext(os.path.basename(zip_path))[0]
        print(f"\n=== Processing Site: {site_name} ===")
        
        # Create unique extract folder for this site
        extract_dir = os.path.join(self.input_dir, f"temp_{site_name}")
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        # 1. Unzip
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            print(f"Error: {site_name} is a bad zip file.")
            return

        # 2. Read all valid files inside
        dfs = []
        file_paths = glob.glob(os.path.join(extract_dir, "*"))
        # Looking for standard Aeronet extensions
        valid_exts = (".aod", ".ssa", ".rin", ".cad", ".tab")
        
        for file in file_paths:
            if file.lower().endswith(valid_exts):
                print(f"  Reading {os.path.basename(file)}")
                df = self.read_aeronet_file(file)
                if df is not None:
                    dfs.append(df)

        if not dfs:
            print(f"No valid data files found in {zip_path}")
            return

        # 3. Merge (User Logic: Reduce + Inner Join)
        try:
            df_merged = reduce(
                lambda left, right: pd.merge(left, right, on=["Date", "Time"], how="inner"),
                dfs
            )
        except Exception as e:
            print(f"Error merging files for {site_name}: {e}")
            return
            
        print(f"  Merged shape: {df_merged.shape}")

        # 4. FIX COLUMN NAMES for Downstream Compatibility
        # The merge created 'AOD_AOD_Extinction...' but we need 'AOD_Extinction...' for the formulas
        new_columns = []
        for col in df_merged.columns:
            # Fix AOD double prefix
            if col.startswith("AOD_AOD_"): 
                new_columns.append(col.replace("AOD_AOD_", "AOD_"))
            # Fix SSA double prefix (optional, if your formulas need it)
            elif col.startswith("SSA_Single_"):
                new_columns.append(col.replace("SSA_Single_", "Single_"))
            # Keep original if no fix needed (Date, Time)
            else:
                new_columns.append(col)
        
        df_merged.columns = new_columns

        # Clean up temp files
        for f in file_paths:
            os.remove(f)
        os.rmdir(extract_dir)

        # 5. Run the Physics & ML Pipelines
        self.process_site_analysis(df_merged, site_name)

    def process_site_analysis(self, df, site_name):
        """
        Runs the cleaning, AE/FMF calculation (Exp2) and ML (Exp3).
        """
        site_out_dir = os.path.join(self.output_dir, site_name)
        if not os.path.exists(site_out_dir):
            os.makedirs(site_out_dir)

        # --- CLEANING ---
        df.replace(-999, np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # --- CALCULATIONS (AE & FMF) ---
        # Ensure required columns exist
        req_col_ae = 'AOD_Extinction-Total[440nm]'
        if req_col_ae not in df.columns:
            # Fallback: check if the prefix fix missed something or if data is missing
            possible = [c for c in df.columns if "440nm" in c and "Total" in c]
            if possible:
                df.rename(columns={possible[0]: req_col_ae}, inplace=True)
            else:
                print(f"  Skipping AE Calc: Missing 440nm Total AOD in {site_name}")
                return

        # AE Calculation
        try:
            df['AE'] = -(np.log(df['AOD_Extinction-Total[870nm]'] / df['AOD_Extinction-Total[440nm]']) / np.log(870/440))
            df['AOD_Extinction-Total[500nm]'] = df['AOD_Extinction-Total[440nm]'] * ((500/440)**(-df['AE']))
        except Exception as e:
            print(f"  Error in AE Calc: {e}")
            return

        # FMF Calculation (Curve Fit)
        fmf_list = []
        wavelengths = np.array([440, 675, 870, 1020])
        # Note: Depending on file version, these might be named differently.
        # We try to find them dynamically if exact match fails.
        total_cols = [f'AOD_Extinction-Total[{w}nm]' for w in wavelengths]
        fine_cols = [f'AOD_Extinction-Fine[{w}nm]' for w in wavelengths]

        # Check if fine mode columns exist (usually in .sda file or .aod V3 L2)
        has_fine = all(c in df.columns for c in fine_cols)
        
        if has_fine:
            def poly2(x, a, b, c): return a * x**2 + b * x + c
            
            for i, row in df.iterrows():
                try:
                    totals = row[total_cols].astype(float).values
                    fines = row[fine_cols].astype(float).values
                    if np.any(totals <= 0) or np.any(fines <= 0):
                        fmf_list.append(np.nan)
                        continue
                        
                    log_tot = np.log(totals)
                    log_fine = np.log(fines)
                    
                    popt_tot, _ = curve_fit(poly2, wavelengths, log_tot, maxfev=1000)
                    popt_fine, _ = curve_fit(poly2, wavelengths, log_fine, maxfev=1000)
                    
                    tgt_tot = np.exp(poly2(550, *popt_tot))
                    tgt_fine = np.exp(poly2(550, *popt_fine))
                    
                    fmf_list.append(tgt_fine / tgt_tot if tgt_tot != 0 else np.nan)
                except:
                    fmf_list.append(np.nan)
            df['FMF'] = fmf_list
            df['FMF'] = df['FMF'].fillna(df['FMF'].median())
        else:
            print("  Warning: Fine Mode columns not found. FMF will be 0.5 (Placeholder).")
            df['FMF'] = 0.5 # Placeholder to allow script to continue

        # --- CLASSIFICATION ---
        def classify_aerosol(row):
            # Using 'Single_Scattering_Albedo[440nm]' if available, else standard fallback
            SSA = row.get('Single_Scattering_Albedo[440nm]', 0.9)
            FMF = row['FMF']
            AE = row['AE']
            
            # Coarse regime
            if FMF <= 0.4 and AE <= 0.6:
                return "CNA" if SSA > 0.95 else "CA"
            # Mixed regime
            elif 0.4 < FMF <= 0.6 and 0.6 < AE <= 1.2:
                return "MNA" if SSA > 0.95 else "MA"
            # Fine regime
            elif FMF > 0.6 and AE > 1.2:
                if SSA > 0.95: return "FNA"
                elif 0.9 <= SSA < 0.95: return "BCL"
                elif 0.85 <= SSA < 0.9: return "BCM"
                else: return "BCH"
            return "Unclassified"

        def classify_source(row):
            aod = row.get('AOD_Extinction-Total[500nm]', 0)
            ae = row.get('AE', 0)
            if 0.2 <= aod <= 0.4 and ae > 1: return "Urban"
            elif aod < 0.3 and 0.5 <= ae <= 1.7: return "Maritime"
            elif aod > 0.4 and ae < 1: return "Desert"
            elif aod > 0.7 and ae > 1: return "Biomass"
            elif aod > 0.45 and ae > 1.2: return "Arid"
            else: return "Unclassified"

        df['Label'] = df.apply(classify_aerosol, axis=1)
        df['Source'] = df.apply(classify_source, axis=1)
        
        # Season
        # Parse Day of Year from Date
        df['Day_of_Year'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.dayofyear
        
        def get_season(day):
            if pd.isna(day): return "Unknown"
            if 1 <= day <= 60 or 335 <= day <= 366: return "Winter"
            elif 61 <= day <= 152: return "Pre monsoon"
            elif 153 <= day <= 244: return "Monsoon"
            elif 245 <= day <= 334: return "Post monsoon"
            return "Unknown"
        df['Season'] = df['Day_of_Year'].apply(get_season)

        # Save Processed Data
        out_path = os.path.join(site_out_dir, f'{site_name}_Processed.csv')
        df.to_csv(out_path, index=False)
        print(f"  Saved processed CSV to {out_path}")

        # --- VISUALIZATION (Plots from Exp 2) ---
        print("  Generating Visualization Plots...")
        
        # Plot 1: AOD vs AE by Label
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='AE', y='AOD_Extinction-Total[500nm]', hue='Label', alpha=0.7)
            plt.title(f"AOD vs AE (Label) - {site_name}")
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(site_out_dir, 'Plot_AOD_AE_Label.png'))
            plt.close()
        except Exception as e:
            print(f"  Could not plot Label Scatter: {e}")

        # Plot 2: Source Distribution Bar Chart
        try:
            plt.figure(figsize=(10, 6))
            df['Source'].value_counts().plot(kind='barh', color='teal')
            plt.xlabel('Number of Entries')
            plt.ylabel('Source')
            plt.title(f"Source Distribution - {site_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(site_out_dir, 'Plot_Source_Dist.png'))
            plt.close()
        except Exception as e:
            print(f"  Could not plot Source Distribution: {e}")

        # --- ML ANALYSIS (Exp 3) ---
        self.run_ml(df, site_name, site_out_dir)

    def run_ml(self, df, site_name, site_out_dir):
        # Filter for ML
        valid_sources = df['Source'].value_counts()
        valid_sources = valid_sources[valid_sources >= 5].index
        df_ml = df[df['Source'].isin(valid_sources)].copy()
        
        if len(df_ml['Source'].unique()) < 2:
            print("  Skipping ML: Not enough source classes.")
            return

        # Prepare X and y
        drop_cols = ['Date', 'Time', 'Source', 'Season', 'Day_of_Year', 'Label']
        X = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns], errors='ignore')
        y = df_ml['Source']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Preprocessing (Force Dense for Naive Bayes)
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ], sparse_threshold=0) # <--- FIX for Sparse Error

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Naive Bayes": GaussianNB()
        }

        results = []
        print("  Running ML Models...")
        
        for name, model in models.items():
            try:
                clf = Pipeline([('prep', preprocessor), ('model', model)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                results.append({'Model': name, 'Accuracy': acc})
                
                # Save CM
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name} - {site_name}')
                plt.savefig(os.path.join(site_out_dir, f'CM_{name}.png'))
                plt.close()
            except Exception as e:
                print(f"    Failed {name}: {e}")

        pd.DataFrame(results).to_csv(os.path.join(site_out_dir, 'ML_Results.csv'), index=False)
        print("  ML Completed.")

    def execute(self):
        # Look for ZIP files
        zip_files = glob.glob(os.path.join(self.input_dir, "*.zip"))
        
        if not zip_files:
            print("No .zip files found in raw_data folder.")
            return

        print(f"Found {len(zip_files)} zip files.")
        for zip_path in zip_files:
            self.process_zip_site(zip_path)

if __name__ == "__main__":
    # Assumes you have a folder 'raw_data' with your .zip files
    pipeline = AerosolPipeline(input_dir='raw_data', output_dir='output')
    pipeline.execute()