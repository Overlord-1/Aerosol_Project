import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    def find_sites(self):
        """Scans input directory and groups files by Site Name."""
        all_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        sites = {}
        
        # Assumes format: DateRange_SiteName.type.csv
        # Example: 20060101_20251231_ICIPE-Mbita.aod.csv
        for f in all_files:
            filename = os.path.basename(f)
            try:
                # Splitting by '.' to get the extension part (aod, cad, etc)
                parts = filename.split('.')
                file_type = parts[-2] # aod, cad, rin, ssa, tab
                
                # The prefix is everything before the first '.'
                prefix = parts[0]
                
                if prefix not in sites:
                    sites[prefix] = {}
                sites[prefix][file_type] = f
            except Exception as e:
                print(f"Skipping file {filename}: {e}")
                
        return sites

    def process_site_data(self, site_name, files):
        """
        Logic from DA-Exp2: Merges, Cleans, Imputes, Calculates FMF/AE, Classifies.
        """
        print(f"--- Processing Data for: {site_name} ---")
        site_out_dir = os.path.join(self.output_dir, site_name)
        if not os.path.exists(site_out_dir):
            os.makedirs(site_out_dir)

        # 1. Load Data
        required_types = ['aod', 'cad', 'rin', 'ssa', 'tab']
        dfs = {}
        for rt in required_types:
            if rt not in files:
                print(f"Missing {rt} file for {site_name}. Skipping site.")
                return None
            dfs[rt] = pd.read_csv(files[rt])

        # 2. Merge
        core_columns = dfs['aod'].iloc[:, :5] # Date, Time, etc.
        unique_dfs = [dfs[t].iloc[:, 5:] for t in required_types]
        merged_df = pd.concat([core_columns] + unique_dfs, axis=1)

        # --- CRITICAL FIX: Remove Duplicate Columns ---
        # This keeps the first occurrence of a column name and drops duplicates
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # 3. Clean (-999 to NaN) and Impute (Median)
        merged_df.replace(-999, np.nan, inplace=True)
        # Select numeric columns for median calculation to avoid string errors
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        medians = merged_df[numeric_cols].median()
        merged_df.fillna(medians, inplace=True)

        # 4. Calculate AE (Angstrom Exponent)
        # Formula: - ln(870/440) / ln(870/440) logic from your script
        try:
            merged_df['AE'] = -(np.log(merged_df['AOD_Extinction-Total[870nm]'] / 
                                       merged_df['AOD_Extinction-Total[440nm]']) / np.log(870/440))
            
            merged_df['AOD_Extinction-Total[500nm]'] = merged_df['AOD_Extinction-Total[440nm]'] * \
                                                       ((500/440)**(-merged_df['AE']))
        except KeyError:
            print(f"Critical columns missing for AE calculation in {site_name}")
            return None

        # 5. Calculate FMF (Fine Mode Fraction) - Curve Fitting
        # Note: Optimized version of your loop
        fmf_list = []
        wavelengths = np.array([440, 675, 870, 1020])
        
        total_cols = ['AOD_Extinction-Total[440nm]', 'AOD_Extinction-Total[675nm]',
                      'AOD_Extinction-Total[870nm]', 'AOD_Extinction-Total[1020nm]']
        fine_cols = ['AOD_Extinction-Fine[440nm]', 'AOD_Extinction-Fine[675nm]',
                     'AOD_Extinction-Fine[870nm]', 'AOD_Extinction-Fine[1020nm]']

        def poly2(x, a, b, c): return a * x**2 + b * x + c

        for i, row in merged_df.iterrows():
            try:
                totals = row[total_cols].astype(float).values
                fines = row[fine_cols].astype(float).values
                
                # Basic check to avoid log(<=0)
                if np.any(totals <= 0) or np.any(fines <= 0):
                    fmf_list.append(np.nan)
                    continue

                log_tot = np.log(totals)
                log_fine = np.log(fines)

                popt_tot, _ = curve_fit(poly2, wavelengths, log_tot, maxfev=1000)
                popt_fine, _ = curve_fit(poly2, wavelengths, log_fine, maxfev=1000)

                target_aod_tot = np.exp(poly2(550, *popt_tot))
                target_aod_fine = np.exp(poly2(550, *popt_fine))

                fmf = target_aod_fine / target_aod_tot if target_aod_tot != 0 else np.nan
                fmf_list.append(fmf)
            except:
                fmf_list.append(np.nan)
        
        merged_df['FMF'] = fmf_list
        merged_df['FMF'] = merged_df['FMF'].fillna(merged_df['FMF'].median()) # Impute failed fits

        # 6. Classification Logic (Label & Source)
        def classify_aerosol(row):
            SSA = row.get('Single_Scattering_Albedo[440nm]', 0.9) # Default safe fallback
            FMF = row['FMF']
            AE = row['AE']
            
            if FMF <= 0.4 and AE <= 0.6: return "CNA" if SSA > 0.95 else "CA"
            elif 0.4 < FMF <= 0.6 and 0.6 < AE <= 1.2: return "MNA" if SSA > 0.95 else "MA"
            elif FMF > 0.6 and AE > 1.2:
                if SSA > 0.95: return "FNA"
                elif 0.9 <= SSA < 0.95: return "BCL"
                elif 0.85 <= SSA < 0.9: return "BCM"
                else: return "BCH"
            return "Unclassified"

        def classify_source(row):
            aod = row['AOD_Extinction-Total[500nm]']
            ae = row['AE']
            if 0.2 <= aod <= 0.4 and ae > 1: return "Urban"
            elif aod < 0.3 and 0.5 <= ae <= 1.7: return "Maritime"
            elif aod > 0.4 and ae < 1: return "Desert"
            elif aod > 0.7 and ae > 1: return "Biomass"
            elif aod > 0.45 and ae > 1.2: return "Arid"
            else: return "Unclassified"

        merged_df['Label'] = merged_df.apply(classify_aerosol, axis=1)
        merged_df['Source'] = merged_df.apply(classify_source, axis=1)

        # 7. Add Season
        def get_season(day):
            if 1 <= day <= 60 or 335 <= day <= 366: return "winter"
            elif 61 <= day <= 152: return "Pre monsoon"
            elif 153 <= day <= 244: return "Monsoon"
            elif 245 <= day <= 334: return "Post monsoon"
            return "Unknown"
            
        merged_df['Season'] = merged_df['Day_of_Year'].apply(get_season)

        # Save Final CSV
        final_path = os.path.join(site_out_dir, f'{site_name}_Processed.csv')
        merged_df.to_csv(final_path, index=False)
        print(f"Saved processed data to {final_path}")
        
        # Generate and Save Basic Plots (from Exp 2)
        self.generate_exp2_plots(merged_df, site_name, site_out_dir)
        
        return merged_df

    def generate_exp2_plots(self, df, site_name, save_dir):
        """Generates the visualization plots from Exp 2 and saves them."""
        # Plot 1: AOD vs AE by Label
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df, x="AE", y="AOD_Extinction-Total[500nm]", hue="Label", alpha=0.7)
        plt.title(f"AOD vs AE (Label) - {site_name}")
        plt.savefig(os.path.join(save_dir, 'plot_AOD_AE_Label.png'))
        plt.close()

        # Plot 2: Source Distribution
        plt.figure(figsize=(10,6))
        df['Source'].value_counts().plot(kind='barh')
        plt.title(f"Source Distribution - {site_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plot_Source_Dist.png'))
        plt.close()

    def run_ml_analysis(self, df, site_name):
        """
        Logic from DA-Exp3: Train ML models and export metrics.
        """
        print(f"--- Running ML Analysis for: {site_name} ---")
        site_out_dir = os.path.join(self.output_dir, site_name)
        
        # Prepare Data
        drop_cols = ["Time(hh:mm:ss)", "Date(dd:mm:yyyy)", "Site", "Day_of_Year"]
        # Drop columns not needed for ML features
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # We need to predict Source (or Label). Let's stick to 'Source' as per Exp3 logic mostly
        if 'Source' not in df.columns: return
        
        y = df['Source']
        X = X.drop(columns=['Source', 'Label', 'Season'], errors='ignore') # Ensure targets aren't in features
        
        # Filter rare classes (min 5 samples required for splits)
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]

        if len(y.unique()) < 2:
            print("Not enough classes for ML classification.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Preprocessing Pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(exclude=[np.number]).columns

        # --- CRITICAL FIX: sparse_threshold=0 forces dense output for models like GaussianNB ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
            ],
            sparse_threshold=0 
        )

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Naive Bayes": GaussianNB()
        }

        results = []

        for name, model in models.items():
            try:
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                results.append({
                    'Site': site_name,
                    'Model': name,
                    'Accuracy': acc,
                    'Macro F1': report['macro avg']['f1-score'],
                    'Weighted F1': report['weighted avg']['f1-score']
                })
                
                # Save Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name} CM - {site_name}')
                plt.ylabel('True')
                plt.xlabel('Pred')
                plt.savefig(os.path.join(site_out_dir, f'ML_CM_{name}.png'))
                plt.close()
            except Exception as e:
                print(f"Error training {name} for {site_name}: {e}")

        # Save Metrics to CSV
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(site_out_dir, f'{site_name}_ML_Metrics.csv'), index=False)
            print(f"ML Metrics saved for {site_name}")

    def execute(self):
        sites = self.find_sites()
        if not sites:
            print("No matching file sets found in input directory.")
            return

        print(f"Found {len(sites)} sites to process.")
        
        for site, files in sites.items():
            try:
                # Step 1: Process (Exp 2)
                processed_df = self.process_site_data(site, files)
                
                # Step 2: ML (Exp 3)
                if processed_df is not None:
                    self.run_ml_analysis(processed_df, site)
                    
            except Exception as e:
                print(f"FATAL ERROR processing {site}: {e}")
                # Continue to next site even if this one fails
                continue

if __name__ == "__main__":
    # Create the pipeline and run it
    # Ensure your raw csv files are in a folder named 'raw_data' next to this script
    pipeline = AerosolPipeline(input_dir='raw_data', output_dir='output')
    pipeline.execute()