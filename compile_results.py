import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compile_model_results(output_dir='output'):
    print("=== Compiling ML Results from all Regions ===")
    
    # 1. Find all ML_Results.csv files inside subdirectories of output
    # This looks for: output/ANY_FOLDER/ML_Results.csv
    search_path = os.path.join(output_dir, "*", "ML_Results.csv")
    files = glob.glob(search_path)
    
    if not files:
        print("No ML_Results.csv files found. Make sure the pipeline has finished successfully.")
        return

    all_results = []

    # 2. Iterate and Read
    for filepath in files:
        # Get the parent folder name (which serves as the Site ID)
        # e.g., "output/20020101_20251231_Ji_Parana_UNIR/ML_Results.csv" -> "20020101_20251231_Ji_Parana_UNIR"
        folder_name = os.path.basename(os.path.dirname(filepath))
        
        # Optional: Clean the site name to remove dates if the format is consistent
        # Splits by underscore and takes parts from index 2 onwards (removes dates)
        try:
            # Assuming format: Date_Date_SiteName
            clean_name = "_".join(folder_name.split('_')[2:])
            if not clean_name: clean_name = folder_name # Fallback
        except:
            clean_name = folder_name

        try:
            df = pd.read_csv(filepath)
            
            # Add the site name to this dataframe
            df['Region'] = clean_name
            
            all_results.append(df)
            print(f"  Loaded results for: {clean_name}")
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")

    if not all_results:
        print("No valid data extracted.")
        return

    # 3. Combine all dataframes
    # Current shape: Long format (Region, Model, Accuracy)
    combined_df = pd.concat(all_results, ignore_index=True)

    # 4. Pivot to Wide format (Rows=Region, Cols=Models, Values=Accuracy)
    final_table = combined_df.pivot(index='Region', columns='Model', values='Accuracy')
    
    # Reset index so Region is a proper column, not the index
    final_table.reset_index(inplace=True)
    
    # Fill NaNs (if a specific model failed for a specific region)
    final_table = final_table.fillna(0)

    # 5. Save to CSV
    final_csv_path = 'Final_Model_Performance_Comparison.csv'
    final_table.to_csv(final_csv_path, index=False)
    print(f"\nSUCCESS: Comparison table saved to: {final_csv_path}")
    print("-" * 30)
    print(final_table.to_string())

    # --- OPTIONAL: VISUALIZATION ---
    # Create a Heatmap for easy visual comparison
    create_heatmap(final_table, 'Region')

def create_heatmap(df, index_col):
    """
    Generates a heatmap where darker colors = higher accuracy
    """
    # Set Region as index for plotting
    plot_data = df.set_index(index_col)
    
    plt.figure(figsize=(12, len(plot_data) * 0.8 + 2)) # Dynamic height based on N regions
    sns.heatmap(plot_data, annot=True, cmap='viridis', fmt='.2f', vmin=0, vmax=1)
    plt.title("Model Accuracy Comparison by Region")
    plt.ylabel("Region")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig('Final_Comparison_Heatmap.png')
    print("Heatmap saved to: Final_Comparison_Heatmap.png")

if __name__ == "__main__":
    compile_model_results()