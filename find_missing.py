import os

def find_missing_ml_results(output_dir='output'):
    if not os.path.exists(output_dir):
        print(f"Error: The directory '{output_dir}' does not exist.")
        return

    # Get a list of all subdirectories (regions) in the output folder
    all_regions = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    missing_regions = []

    print(f"scanning {len(all_regions)} folders...\n")
    print("--- REGIONS MISSING ML RESULTS ---")

    for region in all_regions:
        # Construct the expected path to the results file
        expected_file = os.path.join(output_dir, region, 'ML_Results.csv')
        
        # Check if file exists
        if not os.path.exists(expected_file):
            print(f"❌ {region}")
            missing_regions.append(region)

    print("-" * 40)
    print(f"Total Missing: {len(missing_regions)} / {len(all_regions)}")
    
    if len(missing_regions) > 0:
        print("\nReason: These regions likely had fewer than 2 source classes (e.g., only 'Maritime')")
        print("and were skipped by the pipeline logic: `if len(df_ml['Source'].unique()) < 2`")

if __name__ == "__main__":
    find_missing_ml_results()