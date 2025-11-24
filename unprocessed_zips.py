import os
import glob

def check_unprocessed_zips(input_dir='raw_data', output_dir='output'):
    print(f"=== Checking for Unprocessed Zip Files ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")

    # 1. Get all zip files in raw_data
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    
    if not zip_files:
        print(f"No .zip files found in {input_dir}")
        return

    missing_count = 0
    print("--- The following zip files have NOT been processed (No Output Folder) ---")

    for zip_path in zip_files:
        # Extract site name exactly as the pipeline does
        # e.g., "raw_data/20020101_Ji_Parana.zip" -> "20020101_Ji_Parana"
        site_name = os.path.splitext(os.path.basename(zip_path))[0]
        
        # Construct the expected output path
        expected_output_path = os.path.join(output_dir, site_name)
        
        # Check if the folder exists
        if not os.path.exists(expected_output_path):
            print(f" [MISSING] {os.path.basename(zip_path)}")
            missing_count += 1

    print("-" * 30)
    if missing_count == 0:
        print("All zip files have corresponding output folders! \u2705")
    else:
        print(f"Total Unprocessed Files: {missing_count}")

if __name__ == "__main__":
    check_unprocessed_zips()