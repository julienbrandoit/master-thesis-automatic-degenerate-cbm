import os
import pandas as pd
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

def concat_csv_files(input_folder, output_file):
    """
    Concatenate all CSV files in the input folder and save to the output file.

    Parameters:
    - input_folder: str, the folder containing the input CSV files
    - output_file: str, the file to save the concatenated output
    """
    logging.info(f"Scanning folder: {input_folder} for CSV files.")

    # Collect all CSV file paths
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        logging.error("No CSV files found in the specified input folder.")
        return

    logging.info(f"Found {len(csv_files)} CSV files to concatenate.")

    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        try:
            logging.info(f"Reading file: {file}")
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            logging.error(f"Failed to read {file}. Error: {e}")
            continue

    if not dataframes:
        logging.error("No valid CSV files could be read. Exiting.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logging.info("All files successfully concatenated.")

    # Save the concatenated dataframe to the output file
    try:
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Concatenated data saved to: {output_file} successfully. The number of shape is {combined_df.shape} with columns {combined_df.columns}.")
    except Exception as e:
        logging.error(f"Failed to save concatenated data to {output_file}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate CSV files from a folder.")
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help="Folder containing the input CSV files")
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help="File to save the concatenated output")

    args = parser.parse_args()
    concat_csv_files(args.input_folder, args.output_file)
