#!/usr/bin/env python3

import argparse
import pandas as pd

def subset_main_file(main_csv_path, test_csv_path, output_path=None):
    # Load the CSV files
    main_df = pd.read_csv(main_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Convert the 'x' column from test file to a list
    patient_ids = test_df['x'].dropna().astype(str).tolist()

    # Subset the main file using DEID
    subset_df = main_df[main_df['DEID'].astype(str).isin(patient_ids)]

    # Display or save the result
    print(f"Subset contains {len(subset_df)} records.")
    
    if output_path:
        subset_df.to_csv(output_path, index=False)
        print(f"Subset saved to {output_path}")
    else:
        print(subset_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset main CSV file using patient IDs from test file")
    parser.add_argument("--main_csv", required=True, help="Path to the main CSV file containing DEID")
    parser.add_argument("--test_csv", required=True, help="Path to the test CSV file containing column x")
    parser.add_argument("--output", help="Optional path to save the subsetted CSV")

    args = parser.parse_args()
    subset_main_file(args.main_csv, args.test_csv, args.output)
