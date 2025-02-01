import pandas as pd
import os
from pathlib import Path


def process_metrics_files(root_directory):
    """
    Process all average_metrics.csv files in subdirectories and combine them with folder names as separators.

    Args:
        root_directory (str): Path to the root directory containing subdirectories

    Returns:
        pd.DataFrame: Combined DataFrame with folder names as separator rows
    """
    # Initialize an empty list to store individual DataFrames
    dfs = []

    # Walk through all subdirectories
    for folder_path in root_directory.rglob('*'):
        if folder_path.is_dir():
            csv_path = folder_path / 'average_metrics.csv'

            # Check if the CSV exists in this folder
            if csv_path.exists():
                # Create a separator DataFrame with folder name
                relative_path = folder_path.relative_to(root_directory)
                separator_df = pd.DataFrame([{
                    # Fill all columns with None except first one with folder name
                    col: None for col in pd.read_csv(csv_path, nrows=0).columns
                }])
                separator_df.iloc[0, 0] = f"=== {relative_path} ==="

                # Read the CSV file
                metrics_df = pd.read_csv(csv_path)

                # Append both DataFrames to our list
                dfs.append(separator_df)
                dfs.append(metrics_df)

    # Combine all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        raise FileNotFoundError("No average_metrics.csv files found in subdirectories")


def main():
    try:
        # Get the current working directory or specify your root directory
        root_dir = Path(r"")

        # Process all files
        result_df = process_metrics_files(root_dir)

        # Save the combined results to a CSV file in the root directory
        output_path = root_dir / 'combined_metrics.csv'
        result_df.to_csv(output_path, index=False)

        print(f"Successfully processed metrics files. Results saved to {output_path}")

    except Exception as e:
        print(f"Error processing files: {str(e)}")


if __name__ == "__main__":
    main()
