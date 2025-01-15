import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import re


def parse_execution_time(file_path: Path) -> Optional[float]:
    """
    Parse execution time from the file and convert it to seconds
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        time_pattern = r'(\d{2}):(\d{2}):(\d{2}\.\d{2})'
        match = re.search(time_pattern, content)

        if match:
            hours, minutes, seconds = match.groups()
            total_seconds = (
                    int(hours) * 3600 +
                    int(minutes) * 60 +
                    float(seconds)
            )
            return round(total_seconds, 2)

        return None

    except Exception as e:
        print(f"Error parsing execution time from {file_path}: {str(e)}")
        return None


def find_average_metrics_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all average_metrics.csv files in the directory structure
    """
    return list(root_dir.glob('**/average_metrics.csv'))


def extract_info(file_path: Path) -> Tuple[str, str, str]:
    """
    Extract model, base classifier, and approach information from the file path
    """
    parts = list(file_path.parts)
    try:
        combined_idx = parts.index('Combined')
        model = parts[combined_idx + 1]
        base_classifier = parts[combined_idx + 2]
        approach = parts[combined_idx + 3]
        return model, base_classifier, approach
    except (ValueError, IndexError):
        print(f"Warning: Could not extract full path information from {file_path}")
        return "Unknown", "Unknown", "Unknown"


def aggregate_metrics(root_dir: Path) -> pd.DataFrame:
    """
    Aggregate metrics from all average_metrics.csv files and execution times
    """
    metrics_files = find_average_metrics_files(root_dir)
    all_results = []

    print(f"Found {len(metrics_files)} metrics files")

    for metrics_file in metrics_files:
        try:
            print(f"\nProcessing file: {metrics_file}")

            # Extract information from path
            model, base_classifier, approach = extract_info(metrics_file)
            print(f"Extracted info: Model={model}, Classifier={base_classifier}, Approach={approach}")

            # Find corresponding execution time file
            execution_time_file = metrics_file.parent / 'Execution_time.txt'
            execution_time = None
            if execution_time_file.exists():
                execution_time = parse_execution_time(execution_time_file)
                print(f"Execution time found: {execution_time} seconds")

            # Read the CSV file and set the index column as Technique
            df = pd.read_csv(metrics_file, index_col=0)

            # For each row in the metrics file
            for technique, row in df.iterrows():
                # Calculate execution time per run if both values are available
                execution_time_per_run = None
                if execution_time is not None and row['Run_Executed'] > 0:
                    execution_time_per_run = round(execution_time / row['Run_Executed'], 2)

                result = {
                    'Model': model,
                    'Base_Classifier': base_classifier,
                    'Approach': approach,
                    'Technique': technique,
                    'Accuracy': row['Accuracy'],
                    'Precision': row['Precision'],
                    'Sensitivity': row['Sensitivity'],
                    'Specificity': row['Specificity'],
                    'F1_Score': row['F1_Score'],
                    'MCC': row['MCC'],
                    'Run_Executed': row['Run_Executed'],
                    'Execution_Time_Seconds': execution_time,
                    'Execution_Time_Per_Run': execution_time_per_run
                }
                all_results.append(result)

        except Exception as e:
            print(f"Error processing file {metrics_file}: {str(e)}")
            continue

    if not all_results:
        print("Warning: No results were collected!")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def main():
    # Set the root directory - use absolute path
    root_dir = Path(r"C:\Users\Emanuele\Documents\ProgettiPython\bayesian-combining\output\Combined")

    # Verify the path exists
    if not root_dir.exists():
        print(f"Error: The path {root_dir} does not exist")
        return

    # Get output directory path
    output_dir = Path(r"C:\Users\Emanuele\Documents\ProgettiPython\bayesian-combining\output\Combined_aggregated")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics
    print(f"\nProcessing files from: {root_dir}")
    results_df = aggregate_metrics(root_dir)

    if results_df.empty:
        print("No results to save!")
        return

    # Save results
    output_file = output_dir / 'all_metrics_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # # Print summary statistics grouped by Model, Base_Classifier, Approach, and Technique
    # print("\nSummary Statistics:")
    # summary = results_df.groupby(['Model', 'Base_Classifier', 'Approach', 'Technique']).agg({
    #     'Accuracy': ['mean', 'std'],
    #     'F1_Score': ['mean', 'std'],
    #     'MCC': ['mean', 'std'],
    #     'Execution_Time_Seconds': ['mean', 'std']
    # }).round(4)
    # print(summary)
    #
    # # Print basic statistics
    # print(f"\nProcessed {len(results_df)} entries from "
    #       f"{len(results_df.groupby(['Model', 'Base_Classifier', 'Approach']))} unique combinations")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
