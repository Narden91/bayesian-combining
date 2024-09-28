from pathlib import Path
import pandas as pd
import logging
from collections import Counter
import re
import csv


def count_tasks_in_mb(bayesian_clf_folder):
    task_counts = Counter()
    total_elements = 0
    run_count = 0

    # Iterate over run_x folders inside Bayesian_stacking_clf
    for run_dir in bayesian_clf_folder.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            # Extract the run number
            match = re.match(r'run_(\d+)', run_dir.name)
            if match:
                run_number = match.group(1)
                blanket_file = run_dir / f"Markov_Blanket_{run_number}.txt"
                # Check if the Markov_Blanket_{run_number}.txt file exists
                if blanket_file.is_file():
                    run_count += 1
                    with open(blanket_file, 'r') as file:
                        elements = 0
                        # Read tasks from the file and count occurrences
                        for line in file:
                            line = line.strip()
                            # Use regex to extract task names without predefined knowledge
                            task_match = re.match(r'Task_\d+_[\w\d]+', line)
                            if task_match:
                                task_name = task_match.group(0)
                                task_counts[task_name] += 1
                                elements += 1
                        total_elements += elements
                else:
                    print(f"    Warning: {blanket_file} does not exist.")

    # Calculate average number of elements per run
    average_elements = total_elements / run_count if run_count > 0 else 0

    return task_counts, average_elements


def count_tasks_in_bayesian_clf(bayesian_clf_folder):
    task_counts = Counter()
    task_prefix = "Task_"
    task_range = range(1, 26)  # Task_1 to Task_25
    task_names = [f"{task_prefix}{i}" for i in task_range]
    total_elements = 0
    run_count = 0

    # Iterate over run_x folders inside Bayesian_stacking_clf
    for run_dir in bayesian_clf_folder.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            # Extract the run number
            match = re.match(r'run_(\d+)', run_dir.name)
            if match:
                run_number = match.group(1)
                blanket_file = run_dir / f"Markov_Blanket_{run_number}.txt"
                # Check if the Markov_Blanket_{run_number}.txt file exists
                if blanket_file.is_file():
                    run_count += 1
                    with open(blanket_file, 'r') as file:
                        elements = 0
                        # Read tasks from the file and count occurrences
                        for line in file:
                            line = line.strip()
                            if line in task_names:
                                task_counts[line] += 1
                                elements += 1
                        total_elements += elements
                else:
                    print(f"    Warning: {blanket_file} does not exist.")

    # Calculate average number of elements per run
    average_elements = total_elements / run_count if run_count > 0 else 0

    return task_counts, average_elements


def save_results_to_csv(results, filename):
    # Determine the CSV columns: start with subfolder details, then tasks from Task_1 to Task_25
    fieldnames = ['Main_Subfolder', 'Subfolder', 'Average_Elements'] + [f"Task_{i}" for i in range(1, 26)]

    # Open the file in write mode and write the header and data rows
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Fill missing task occurrences with 0
            for task in fieldnames[3:]:
                if task not in result:
                    result[task] = 0
            writer.writerow(result)


def traverse_and_count_tasks_separately(root_folder):
    root_path = Path(root_folder)
    results = []  # List to store results for CSV

    # Iterate over each main subfolder in the root directory
    for main_subfolder in root_path.iterdir():
        if main_subfolder.is_dir():
            print(f"\nAnalyzing tasks in: {main_subfolder.name}")

            # Analyze each subfolder inside the main subfolder
            for subfolder in main_subfolder.iterdir():
                if subfolder.is_dir():
                    print(f"  Subfolder: {subfolder.name}")
                    bayesian_clf_folder = subfolder / "Bayesian_stacking_clf"
                    # Check if the Bayesian_stacking_clf folder exists
                    if bayesian_clf_folder.is_dir():
                        task_counts, average_elements = count_tasks_in_bayesian_clf(bayesian_clf_folder)

                        # Collect results in a dictionary
                        if task_counts:
                            result = {
                                "Main_Subfolder": main_subfolder.name,
                                "Subfolder": subfolder.name,
                                "Average_Elements": average_elements
                            }
                            for task in sorted(task_counts.keys(), key=lambda x: int(x.split('_')[1])):
                                result[task] = task_counts[task]
                                print(f"    {task}: {task_counts[task]}")
                            print(f"    Average Elements: {average_elements}")
                            results.append(result)
                        else:
                            print(f"    No tasks found in Bayesian_stacking_clf under {subfolder.name}.")
                    else:
                        print(f"    Bayesian_stacking_clf folder not found in {subfolder.name}.")

    # Save results to a CSV file
    save_results_to_csv(results, 'task_occurrences.csv')


# def result_analysis(base_dir: Path, data_type: str = "ML") -> int:
#     """
#     Analyzes the results of the experiments and saves CSV files.
#     :param base_dir:
#     :param data_type:
#     :return:
#     """
#     output_dir = base_dir / Path("output_analysis") / Path(data_type)
#
#     if not output_dir.exists():
#         output_dir.mkdir(parents=True)
#
#     base_dir = base_dir / Path(data_type)
#
#     # Check if the base directory exists
#     if not base_dir.exists() or not base_dir.is_dir():
#         logging.error(f"The provided base path {base_dir} is not valid.")
#         return
#
#     # Iterate through the first level: CNN models like InceptionResNetV2
#     for dataset in base_dir.iterdir():
#         if dataset.is_dir():
#             logging.info(f"Found Dataset directory: {dataset.name}")
#
#             dataset_output_dir = output_dir / dataset.name
#             if not dataset_output_dir.exists():
#                 dataset_output_dir.mkdir(parents=True)
#
#             # Iterate through the second level: Classifiers like DecisionTree_base_clf
#             for classifier in dataset.iterdir():
#                 if classifier.is_dir():
#                     logging.info(f"  Found classifier directory: {classifier.name}")
#
#                     # Iterate through the third level: Methods like Bayesian_stacking_clf
#                     for method in classifier.iterdir():
#                         if method.is_dir():
#                             logging.info(f"    Found method directory: {method.name}")
#
#                             # Iterate through the runs or files within method directories
#                             for item in method.iterdir():
#                                 if item.is_file() and item.name == 'average_metrics.txt':
#                                     logging.info(f"      Found metrics file: {item.name}")
#                                     df = parse_average_metrics(item)
#                                     # display_dataframe(df)
#
#                                     # Save the DataFrame as CSV
#                                     csv_filename = f"{classifier.name}_{method.name}.csv"
#                                     csv_path = dataset_output_dir / csv_filename
#
#                                     # filter columns Accuracy  Precision  Sensitivity  Specificity    Score      MCC
#                                     # df = df[['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'Score', 'MCC']]
#                                     display_dataframe(df)
#                                     df.to_csv(csv_path)
#                                     logging.info(f"      Saved CSV file: {csv_path}")
#                         # break
#                 break
#         break


def result_analysis(base_dir: Path, data_type: str = "ML") -> int:
    """
    Analyzes the results of the experiments and saves CSV files.
    :param base_dir: Base directory containing the experiment results.
    :param data_type: Type of data (e.g., "ML").
    :return: None
    """
    output_dir = base_dir / Path("output_analysis") / Path(data_type)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    base_dir = base_dir / Path(data_type)

    # Check if the base directory exists
    if not base_dir.exists() or not base_dir.is_dir():
        logging.error(f"The provided base path {base_dir} is not valid.")
        return

    # Iterate through the first level: CNN models like InceptionResNetV2
    for dataset in base_dir.iterdir():
        if dataset.is_dir():
            logging.info(f"Found Dataset directory: {dataset.name}")

            dataset_output_dir = output_dir / dataset.name
            if not dataset_output_dir.exists():
                dataset_output_dir.mkdir(parents=True)

            # Iterate through the second level: Classifiers like DecisionTree_base_clf
            for classifier in dataset.iterdir():
                if classifier.is_dir():
                    # logging.info(f"  Found classifier directory: {classifier.name}")

                    # Create an empty DataFrame with the specified columns
                    columns = ["Methods",
                               "accuracy", "precision", "sensitivity", "specificity", "f1_score", "mcc",
                               "Accuracy", "Precision", "Sensitivity", "Specificity", "Score", "MCC"
                               ]
                    classifier_df = pd.DataFrame(columns=columns)

                    # Iterate through the third level: Methods like Bayesian_stacking_clf
                    for method in classifier.iterdir():
                        if method.is_dir():
                            # logging.info(f"    Found method directory: {method.name}")

                            # Iterate through the runs or files within method directories
                            for item in method.iterdir():
                                if item.is_file() and item.name == 'average_metrics.txt':
                                    # logging.info(f"      Found metrics file: {item.name}")
                                    df = parse_average_metrics(item)

                                    # logging.info(f"      Displaying DataFrame:")
                                    # display_dataframe(df)

                                    # # Add the method name as a new column to identify the method
                                    df['Method_file'] = method.name

                                    # Concatenate the current df with classifier_df
                                    classifier_df = pd.concat([classifier_df, df], ignore_index=True)

                        # Save the DataFrame as CSV
                    csv_filename = f"{classifier.name}.csv"
                    csv_path = dataset_output_dir / csv_filename
                    classifier_df.to_csv(csv_path, index=False)
                    logging.info(f"  Saved CSV file: {csv_path}")


def parse_csv(file_path: Path) -> pd.DataFrame:
    """
    Parses the CSV file into a DataFrame.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


def calculate_task_accuracy(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the accuracy of each task column based on the 'Label' column.
    :param df: DataFrame containing task columns and a 'Label' column.
    :return: A Series containing accuracy of each task.
    """
    accuracies = {}
    label = df['Label']

    # Iterate over task columns (assuming columns are named Task_1, Task_2, ...)
    for task in [col for col in df.columns if col.startswith("Task_")]:
        task_predictions = df[task]
        accuracy = (task_predictions == label).mean()  # Calculate accuracy
        accuracies[task] = accuracy

    return pd.Series(accuracies)


def result_analysis_tasks(base_dir: Path, data_type: str = "ML") -> int:
    """
    Analyzes the results of the experiments and saves CSV files.
    :param base_dir:
    :param data_type:
    :return:
    """
    output_dir = base_dir / Path("output_analysis_tasks") / Path(data_type)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    base_dir = base_dir / Path(data_type)

    # Check if the base directory exists
    if not base_dir.exists() or not base_dir.is_dir():
        logging.error(f"The provided base path {base_dir} is not valid.")
        return

    # Iterate through the first level: Datasets
    for dataset in base_dir.iterdir():
        if dataset.is_dir():
            logging.info(f"Found Dataset directory: {dataset.name}")

            dataset_output_dir = output_dir / dataset.name
            if not dataset_output_dir.exists():
                dataset_output_dir.mkdir(parents=True)

            # Iterate through the second level: Classifiers
            for classifier in dataset.iterdir():
                if classifier.is_dir():
                    # logging.info(f"  Found classifier directory: {classifier.name}")

                    # Initialize DataFrames to collect accuracies for all runs
                    test_accuracies_df = pd.DataFrame()
                    training_accuracies_df = pd.DataFrame()

                    # Iterate through the third level: Methods
                    for method in classifier.iterdir():
                        if method.is_dir() and method.name == "Pure_MajorityVote":
                            # logging.info(f"    Found Pure_MajorityVote directory: {method.name}")

                            # Iterate through run folders (e.g., run_1, run_2, ..., run_30)
                            for run_folder in method.iterdir():
                                if run_folder.is_dir() and run_folder.name.startswith("run_"):
                                    # logging.info(f"      Found run directory: {run_folder.name}")

                                    first_level_data = run_folder / "First_level_data"

                                    # Check if First_level_data folder exists
                                    if first_level_data.exists() and first_level_data.is_dir():
                                        test_file = first_level_data / "Test_data.csv"
                                        training_file = first_level_data / "Trainings_data.csv"

                                        # Read Test_data.csv and Trainings_data.csv
                                        if test_file.exists() and training_file.exists():
                                            # logging.info(f"        Reading Test_data.csv and Trainings_data.csv")

                                            test_df = parse_csv(test_file)
                                            training_df = parse_csv(training_file)

                                            # Calculate accuracies for the current run
                                            test_run_accuracies = calculate_task_accuracy(test_df)
                                            training_run_accuracies = calculate_task_accuracy(training_df)

                                            # Add the run number as a column
                                            run_number = int(run_folder.name.split('_')[-1])
                                            test_run_accuracies['Run'] = run_number
                                            training_run_accuracies['Run'] = run_number

                                            # Append accuracies as new rows in the overall DataFrame
                                            test_accuracies_df = pd.concat(
                                                [test_accuracies_df, test_run_accuracies.to_frame().T],
                                                ignore_index=True
                                            )
                                            training_accuracies_df = pd.concat(
                                                [training_accuracies_df, training_run_accuracies.to_frame().T],
                                                ignore_index=True
                                            )

                    # Calculate the mean accuracy for each task and add it as a new row
                    test_mean_accuracy = test_accuracies_df.mean(numeric_only=True)
                    test_mean_accuracy['Run'] = 'Mean_Accuracy'
                    test_accuracies_df = pd.concat([test_accuracies_df, test_mean_accuracy.to_frame().T],
                                                   ignore_index=True)

                    training_mean_accuracy = training_accuracies_df.mean(numeric_only=True)
                    training_mean_accuracy['Run'] = 'Mean_Accuracy'
                    training_accuracies_df = pd.concat([training_accuracies_df, training_mean_accuracy.to_frame().T],
                                                       ignore_index=True)

                    # Move the Run column to the first position and round values to 4 decimal places
                    test_accuracies_df = test_accuracies_df[
                        ['Run'] + [col for col in test_accuracies_df.columns if col != 'Run']]
                    training_accuracies_df = training_accuracies_df[
                        ['Run'] + [col for col in training_accuracies_df.columns if col != 'Run']]

                    def convert_run_value(run):
                        try:
                            return int(run)  # Attempt to convert to integer
                        except ValueError:
                            return run  # Leave as string if conversion fails (e.g., 'Mean_Accuracy')

                    # Apply the conversion function to the Run column
                    test_accuracies_df['Run'] = test_accuracies_df['Run'].apply(convert_run_value)
                    training_accuracies_df['Run'] = training_accuracies_df['Run'].apply(convert_run_value)

                    # Separate DataFrames into numeric and non-numeric 'Run' values
                    test_numeric = test_accuracies_df[test_accuracies_df['Run'].apply(lambda x: isinstance(x, int))]
                    test_non_numeric = test_accuracies_df[
                        ~test_accuracies_df['Run'].apply(lambda x: isinstance(x, int))]

                    training_numeric = training_accuracies_df[
                        training_accuracies_df['Run'].apply(lambda x: isinstance(x, int))]
                    training_non_numeric = training_accuracies_df[
                        ~training_accuracies_df['Run'].apply(lambda x: isinstance(x, int))]

                    # Sort numeric DataFrame by 'Run'
                    test_numeric = test_numeric.sort_values(by='Run').reset_index(drop=True)
                    training_numeric = training_numeric.sort_values(by='Run').reset_index(drop=True)

                    # Concatenate sorted numeric DataFrame with non-numeric rows (e.g., Mean_Accuracy)
                    test_accuracies_df = pd.concat([test_numeric, test_non_numeric], ignore_index=True)
                    training_accuracies_df = pd.concat([training_numeric, training_non_numeric], ignore_index=True)

                    # Round task accuracy values to 4 decimal places
                    # test_accuracies_df = test_accuracies_df.round(4)
                    # training_accuracies_df = training_accuracies_df.round(4)

                    # Round task accuracy values to 4 decimal places specifically for task columns
                    task_columns = [col for col in test_accuracies_df.columns if col.startswith('Task_')]

                    # Round values in task columns for both DataFrames
                    test_accuracies_df[task_columns] = test_accuracies_df[task_columns].round(4)
                    training_accuracies_df[task_columns] = training_accuracies_df[task_columns].round(4)

                    # Save the DataFrames to CSV
                    # test_csv_filename = f"{classifier.name}_test_accuracies.csv"
                    # training_csv_filename = f"{classifier.name}_training_accuracies.csv"
                    training_csv_filename = f"{classifier.name}_accuracies.csv"

                    # test_csv_path = dataset_output_dir / test_csv_filename
                    training_csv_path = dataset_output_dir / training_csv_filename

                    # test_accuracies_df.to_csv(test_csv_path, index=False)
                    training_accuracies_df.to_csv(training_csv_path, index=False)

                    # logging.info(f"Saved Test Accuracy CSV file: {test_csv_path}")
                    # logging.info(f"Saved Training Accuracy CSV file: {training_csv_path}")


def parse_average_metrics(file_path):
    """
    Parses the average_metrics.txt file to extract metrics into a DataFrame.
    Each approach (method) will be a row and each metric will be a column.
    """
    metrics = {}

    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_approach = None

    # Parse the file line by line
    for line in lines:
        line = line.strip()
        if line.endswith('Approach:'):
            current_approach = line.replace(' Approach:', '')
            metrics[current_approach] = {}  # Initialize a new dictionary for the current approach
        elif current_approach and ':' in line:
            key, value = line.split(':')
            key = key.strip()
            value = float(value.strip())
            metrics[current_approach][key] = value  # Store the metric value under the current approach

    # logging.info(f"Metrics extracted: {metrics}")

    # Create a DataFrame from the parsed metrics
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df['Methods'] = df.index

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Reorder columns to place 'Methods' first
    cols = ['Methods'] + [col for col in df.columns if col != 'Methods']
    df = df[cols]

    return df


def display_dataframe(df):
    """
    Displays the DataFrame.
    """
    if not df.empty:
        logging.info("\n" + df.to_string(index=True))
