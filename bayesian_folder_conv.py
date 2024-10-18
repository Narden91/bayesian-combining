from pathlib import Path


def adjust_folder_structure(base_path):
    correct_structure = {
        'DecisionTree': ['Bayesian', 'LogisticRegression_stacking', 'MajorityVote', 'MLP_stacking',
                         'WeightedMajorityVote'],
        'RandomForest': ['Bayesian', 'LogisticRegression_stacking', 'MajorityVote', 'MLP_stacking',
                         'WeightedMajorityVote'],
        'SVC': ['Bayesian', 'LogisticRegression_stacking', 'MajorityVote', 'MLP_stacking', 'WeightedMajorityVote'],
        'XGB': ['Bayesian', 'LogisticRegression_stacking', 'MajorityVote', 'MLP_stacking', 'WeightedMajorityVote']
    }

    def get_base_classifier_name(folder_name):
        return next((c for c in correct_structure.keys() if folder_name.startswith(c)), None)

    def get_base_method_name(folder_name):
        folder_name_lower = folder_name.lower()
        for methods in correct_structure.values():
            for method in methods:
                if method.lower() == folder_name_lower.replace('pure_', '').replace('_clf', ''):
                    return method
        return None

    for cnn_folder in base_path.iterdir():
        if cnn_folder.is_dir():
            print(f"Adjusting structure for {cnn_folder.name}")

            # Rename classifier folders
            for item in list(cnn_folder.iterdir()):
                if item.is_dir():
                    base_name = get_base_classifier_name(item.name)
                    if base_name and item.name != base_name:
                        new_name = cnn_folder / base_name
                        if not new_name.exists():
                            print(f"Renaming {item.name} to {base_name}")
                            item.rename(new_name)
                        else:
                            print(f"Warning: Cannot rename {item.name} to {base_name} as it already exists")

            # Rename method folders within classifier folders
            for classifier_folder in cnn_folder.iterdir():
                if classifier_folder.is_dir() and classifier_folder.name in correct_structure:
                    for method_folder in list(classifier_folder.iterdir()):
                        if method_folder.is_dir():
                            base_method_name = get_base_method_name(method_folder.name)
                            if base_method_name and method_folder.name != base_method_name:
                                new_method_name = classifier_folder / base_method_name
                                if not new_method_name.exists():
                                    print(
                                        f"Renaming {method_folder.name} to {base_method_name} in {classifier_folder.name}")
                                    method_folder.rename(new_method_name)
                                else:
                                    print(
                                        f"Warning: Cannot rename {method_folder.name} to {base_method_name} in {classifier_folder.name} as it already exists")

            # Check for missing folders and print warnings
            for classifier, methods in correct_structure.items():
                classifier_path = cnn_folder / classifier
                if not classifier_path.exists():
                    print(f"Warning: {classifier} folder is missing in {cnn_folder.name}")
                else:
                    existing_methods = set(folder.name for folder in classifier_path.iterdir() if folder.is_dir())
                    for method in methods:
                        if method not in existing_methods and not any(
                                get_base_method_name(existing) == method for existing in existing_methods):
                            print(f"Warning: {method} folder is missing in {cnn_folder.name}/{classifier}")

    print("Folder structure adjustment completed.")


# Usage
# base_path = Path.home() / 'Desktop' / 'DL'
base_path = Path('/output/ML')

# check the path existance
if base_path.exists():
    print(f"Ok")
else:
    raise FileExistsError

adjust_folder_structure(base_path)
