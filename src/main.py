import os
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import utils
import hyperparameters as hp
import preprocessing as prep
import tqdm


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # region Folder Paths
    project_root = Path(__file__).parent.parent
    data_parent_path = project_root / Path(cfg.paths.source)
    output_path = project_root / Path(cfg.paths.output)
    data_type = cfg.data.type
    data_folder = data_parent_path / data_type
    # endregion

    # region Experiment Settings
    num_runs = cfg.settings.runs
    seed = cfg.settings.seed
    seed_val = 0
    verbose = cfg.settings.verbose

    # endregion

    # region Folder Check
    if not data_folder.exists():
        logging.error(f"Data folder {data_folder} does not exist.")
        return

    if not output_path.exists():
        os.makedirs(output_path)
    # endregion

    # Load the CSV files
    file_list = utils.load_csv_file(data_folder, cfg.data.extension)

    for run in range(num_runs):
        logging.info(f"Run {run + 1}") if verbose else None
        logging.info(f"Seed: {seed}") if verbose else None

        for file in file_list:
            task_df = pd.read_csv(file, sep=cfg.data.separator)
            train_df, test_df = prep.data_split(task_df, cfg.data.target, cfg.experiment.test_size, seed, verbose)
            # train_df, val_df = prep.data_split(train_df, cfg.data.target, cfg.experiment.val_size, seed, verbose)

            logging.info(f"Train shape: {train_df.shape}") if verbose else None
            logging.info(f"Test shape: {test_df.shape}") if verbose else None
            # logging.info(f"Validation shape: {val_df.shape}") if verbose else None

            X_train = train_df.drop(columns=cfg.data.target)
            y_train = train_df[cfg.data.target]

            X_test = test_df.drop(columns=cfg.data.target)
            y_test = test_df[cfg.data.target]

            best_val_score = 0
            best_model = None
            scaler_opt = None

            # Cross-validation loop
            for fold in range(cfg.experiment.folds):
                logging.info(f"Fold {fold + 1}") if verbose else None
                # Split the data into train and validation sets
                X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train, y_train,
                                                                        test_size=cfg.experiment.val_size,
                                                                        random_state=seed, stratify=y_train)
                # Preprocess the data
                if data_type == "ML":
                    logging.info("Task: Machine Learning") if verbose else None
                    X_train_cv, scaler_cv = prep.data_scaling(X_train_cv, cfg.data.id, cfg.scaling.type, verbose)
                    X_val = prep.apply_scaling(X_val, scaler_cv, cfg.data.id, verbose)

                # Hyperparameter tuning using optuna
                best_hyperparameters, best_model_cv, val_score = hp.hyperparameter_tuning(cfg.model.name, X_train_cv,
                                                                                          y_train_cv,
                                                                                          X_val, y_val,
                                                                                          n_trials=cfg.optuna.n_trials,
                                                                                          verbose=verbose)

                logging.info(f"Best hyperparameters: {best_hyperparameters}") if verbose else None
                logging.info(f"Best model: {best_model_cv}") if verbose else None

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = best_model_cv
                    scaler_opt = scaler_cv if data_type == "ML" else None

                logging.info(f"Validation score: {val_score}") if verbose else None

                seed_val += 1
                break

            X_test = prep.apply_scaling(X_test, scaler_cv, cfg.data.id, verbose)

            best_model.fit(X_train, y_train)
            y_pred_test = best_model.predict(X_test)

            seed += 1
            seed_val = 0

            break

            # if data_type == "ML":
            #     logging.info("Task: Machine Learning") if verbose else None
            #     X_train = prep.data_scaling(X_train, cfg.data.id, cfg.scaling.type, verbose)
            #     X_test = prep.data_scaling(X_test, cfg.data.id, cfg.scaling.type, verbose)
            # break


if __name__ == "__main__":
    main()
