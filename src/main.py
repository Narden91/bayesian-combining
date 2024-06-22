import os
import random

import graphviz
import hydra
import numpy as np
from omegaconf import DictConfig
import logging
from pathlib import Path
import pandas as pd
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator, PC, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
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
    global_seed = cfg.settings.seed
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

    sample_df = pd.read_csv(file_list[0], sep=cfg.data.separator)
    num_subjects = sample_df.shape[0]
    num_tasks = len(file_list)

    # Stacking placeholders
    first_level_train_preds = np.zeros((num_subjects, num_tasks))
    first_level_test_preds = np.zeros((int(num_subjects * cfg.experiment.test_size)+1, num_tasks))

    first_level_train_preds = pd.DataFrame(index=sample_df[cfg.data.id],
                                           columns=[f'Task_{i + 1}' for i in range(num_tasks)])
    first_level_train_preds_proba = pd.DataFrame(index=sample_df[cfg.data.id],
                                                 columns=[f'Task_{i + 1}' for i in range(num_tasks)])
    first_level_test_preds = pd.DataFrame(index=None,
                                          columns=[cfg.data.id] + [f'Task_{i + 1}' for i in range(num_tasks)])
    first_level_test_preds_proba = pd.DataFrame(index=None,
                                                columns=[cfg.data.id] + [f'Task_{i + 1}' for i in range(num_tasks)])

    # if verbose:
    #     logging.info(f"First level train preds shape: {first_level_train_preds.shape}")
    #     logging.info(f"First level train preds proba shape: {first_level_train_preds_proba.shape}")
    #     logging.info(f"First level test preds shape: {first_level_test_preds.shape}")
    #     logging.info(f"First level test preds proba shape: {first_level_test_preds_proba.shape}")

    logging.info(f"Bayesian Network Analysis Experiment starting...") if verbose else None
    logging.info(f"Number of runs: {num_runs}") if verbose else None
    logging.info(f"Tasks: {num_tasks}") if verbose else None
    logging.info(f"Data Type: {data_type} ") if verbose else None
    logging.info(f"Number of Folds: {cfg.experiment.folds}") if verbose else None

    for run in range(num_runs):
        logging.info(f"------------------------------------------------") if verbose else None
        run_folder = output_path / f"run_{run + 1}"
        seed = global_seed + run
        logging.info(f"Run {run + 1}") if verbose else None
        logging.info(f"Seed: {seed}") if verbose else None

        for file_idx, file in enumerate(file_list):
            task_df = pd.read_csv(file, sep=cfg.data.separator)
            train_df, test_df = prep.data_split(task_df, cfg.data.target, cfg.experiment.test_size, seed, verbose)

            # sort by Id column
            train_df.sort_values(by=cfg.data.id, inplace=True)
            test_df.sort_values(by=cfg.data.id, inplace=True)

            # set Id column as index and call it Id_index but keep it as a column
            train_df.set_index(cfg.data.id, inplace=True, drop=False)
            train_df.index.name = cfg.data.id_index

            test_df.set_index(cfg.data.id, inplace=True, drop=False)
            test_df.index.name = cfg.data.id_index

            # Check correct split
            test_ids = test_df[cfg.data.id].values
            train_ids = train_df[cfg.data.id].values
            common_ids = np.intersect1d(test_ids, train_ids)
            if common_ids.size > 0:
                raise ValueError(f"Test set contains Ids also present in the train set: {common_ids}")

            # if verbose:
            #     logging.info(f"Train shape: {train_df.shape}")
            #     logging.info(f"Train set: \n {train_df}")
            #     logging.info(f"Test shape: {test_df.shape}")
            #     logging.info(f"Test set: \n {test_df}")

            X_train = train_df.drop(columns=cfg.data.target)
            y_train = train_df[cfg.data.target]

            X_test = test_df.drop(columns=cfg.data.target)
            y_test = test_df[cfg.data.target]

            best_val_score = 0
            best_model = None
            scaler_opt = None

            # Stratified K-Fold Cross Validation to avoid class imbalance into folds
            skf = StratifiedKFold(n_splits=cfg.experiment.folds, shuffle=True, random_state=seed)

            for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
                logging.info(f"--------- Fold {fold}--------------") if verbose else None

                X_train_cv, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

                # Preprocess the data
                if data_type == "ML":
                    X_train_cv, scaler_cv = prep.data_scaling(X_train_cv, cfg.data.id, cfg.scaling.type, verbose)
                    X_val = prep.apply_scaling(X_val, scaler_cv, cfg.data.id, verbose)

                # Hyperparameter tuning using optuna
                best_hyperparameters, best_model_cv, val_score = hp.hyperparameter_tuning(cfg.model.name, X_train_cv,
                                                                                          y_train_cv,
                                                                                          X_val, y_val,
                                                                                          n_trials=cfg.optuna.n_trials,
                                                                                          verbose=verbose)

                val_pred = best_model_cv.predict(X_val)
                val_proba = best_model_cv.predict_proba(X_val)[:, 1]
                first_level_train_preds.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_pred
                # first_level_train_preds.loc[X_train.index[val_index], cfg.data.id] = X_val.index.to_list()
                first_level_train_preds_proba.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_proba
                # first_level_train_preds_proba.loc[X_train.index[val_index], cfg.data.id] = X_val.index.to_list()

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = best_model_cv
                    scaler_opt = scaler_cv if data_type == "ML" else None

                if verbose:
                    logging.info(f"Best hyperparameters: {best_hyperparameters}")
                    logging.info(f"Best model: {best_model_cv}")
                    logging.info(f"Validation score: {val_score}")

                seed_val += 1

            if data_type == "ML":
                X_test = prep.apply_scaling(X_test, scaler_opt, cfg.data.id, verbose)

            best_model.fit(X_train, y_train)
            y_pred_test = best_model.predict(X_test)
            first_level_test_preds[cfg.data.id] = test_df.index.to_list()
            first_level_test_preds[f'Task_{file_idx + 1}'] = y_pred_test

            y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
            first_level_test_preds_proba[cfg.data.id] = test_df.index.to_list()
            first_level_test_preds_proba[f'Task_{file_idx + 1}'] = y_pred_test_proba

            break

        # Clean predictions
        first_level_train_preds = utils.clean_predictions(first_level_train_preds, cfg.data.id)
        first_level_train_preds['Label'] = y_train.to_numpy()
        first_level_train_preds_proba = utils.clean_predictions(first_level_train_preds_proba, cfg.data.id)
        first_level_train_preds_proba['Label'] = y_train.to_numpy()

        first_level_test_preds['Label'] = y_test.to_numpy()
        first_level_test_preds.sort_values(by=cfg.data.id, inplace=True)

        first_level_test_preds_proba['Label'] = y_test.to_numpy()
        first_level_test_preds_proba.sort_values(by=cfg.data.id, inplace=True)

        # Prepare stacking data
        stacking_trainings_data = first_level_train_preds.drop(cfg.data.id, axis=1)
        stacking_trainings_data_proba = first_level_train_preds_proba.drop(cfg.data.id, axis=1)
        stacking_data_predictions = first_level_test_preds.drop(cfg.data.id, axis=1)
        stacking_data_probabilities = first_level_test_preds_proba.drop(cfg.data.id, axis=1)

        logging.info(f"Stacking trainings data: \n {stacking_trainings_data}") if verbose else None
        logging.info(f"Stacking trainings data proba: \n {stacking_trainings_data_proba}") if verbose else None
        logging.info(f"Stacking test data predictions: \n {stacking_data_predictions}") if verbose else None
        logging.info(f"Stacking test data probabilities: \n {stacking_data_probabilities}") if verbose else None

        # Function to randomly fill NaN with 0 or 1 TODO: Debug, eliminare
        # def random_binary_fill(x):
        #     return x if pd.notnull(x) else random.choice([0, 1])
        #
        # # Apply random filling to all columns except 'Label'
        # for col in stacking_data_predictions.columns:
        #     if col != cfg.data.target:
        #         stacking_data_predictions[col] = stacking_data_predictions[col].apply(random_binary_fill)
        #
        # logging.info(f"Data: \n {stacking_data_predictions}") if verbose else None
        #
        # # Ensure all columns are of type int
        # for col in stacking_data_predictions.columns:
        #     stacking_data_predictions[col] = stacking_data_predictions[col].astype(int)
        #
        # # Build Bayesian Network
        # logging.info("\nBuilding Bayesian Network...")
        # hc = HillClimbSearch(stacking_data_predictions)
        # best_model = hc.estimate()
        #
        # logging.info("\nBest model edges:")
        # logging.info(best_model.edges())
        #
        # model = BayesianNetwork(best_model.edges())
        #
        # # Fit the parameters
        # model.fit(stacking_data_predictions, estimator=BayesianEstimator, prior_type="BDeu")
        #
        # # Create a Graphviz object
        # dot = graphviz.Digraph(comment=f'Bayesian Network {run+1}')
        # dot.attr(rankdir='LR')
        #
        # # Add nodes
        # for node in model.nodes():
        #     if node == 'Label':
        #         dot.node(node, node, shape='diamond')
        #     else:
        #         dot.node(node, node)
        #
        # # Add edges
        # for edge in model.edges():
        #     dot.edge(edge[0], edge[1])
        #
        # bn_path = run_folder / f"bayesian_network_{run+1}"
        #
        # # Render the graph
        # dot.render(bn_path, format='png', cleanup=True)
        # logging.info("Bayesian Network saved")
        #
        # # Print network structure
        # logging.info("\nNetwork Structure:")
        # for edge in model.edges():
        #     logging.info(f"{edge[0]} -> {edge[1]}")
        #
        # # Analyze network
        # logging.info("\nNode degrees:")
        # for node in model.nodes():
        #     in_degree = len(model.get_parents(node))
        #     out_degree = len(model.get_children(node))
        #     logging.info(f"{node}: In-degree = {in_degree}, Out-degree = {out_degree}")
        #
        # # Find Markov blanket for Label
        # logging.info("\nMarkov blanket for Label:")
        # label_markov_blanket = model.get_markov_blanket('Label')
        # logging.info(f"Label: {label_markov_blanket}")
        #
        # # Select columns based on Markov blanket
        # selected_columns = list(label_markov_blanket) + ['Label']
        # stacking_data_predictions_selected = stacking_data_predictions[selected_columns]
        #
        # # Perform inference
        # logging.info("\nPerforming inference...")
        # inference = VariableElimination(model)
        #
        # # Make predictions using majority vote
        # predictions = []
        # for _, row in stacking_data_predictions_selected.iterrows():
        #     evidence = {col: row[col] for col in label_markov_blanket}
        #     prediction = inference.map_query(['Label'], evidence=evidence)
        #     predictions.append(prediction['Label'])
        #
        # # Calculate accuracy
        # accuracy = accuracy_score(stacking_data_predictions_selected['Label'], predictions)
        # logging.info(f"\nAccuracy: {accuracy}")

        seed += 1
        seed_val = 0

        break


if __name__ == "__main__":
    main()



