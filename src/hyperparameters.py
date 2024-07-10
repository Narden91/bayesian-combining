import io
import sys
import warnings

import optuna
from catboost import CatBoostClassifier
from optuna.pruners import NopPruner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import optuna
from xgboost import XGBClassifier


def hyperparameter_tuning(model_name, X_train, y_train, X_val, y_val, n_trials=100,
                          scoring='accuracy', verbose=False):
    """
    Perform hyperparameter tuning for a given model using Optuna.

    Args:
        model_name: The model to be tuned.
        X_train (numpy.ndarray or pandas.DataFrame): The training input features.
        y_train (numpy.ndarray or pandas.Series): The training target variable.
        X_val (numpy.ndarray or pandas.DataFrame): The validation input features.
        y_val (numpy.ndarray or pandas.Series): The validation target variable.
        n_trials (int, optional): The number of trials (iterations) for the Optuna study. Default is 100.
        scoring (str, optional): The scoring metric to use for model evaluation. Default is 'accuracy'.
        verbose (bool, optional): Whether to display the Optuna logs. Default is False.

    Returns:
        tuple: A tuple containing the best hyperparameters, the best model, and the validation score.
    """

    def objective(trial):
        if model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                max_depth=trial.suggest_int('max_depth', 2, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4)
            )
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int('max_depth', 2, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4)
            )
        elif model_name == "LogisticRegression":
            model = LogisticRegression(
                C=trial.suggest_float('C', 1e-6, 1e6, log=True),
                solver=trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                max_iter=trial.suggest_int('max_iter', 100, 1000)
            )
        elif model_name == "SVC":
            model = SVC(
                C=trial.suggest_loguniform('C', 1e-6, 1e6),
                kernel=trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto'])
            )
        elif model_name == "MLP":
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', ['50', '100', '50,50', '100,50'])
            model = MLPClassifier(
                hidden_layer_sizes=tuple(map(int, hidden_layer_sizes.split(','))),
                activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
                solver=trial.suggest_categorical('solver', ['adam', 'sgd']),
                alpha=trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                learning_rate=trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                max_iter=trial.suggest_int('max_iter', 200, 1000)
            )
        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                iterations=trial.suggest_int('iterations', 100, 1000),
                depth=trial.suggest_int('depth', 2, 12),
                learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-1),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-6, 1e2),
                verbose=False  # Disable CatBoost's own verbosity
            )
        elif model_name == "XGB":
            model=XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 2, 12),
                learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-1),
                gamma=trial.suggest_float('gamma', 0, 1),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
                subsample=trial.suggest_float('subsample', 0.5, 1),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1),
                verbosity=0
            )
        else:
            raise ValueError("Invalid model type.")

            # Suppress stdout and stderr during model fitting
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if X_val is None or y_val is None:
            return model.score(X_train, y_train)
        else:
            return model.score(X_val, y_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    best_hyperparameters = study.best_params
    val_score = study.best_value

    # Create the best model with the optimal hyperparameters
    if model_name == "RandomForest":
        best_model = RandomForestClassifier(**best_hyperparameters)
    elif model_name == "DecisionTree":
        best_model = DecisionTreeClassifier(**best_hyperparameters)
    elif model_name == "LogisticRegression":
        best_model = LogisticRegression(**best_hyperparameters)
    elif model_name == "SVC":
        best_model = SVC(**best_hyperparameters)
    elif model_name == "MLP":
        hidden_layer_sizes = tuple(map(int, best_hyperparameters['hidden_layer_sizes'].split(',')))
        best_hyperparameters['hidden_layer_sizes'] = hidden_layer_sizes
        best_model = MLPClassifier(**best_hyperparameters)
    elif model_name == "CatBoost":
        best_model = CatBoostClassifier(**best_hyperparameters)
    elif model_name == "XGB":
        best_model = XGBClassifier(**best_hyperparameters)
    else:
        raise ValueError(f"{model_name} is an invalid model choice.")

    # Fit the best model on the entire training set
    best_model.fit(X_train, y_train)

    return best_hyperparameters, best_model, val_score
