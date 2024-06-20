import optuna
from optuna.pruners import NopPruner
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import optuna


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
        tuple: A tuple containing the best hyperparameters and the best model.
    """

    def objective(trial):
        if model_name == "RandomForest":
            hyperparameter_ranges = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
        elif model_name == "DecisionTree":
            hyperparameter_ranges = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
        else:
            raise ValueError("Invalid model type. Choose 'rf' for Random Forest or 'dt' for Decision Tree.")

        hyperparameters = {
            param_name: param_range
            for param_name, param_range in hyperparameter_ranges.items()
        }

        model.set_params(**hyperparameters)

        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)

        return val_score

    if model_name == "RandomForest":
        model = RandomForestClassifier()
    elif model_name == "DecisionTree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid model. Please choose 'RandomForestClassifier' or 'DecisionTreeClassifier'.")

    optuna.logging.set_verbosity(optuna.logging.WARNING) if verbose else None
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_hyperparameters = study.best_params
    best_model = model.set_params(**best_hyperparameters)

    return best_hyperparameters, best_model
