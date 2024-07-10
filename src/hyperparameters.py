import optuna
from optuna.pruners import NopPruner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
        elif model_name == "LogisticRegression":
            hyperparameter_ranges = {
                'C': trial.suggest_loguniform('C', 1e-6, 1e6),
                'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
            }
        elif model_name == "SVC":
            hyperparameter_ranges = {
                'C': trial.suggest_loguniform('C', 1e-6, 1e6),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
        elif model_name == "KNN":
            hyperparameter_ranges = {
                'n_neighbors': trial.suggest_int('n_neighbors', 2, 10),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
        elif model_name == "MLP":
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', ['50', '100', '50,50', '100,50'])
            hyperparameter_ranges = {
                'hidden_layer_sizes': tuple(map(int, hidden_layer_sizes.split(','))),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': trial.suggest_int('max_iter', 200, 1000)
            }
        else:
            raise ValueError("Invalid model type. Choose 'rf' for Random Forest or 'dt' for Decision Tree.")

        hyperparameters = {
            param_name: param_range
            for param_name, param_range in hyperparameter_ranges.items()
        }

        model.set_params(**hyperparameters)
        model.fit(X_train, y_train)

        if X_val is None or y_val is None:
            return model.score(X_train, y_train)
        else:
            return model.score(X_val, y_val)

    if model_name == "RandomForest":
        model = RandomForestClassifier()
    elif model_name == "DecisionTree":
        model = DecisionTreeClassifier()
    elif model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "SVC":
        model = SVC()
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "MLP":
        model = MLPClassifier()
    else:
        raise ValueError("Invalid model choice.")

    optuna.logging.set_verbosity(optuna.logging.WARNING) # if verbose else None
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_hyperparameters = study.best_params
    val_score = study.best_value

    if model_name == "MLP":
        best_hyperparameters['hidden_layer_sizes'] = tuple(
            map(int, best_hyperparameters['hidden_layer_sizes'].split(',')))

    best_model = model.set_params(**best_hyperparameters)

    return best_hyperparameters, best_model, val_score
