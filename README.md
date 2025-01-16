# Bayesian Combining

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation for combining multiple machine learning models using Bayesian Networks. This project provides tools for model stacking, feature importance analysis, and automated hyperparameter tuning using Bayesian optimization.

## Features

- Bayesian Network model stacking
- Automated hyperparameter optimization
- Feature importance analysis
- Multiple classifier support (Random Forest, SVM, XGBoost, etc.)
- Comprehensive performance metrics and visualizations
- Support for both sequential and parallel processing

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment management tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Narden91/bayesian-combining.git
cd bayesian-combining
```

2. Create and activate a Python virtual environment:

**Linux/macOS:**
```bash
python -m venv env
source env/bin/activate
```

**Windows:**
```bash
python -m venv env
env\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses YAML configuration files located in the `config/` directory:
- `config.yaml`: Main configuration file for model parameters
- Custom configurations can be added for different experiments

## Usage

### Basic Usage
```bash
python main.py
```

### Parallel Processing
```bash
python main_multiprocessing.py
```

### Configuration Options
```bash
python main.py --config path/to/custom_config.yaml
```

## Project Structure

The project is organized as follows. Each Python module in the `src` directory has a specific responsibility:
```
bayesian-combining/
├── config/              # Configuration files
│   └── config.yaml     # Main configuration file
├── data/               # Data directory
├── output/             # Output directory for results
├── src/               # Source code
│   ├── aggregate_results_combining.py
│   ├── bayesian_net_importance_score.py
│   ├── bayesian_net_importance.py
│   ├── bayesian_network.py
│   ├── classification.py
│   ├── explainability.py
│   ├── hyperparameters.py
│   ├── importance_tracker.py
│   ├── main_backup.py
│   ├── main_multiprocessing.py
│   ├── main_process.py
│   ├── main.py
│   ├── preprocessing.py
│   ├── results_analysis.py
│   ├── task_analysis.py
│   └── utils.py
├── .gitignore         # Git ignore file
├── bayesian_folder_conv.py
├── LICENSE           # License file
├── organize_results.ipynb  # Jupyter notebook for results organization
├── README.md         # Project documentation
└── requirements.txt  # Project dependencies
```

### Key Components

- **Bayesian Network Implementation**:
  - `bayesian_network.py`: Core implementation of Bayesian Network model stacking
  - `bayesian_net_importance.py`: Feature importance analysis using Bayesian Networks
  - `bayesian_net_importance_score.py`: Scoring mechanisms for Bayesian Network features

- **Model Management**:
  - `classification.py`: Implementation of various classification models
  - `hyperparameters.py`: Hyperparameter optimization using Optuna
  - `preprocessing.py`: Data preprocessing and feature engineering

- **Analysis and Tracking**:
  - `importance_tracker.py`: Tracks feature importance across experiments
  - `results_analysis.py`: Analysis of experimental results
  - `explainability.py`: Model explainability tools
  - `task_analysis.py`: Task-specific analysis utilities

- **Core Processing**:
  - `main.py`: Main entry point for sequential processing
  - `main_multiprocessing.py`: Entry point for parallel processing
  - `main_process.py`: Core processing logic
  - `utils.py`: Utility functions used across the project

- **Additional Tools**:
  - `organize_results.ipynb`: Jupyter notebook for organizing and visualizing results
  - `bayesian_folder_conv.py`: Utilities for folder structure conversion

## Development

### Adding New Dependencies
When installing new packages, update requirements.txt:
```bash
pip freeze > requirements.txt
```

### Running Tests
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

- TODO

## Contact

Emanuele Nardone - emanuele.nardone@unicas.it

Project Link: [https://github.com/Narden91/bayesian-combining](https://github.com/Narden91/bayesian-combining)
