import os
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import pandas as pd
import utils
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

    for run in tqdm.tqdm(range(num_runs), desc="Run"):
        logging.info(f"Run {run+1}") if verbose else None
        logging.info(f"Seed: {seed}") if verbose else None

        for file in tqdm.tqdm(file_list, desc="CSV file"):
            task_df = pd.read_csv(file, sep=cfg.data.separator)
            train_df, test_df = prep.data_split(task_df, cfg.data.target, cfg.experiment.test_size, seed, verbose)
            train_df, val_df = prep.data_split(train_df, cfg.data.target, cfg.experiment.val_size, seed, verbose)

            logging.info(f"Train shape: {train_df.shape}") if verbose else None
            logging.info(f"Test shape: {test_df.shape}") if verbose else None
            logging.info(f"Validation shape: {val_df.shape}") if verbose else None

            if data_type == "ML":
                logging.info("Task: Machine Learning") if verbose else None

            break


if __name__ == "__main__":
    main()
