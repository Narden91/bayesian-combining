# output_management.py
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from omegaconf import DictConfig
import logging
import os
import shutil
from dataclasses import dataclass
import time

# Import from your existing modules
from .importance_tracker import ImportanceTracker
from .bayesian_net_importance import BayesianImportanceTracker
from . import utils


@dataclass
class ExperimentConfig:
    """Configuration for experiment setup"""
    num_runs: int
    root_folder: Path
    stacking_method: str
    bayesian_config: Optional[Dict] = None


class OutputManager:
    """Manages output folder structure and importance trackers"""

    def __init__(self, cfg: DictConfig, verbose: bool = False):
        """
        Initialize the output manager.

        Args:
            cfg: Hydra configuration object
            verbose: Enable verbose logging
        """
        self.cfg = cfg
        self.verbose = verbose
        self._importance_tracker = None
        self._bayesian_tracker = None

    def setup_output_structure(
            self,
            output_paths: List[Path],
            analysis_type: str,
            num_runs: int
    ) -> Tuple[Path, Dict[str, Path]]:
        """
        Set up the complete output folder structure.

        Args:
            output_paths: List of output paths
            analysis_type: Type of analysis being performed
            num_runs: Number of experimental runs

        Returns:
            Tuple containing root output folder and dictionary of all output folders
        """
        try:
            # Get output folders using existing utility function
            output_folders = utils.get_output_folder(output_paths, analysis_type, self.cfg)

            if self.verbose:
                logging.info(f"Output folders: {output_folders}")

            root_folder = self._get_root_folder(output_folders)

            if "combined" in output_folders:
                self._setup_importance_tracking(
                    ExperimentConfig(
                        num_runs=num_runs,
                        root_folder=root_folder,
                        stacking_method=self.cfg.experiment.stacking_method,
                        bayesian_config=self.cfg.bayesian_net if hasattr(self.cfg, 'bayesian_net') else None
                    )
                )

                if self.verbose:
                    self._log_folder_structure(output_folders)

            return root_folder, output_folders

        except Exception as e:
            logging.error(f"Error setting up output structure: {e}")
            raise

    def _get_root_folder(self, output_folders: Dict[str, Path]) -> Path:
        """Determine the root output folder"""
        if len(output_folders) == 1:
            return list(output_folders.values())[0]
        elif "combined" in output_folders:
            root_folder = output_folders["combined"]

            # Handle Bayesian method folder renaming
            if self.cfg.experiment.stacking_method == 'Bayesian':
                if os.path.exists(root_folder):
                    shutil.rmtree(root_folder)
                root_folder = root_folder.with_name(
                    f"{root_folder.name}_{self.cfg.bayesian_net.algorithm}_{self.cfg.bayesian_net.score_metric}_{self.cfg.bayesian_net.prior_type}"
                )

            return root_folder
        else:
            raise ValueError("Multiple output folders found but no combined folder.")

    def _setup_importance_tracking(self, config: ExperimentConfig) -> None:
        """
        Set up appropriate importance tracking based on stacking method.

        Args:
            config: Experiment configuration
        """
        try:
            if config.stacking_method == 'Classification':
                self._setup_classification_tracking(config)
            elif config.stacking_method == 'Bayesian':
                self._setup_bayesian_tracking(config)
            else:
                if self.verbose:
                    logging.warning(f"Importance tracking not supported for stacking method: {config.stacking_method}")

        except Exception as e:
            logging.error(f"Error setting up importance tracking: {str(e)}")

    def _setup_classification_tracking(self, config: ExperimentConfig) -> None:
        """Setup tracking for classification method"""
        importance_dir = config.root_folder / "feature_importance_analysis"
        try:
            os.makedirs(importance_dir, exist_ok=True)
            if self.verbose:
                logging.info(f"Created directory: {importance_dir}")

            self._importance_tracker = ImportanceTracker(importance_dir, config.num_runs)
            if self.verbose:
                logging.info("Successfully initialized importance tracker")

        except Exception as e:
            logging.error(f"Error setting up classification importance tracking: {str(e)}")

    def _setup_bayesian_tracking(self, config: ExperimentConfig) -> None:
        """Setup tracking for Bayesian method"""
        importance_dir = config.root_folder / "bayesian_importance_analysis"
        try:
            os.makedirs(importance_dir, exist_ok=True)
            if self.verbose:
                logging.info(f"Created directory: {importance_dir}")

            self._bayesian_tracker = BayesianImportanceTracker(
                importance_dir,
                config.num_runs
            )
            if self.verbose:
                logging.info("Successfully initialized Bayesian importance tracker")

        except Exception as e:
            logging.error(f"Error setting up Bayesian importance tracking: {str(e)}")

    def _log_folder_structure(self, output_folders: Dict[str, Path]) -> None:
        """Log the folder structure for debugging"""
        for dataset, folder in output_folders.items():
            if dataset != "combined":
                logging.info(f"Output folder for {dataset}: {folder}")

    def create_run_folder(self, root_folder: Path, run_number: int) -> Path:
        """
        Create a folder for a specific run.

        Args:
            root_folder: Root output folder
            run_number: Current run number

        Returns:
            Path to the created run folder
        """
        run_folder = root_folder / f"run_{run_number}"
        os.makedirs(run_folder, exist_ok=True)
        if self.verbose:
            logging.info(f"Created run folder: {run_folder}")
        return run_folder

    @property
    def importance_tracker(self) -> Optional[ImportanceTracker]:
        """Get the classification importance tracker"""
        return self._importance_tracker

    @property
    def bayesian_tracker(self) -> Optional[BayesianImportanceTracker]:
        """Get the Bayesian importance tracker"""
        return self._bayesian_tracker

    def generate_final_analysis(self, target_node: Optional[str] = None) -> None:
        """
        Generate final analysis for active trackers.

        Args:
            target_node: Target node for Bayesian analysis
        """
        try:
            if self._importance_tracker is not None:
                logging.info("Generating final importance analysis")
                self._importance_tracker.generate_final_analysis()

            if self._bayesian_tracker is not None:
                logging.info("Generating final Bayesian Network importance analysis")
                self._bayesian_tracker.generate_final_analysis(target_node=target_node)

        except Exception as e:
            logging.error(f"Error generating final analysis: {str(e)}")
            raise
