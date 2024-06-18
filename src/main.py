import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Retrieve paths from the config
    source_path = cfg.paths.source
    output_path = cfg.paths.output

    # Use os.path to ensure the path is correct for the operating system
    source_path = os.path.normpath(source_path)
    output_path = os.path.normpath(output_path)

    # Log paths
    logging.info(f"Source Path: {source_path}")
    logging.info(f"Output Path: {output_path}")


if __name__ == "__main__":
    main()
