import models.model_A.XGBoost as model_A
import models.model_A.feature_engineering.get_dataset as model_A_prep_data
import models.model_B.full_pipeline as model_B
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os
import logging
import numpy as pd
import random
import torch
import numpy as np

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main pipeline function with Hydra configuration."""
    # Print the configuration
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    set_seed(cfg.random_seed)

    # Load dataset based on config
    log.info(f"Loading dataset: {cfg.dataset.name}")
    # Dataset loading code will depend on your dataset format
    # This is a placeholder for the actual loading code

    # Initialize and run the selected model
    if cfg.model.type == "model_A":
        log.info(f"Running model_A (XGBoost) with configuration: {cfg.model}")
        # Call model_A functions with the config
        # model = model_A.YourFunction(cfg.model.params)
        model_A_prep_data.run_full_pipeline()
        model_A.full_pipeline()
    elif cfg.model.type == "model_B":
        log.info(f"Running model_B (Neural Net) with configuration: {cfg.model}")
        # Convert DictConfig to a nested dictionary for compatibility
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_B.full_pipeline(model_config)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    log.info(f"Experiment '{cfg.experiment_name}' completed successfully")


if __name__ == "__main__":
    main()
