import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

import torch

from src.util.train_utils import load_config, set_seeds
from src.models.t_and_s import TimeSpaceAttnModel
from src.runtime.train import Trainer
from src.runtime.test import Tester

sys.path.append(str(Path(__file__).resolve().parent.parent))

def setup_logging(save_dir: Path):
    """Sets up logging to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(save_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main(args):

    # Create experiment folder & logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name if args.name else "experiment"
    save_dir = Path(args.output_dir) / f"{exp_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Saving results to: {save_dir}")

    # Get configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    local_config = load_config(config_path=str(config_path))
    with open(save_dir / "config.json", "w") as f:
        json.dump(local_config, f, indent=4)
    logger.info("Configuration saved.")
    set_seeds(args.seed, tag_mlflow=False)
    logger.info(f"Seed set to: {args.seed}")

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    logger.info(f"Running on device: {device}")
    model = TimeSpaceAttnModel(local_config)
    model.to(device)

    # Initialize Trainer
    run_name = f"{model.__class__.__name__}_{timestamp}"
    trainer = Trainer(model=model, config=local_config, run_name=run_name, run_output_dir=save_dir)

    # Training Loop
    logger.info("Starting training...")
    try:
        trained_model = trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return

    # Save Final Model
    if local_config.get("save_model", True):
        model_save_path = save_dir / "model_final.pt"
        torch.save(trained_model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    # Testing
    logger.info("Starting testing...")
    tester = Tester(trained_model, local_config, trainer.configure_dataset)
    results = tester.test()

    # Save Results
    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    logger.info(f"Test finished. Results saved to {results_path}")
    logger.info(f"Final Results: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TimeSpaceAttnModel Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Root directory for saving experiments.")
    parser.add_argument( "--name", type=str, default="run",
                         help="Optional name/tag for this experiment run.")
    parser.add_argument("--seed", type=int, default=333, help="Random seed for reproducibility.")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU usage (if available).")
    args = parser.parse_args()
    main(args)
