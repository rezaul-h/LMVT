import argparse
from src.train.engine import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.run()
