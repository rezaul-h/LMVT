import argparse
from src.train.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LMVT Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fold", type=int, default=None, help="Fold number for cross-validation")
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint, fold=args.fold)
