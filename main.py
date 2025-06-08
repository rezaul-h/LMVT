import argparse
import yaml
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.train.engine import Trainer
from src.train.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="LMVT Lung Cancer Classification Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], default='train',
                        help="Pipeline mode: 'train' to train, 'eval' to evaluate, 'both' to do both sequentially")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for evaluation')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index for cross-validation evaluation')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)

    # Setup logging
    log_dir = config['logging']['log_dir']
    log_file = config['logging']['log_file']
    setup_logger(log_dir, log_file)

    # Training phase
    if args.mode in ['train', 'both']:
        trainer = Trainer(args.config)
        trainer.run()

    # Evaluation phase
    if args.mode in ['eval', 'both']:
        if args.checkpoint is None:
            raise ValueError("Evaluation mode requires --checkpoint to be specified.")
        evaluate(args.config, args.checkpoint, fold=args.fold)

if __name__ == "__main__":
    main()