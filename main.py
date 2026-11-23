"""
main.py
Runs the steps of the pipeline from the command line.

Examples:
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --train --evaluate
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="House price ML pipeline")
    parser.add_argument("--data", default="kc_house_data.csv", help="CSV path")
    parser.add_argument("--model-path", default="model.joblib", help="model file")
    parser.add_argument("--prepare", action="store_true", help="load and split data")
    parser.add_argument("--train", action="store_true", help="train and save model")
    parser.add_argument("--evaluate", action="store_true", help="evaluate saved model")
    return parser.parse_args()


def main():
    args = parse_args()

    if not (args.prepare or args.train or args.evaluate):
        print("Nothing to do. Use --prepare, --train and/or --evaluate.")
        return

    # Same seed in prepare_data() -> same split on every run,
    # so --evaluate always tests on rows the model never saw.
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.prepare:
        print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")

    if args.train:
        print("Training model...")
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)

    if args.evaluate:
        try:
            model = load_model(args.model_path)
        except FileNotFoundError as err:
            print(err)
            return
        metrics = evaluate_model(model, X_test, y_test)
        print(f"R2   : {metrics['r2']:.4f}")
        print(f"MAE  : {metrics['mae']:,.0f}")
        print(f"RMSE : {metrics['rmse']:,.0f}")


if __name__ == "__main__":
    main()
