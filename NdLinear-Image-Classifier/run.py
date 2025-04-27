import argparse
import torch
import os

from src.model import NdLinearCNN
from src.train import get_data_loaders, training_loop, plot_training_results, plot_confusion_matrix, \
    show_random_predictions


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train NdLinearCNN on SVHN Dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to download/load SVHN data")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir)

    model = NdLinearCNN().to(device)

    # FIX the function call here:
    train_losses, test_accuracies = training_loop(
        model,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    training_results_path = os.path.join(args.output_dir, "training_results.png")
    random_predictions_path = os.path.join(args.output_dir, "random_predictions.png")
    confusion_matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")

    plot_training_results(train_losses, test_accuracies, training_results_path)

    class_names = [str(i) for i in range(10)]
    show_random_predictions(model, test_loader, device, class_names, random_predictions_path)
    plot_confusion_matrix(model, test_loader, device, class_names, confusion_matrix_path)

    print(f"Training complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()