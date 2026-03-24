import argparse
import copy
import os
import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model
# -----------------------------
class SimpleCIFARNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# -----------------------------
# Utilities
# -----------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def load_model_from_state(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


# -----------------------------
# Non-IID partitioning (Dirichlet)
# -----------------------------
def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 100,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Smaller alpha => more non-IID.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))

    while True:
        client_indices = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            class_idx = np.where(labels == c)[0]
            rng.shuffle(class_idx)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            class_splits = np.split(class_idx, split_points)

            for client_id, idxs in enumerate(class_splits):
                client_indices[client_id].extend(idxs.tolist())

        lengths = [len(v) for v in client_indices.values()]
        if min(lengths) >= min_samples_per_client:
            break

    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    return client_indices


def split_train_val(indices: List[int], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    indices = indices.copy()
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    return indices[:split], indices[split:]


# -----------------------------
# Data
# -----------------------------
def load_cifar10_data(data_dir: str = "./data"):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    eval_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, eval_train_dataset, test_dataset


def make_client_loaders(
    train_dataset,
    eval_train_dataset,
    num_clients: int,
    batch_size: int,
    alpha: float,
    seed: int,
) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader], Dict[int, int]]:
    labels = np.array(train_dataset.targets)
    partitions = dirichlet_partition(labels, num_clients=num_clients, alpha=alpha, seed=seed)

    client_train_loaders = {}
    client_val_loaders = {}
    client_sizes = {}

    for cid, idxs in partitions.items():
        train_idxs, val_idxs = split_train_val(idxs, val_ratio=0.2, seed=seed + cid)

        client_train_loaders[cid] = DataLoader(
            Subset(train_dataset, train_idxs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        client_val_loaders[cid] = DataLoader(
            Subset(eval_train_dataset, val_idxs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        client_sizes[cid] = len(train_idxs)

    return client_train_loaders, client_val_loaders, client_sizes


# -----------------------------
# Training/Evaluation
# -----------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    algorithm: str = "fedavg",
    mu: float = 0.0,
    global_state: Dict[str, torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_params = None
    if algorithm.lower() == "fedprox" and global_state is not None:
        global_params = {k: v.to(device) for k, v in global_state.items()}

    epoch_loss_sum = 0.0
    epoch_batches = 0

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if algorithm.lower() == "fedprox" and global_params is not None and mu > 0.0:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    prox_term += torch.sum((param - global_params[name]) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_batches += 1

    avg_train_loss = epoch_loss_sum / max(epoch_batches, 1)
    return clone_state_dict(model.state_dict()), avg_train_loss


def weighted_average(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    total_weight = sum(weights)
    avg_state = OrderedDict()

    for key in states[0].keys():
        avg_state[key] = sum(state[key] * (w / total_weight) for state, w in zip(states, weights))

    return avg_state


def sample_clients(num_clients: int, fraction: float, seed: int, round_num: int) -> List[int]:
    rng = np.random.default_rng(seed + round_num)
    num_selected = max(1, int(num_clients * fraction))
    selected = rng.choice(np.arange(num_clients), size=num_selected, replace=False)
    return selected.tolist()


# -----------------------------
# Main FL loop
# -----------------------------
def federated_train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_dataset, eval_train_dataset, test_dataset = load_cifar10_data(args.data_dir)

    client_train_loaders, client_val_loaders, client_sizes = make_client_loaders(
        train_dataset=train_dataset,
        eval_train_dataset=eval_train_dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=args.seed,
    )

    global_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    global_model = SimpleCIFARNet()
    global_state = clone_state_dict(global_model.state_dict())

    round_test_accs = []
    round_test_losses = []
    round_client_val_accs = []
    round_train_losses = []

    for rnd in range(1, args.rounds + 1):
        print(f"\n--- Round {rnd}/{args.rounds} ---")
        selected_clients = sample_clients(args.num_clients, args.client_fraction, args.seed, rnd)
        print(f"Selected clients: {selected_clients}")

        client_states = []
        client_weights = []
        local_losses = []

        for cid in selected_clients:
            local_model = SimpleCIFARNet()
            local_model = load_model_from_state(local_model, global_state, device)

            updated_state, local_loss = train_local(
                model=local_model,
                loader=client_train_loaders[cid],
                device=device,
                epochs=args.local_epochs,
                lr=args.lr,
                algorithm=args.algorithm,
                mu=args.mu,
                global_state=global_state,
            )

            client_states.append(updated_state)
            client_weights.append(client_sizes[cid])
            local_losses.append(local_loss)

        global_state = weighted_average(client_states, client_weights)
        global_model = load_model_from_state(global_model, global_state, device)

        test_loss, test_acc = evaluate(global_model, global_test_loader, device)

        client_val_accs = []
        for cid in range(args.num_clients):
            _, val_acc = evaluate(global_model, client_val_loaders[cid], device)
            client_val_accs.append(val_acc)

        avg_local_loss = float(np.mean(local_losses))
        avg_client_val_acc = float(np.mean(client_val_accs))

        round_train_losses.append(avg_local_loss)
        round_test_losses.append(test_loss)
        round_test_accs.append(test_acc)
        round_client_val_accs.append(avg_client_val_acc)

        print(
            f"Round {rnd} | "
            f"Avg Local Train Loss: {avg_local_loss:.4f} | "
            f"Global Test Loss: {test_loss:.4f} | "
            f"Global Test Acc: {test_acc:.4f} | "
            f"Avg Client Val Acc: {avg_client_val_acc:.4f}"
        )

    # Final per-client evaluation
    print("\n=== Final Per-Client Validation Accuracy ===")
    final_client_metrics = {}
    for cid in range(args.num_clients):
        val_loss, val_acc = evaluate(global_model, client_val_loaders[cid], device)
        final_client_metrics[cid] = {"val_loss": val_loss, "val_acc": val_acc}
        print(f"Client {cid}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Save model
    model_path = os.path.join(args.output_dir, f"{args.algorithm}_global_model.pth")
    torch.save(global_state, model_path)
    print(f"\nSaved global model to: {model_path}")

    # Plot metrics
    plot_metrics(
        round_train_losses,
        round_test_losses,
        round_test_accs,
        round_client_val_accs,
        args.output_dir,
        args.algorithm,
    )

    return final_client_metrics


def plot_metrics(
    round_train_losses: List[float],
    round_test_losses: List[float],
    round_test_accs: List[float],
    round_client_val_accs: List[float],
    output_dir: str,
    algorithm: str,
) -> None:
    rounds = list(range(1, len(round_test_accs) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_train_losses, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Avg Local Train Loss")
    plt.title(f"{algorithm.upper()} - Avg Local Train Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{algorithm}_train_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_test_losses, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Global Test Loss")
    plt.title(f"{algorithm.upper()} - Global Test Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{algorithm}_test_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_test_accs, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Global Test Accuracy")
    plt.title(f"{algorithm.upper()} - Global Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{algorithm}_test_acc.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_client_val_accs, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Average Client Validation Accuracy")
    plt.title(f"{algorithm.upper()} - Client-wise Generalization")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{algorithm}_client_val_acc.png"))
    plt.close()

    print(f"Saved plots in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Federated Learning on CIFAR-10 (FedAvg/FedProx)")
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--client_fraction", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha; smaller => more non-IID")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    federated_train(args)