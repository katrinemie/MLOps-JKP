"""
Continual learning med experience replay og EWC påMNIST.
Task 1: Træn påcifre 0-4
Task 2: Naiv træning på5-9 (catastrophic forgetting)
Task 3: Experience replay + EWC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import json
from pathlib import Path


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def filter_by_classes(dataset, classes):
    indices = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, indices)


def evaluate(model, loader, device, classes=None):
    model.eval()
    correct, total = 0, 0
    per_class = {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for c in y.unique():
                c = c.item()
                if classes and c not in classes:
                    continue
                mask = y == c
                per_class.setdefault(c, [0, 0])
                per_class[c][0] += (pred[mask] == y[mask]).sum().item()
                per_class[c][1] += mask.sum().item()
            if classes:
                mask = torch.tensor([yi.item() in classes for yi in y],
                                    device=device)
                correct += (pred[mask] == y[mask]).sum().item()
                total += mask.sum().item()
            else:
                correct += (pred == y).sum().item()
                total += len(y)
    acc = 100 * correct / total if total > 0 else 0
    class_acc = {c: 100 * v[0] / v[1] for c, v in per_class.items()}
    return acc, class_acc


def train_model(model, loader, optimizer, criterion, device, epochs=5):
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


def compute_fisher(model, loader, device, n_samples=500):
    """Beregn Fisher Information Matrix (diagonal approx) for EWC."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    count = 0
    for x, y in loader:
        if count >= n_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        for n, p in model.named_parameters():
            fisher[n] += p.grad.data ** 2
        count += len(x)
    for n in fisher:
        fisher[n] /= count
    return fisher


def train_with_replay_ewc(model, new_loader, buffer, optimizer, criterion,
                           device, fisher, old_params, ewc_lambda=400,
                           epochs=5, replay_batch=32):
    """Træn med experience replay + EWC regularization."""
    model.train()
    for ep in range(epochs):
        for x, y in new_loader:
            x, y = x.to(device), y.to(device)

            # Mix replay samples ind
            if len(buffer) > 0:
                replay_x, replay_y = zip(*random.sample(
                    buffer, min(replay_batch, len(buffer))
                ))
                replay_x = torch.stack(replay_x).to(device)
                replay_y = torch.tensor(replay_y).to(device)
                x = torch.cat([x, replay_x])
                y = torch.cat([y, replay_y])

            optimizer.zero_grad()
            loss = criterion(model(x), y)

            # EWC penalty
            ewc_loss = 0
            for n, p in model.named_parameters():
                ewc_loss += (fisher[n] * (p - old_params[n]) ** 2).sum()
            loss += (ewc_lambda / 2) * ewc_loss

            loss.backward()
            optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST("data", train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST("data", train=False, transform=transform)

    task1_classes = [0, 1, 2, 3, 4]
    task2_classes = [5, 6, 7, 8, 9]

    task1_train = filter_by_classes(train_data, task1_classes)
    task2_train = filter_by_classes(train_data, task2_classes)
    task1_loader = DataLoader(task1_train, batch_size=64, shuffle=True)
    task2_loader = DataLoader(task2_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    results = {}

    # ========== TASK 1: Træn påcifre 0-4 ==========
    print("\n=== Task 1: Træning påcifre 0-4 ===")
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_model(model, task1_loader, optimizer, criterion, device, epochs=5)

    acc_t1, class_acc_t1 = evaluate(model, test_loader, device, task1_classes)
    print(f"Accuracy på0-4: {acc_t1:.1f}%")
    print(f"Per klasse: {class_acc_t1}")
    results["task1_after_training"] = {
        "accuracy_0_4": acc_t1,
        "per_class": class_acc_t1,
    }

    # ========== TASK 2: Naiv træning på5-9 ==========
    print("\n=== Task 2: Naiv træning påcifre 5-9 (ingen replay) ===")
    naive_model = SimpleNet().to(device)
    naive_model.load_state_dict(model.state_dict())
    naive_opt = optim.Adam(naive_model.parameters(), lr=1e-3)

    train_model(naive_model, task2_loader, naive_opt, criterion, device,
                epochs=5)

    acc_old, _ = evaluate(naive_model, test_loader, device, task1_classes)
    acc_new, _ = evaluate(naive_model, test_loader, device, task2_classes)
    acc_all, class_acc_naive = evaluate(naive_model, test_loader, device)
    print(f"Accuracy på0-4 (gamle): {acc_old:.1f}%")
    print(f"Accuracy på5-9 (nye):   {acc_new:.1f}%")
    print(f"Accuracy total:           {acc_all:.1f}%")
    print(f"Per klasse: {class_acc_naive}")
    results["task2_naive"] = {
        "accuracy_0_4": acc_old,
        "accuracy_5_9": acc_new,
        "accuracy_total": acc_all,
        "per_class": class_acc_naive,
    }

    # ========== TASK 3: Replay + EWC ==========
    print("\n=== Task 3: Experience Replay + EWC ===")
    replay_model = SimpleNet().to(device)
    replay_model.load_state_dict(model.state_dict())
    replay_opt = optim.Adam(replay_model.parameters(), lr=1e-3)

    # Gem samples fra task 1 i memory buffer
    buffer_size = 500
    buffer = []
    for x, y in task1_loader:
        for i in range(len(x)):
            buffer.append((x[i], y[i].item()))
            if len(buffer) >= buffer_size:
                break
        if len(buffer) >= buffer_size:
            break
    print(f"Memory buffer: {len(buffer)} samples fra task 1")

    # Beregn Fisher og gem gamle parametre for EWC
    fisher = compute_fisher(replay_model, task1_loader, device)
    old_params = {n: p.clone().detach()
                  for n, p in replay_model.named_parameters()}

    train_with_replay_ewc(replay_model, task2_loader, buffer, replay_opt,
                          criterion, device, fisher, old_params,
                          ewc_lambda=400, epochs=5)

    acc_old_r, _ = evaluate(replay_model, test_loader, device, task1_classes)
    acc_new_r, _ = evaluate(replay_model, test_loader, device, task2_classes)
    acc_all_r, class_acc_replay = evaluate(replay_model, test_loader, device)
    print(f"Accuracy på0-4 (gamle): {acc_old_r:.1f}%")
    print(f"Accuracy på5-9 (nye):   {acc_new_r:.1f}%")
    print(f"Accuracy total:           {acc_all_r:.1f}%")
    print(f"Per klasse: {class_acc_replay}")
    results["task3_replay_ewc"] = {
        "accuracy_0_4": acc_old_r,
        "accuracy_5_9": acc_new_r,
        "accuracy_total": acc_all_r,
        "per_class": class_acc_replay,
    }

    # ========== Sammenligning ==========
    print("\n=== Sammenligning ===")
    print(f"{'Metode':<25} {'0-4':>8} {'5-9':>8} {'Total':>8}")
    print("-" * 51)
    print(f"{'Efter task 1':<25} {acc_t1:>7.1f}% {'--':>7} {'--':>7}")
    print(f"{'Naiv (forgetting)':<25} {acc_old:>7.1f}% {acc_new:>7.1f}%"
          f" {acc_all:>7.1f}%")
    print(f"{'Replay + EWC':<25} {acc_old_r:>7.1f}% {acc_new_r:>7.1f}%"
          f" {acc_all_r:>7.1f}%")

    # Gem resultater
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    # Konverter int keys til strings for json
    for key in results:
        if "per_class" in results[key]:
            results[key]["per_class"] = {
                str(k): v for k, v in results[key]["per_class"].items()
            }
    with open(out_dir / "continual_learning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultater gemt i {out_dir / 'continual_learning_results.json'}")


if __name__ == "__main__":
    main()
