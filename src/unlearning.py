"""
Machine unlearning med gradient ascent paa MNIST.
Task 1: Traen classifier paa alle cifre 0-9
Task 2: Glem klasse 7 via gradient ascent
Task 3: Evaluer forgetting og retention
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
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


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_class = {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for c in y.unique():
                c = c.item()
                mask = y == c
                per_class.setdefault(c, [0, 0])
                per_class[c][0] += (pred[mask] == y[mask]).sum().item()
                per_class[c][1] += mask.sum().item()
            correct += (pred == y).sum().item()
            total += len(y)
    acc = 100 * correct / total
    class_acc = {c: 100 * v[0] / v[1] for c, v in sorted(per_class.items())}
    return acc, class_acc


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
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    forget_class = 7
    results = {}

    # ========== TASK 1: Traen paa alle cifre ==========
    print("\n=== Task 1: Traening paa MNIST 0-9 ===")
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    baseline_acc, baseline_class = evaluate(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.1f}%")
    print(f"Per klasse: {baseline_class}")
    print(f"Klasse {forget_class} accuracy: "
          f"{baseline_class[forget_class]:.1f}%")
    results["baseline"] = {
        "total_accuracy": baseline_acc,
        "per_class": {str(k): v for k, v in baseline_class.items()},
    }

    # ========== TASK 2: Gradient ascent for at glemme klasse 7 ==========
    print(f"\n=== Task 2: Unlearning klasse {forget_class} "
          f"(gradient ascent) ===")

    # Hent kun samples fra forget-klassen
    forget_indices = [i for i, (_, y) in enumerate(train_data)
                      if y == forget_class]
    forget_subset = Subset(train_data, forget_indices)
    forget_loader = DataLoader(forget_subset, batch_size=64, shuffle=True)

    # Gradient ascent: MAKSIMER loss paa forget-klassen
    unlearn_opt = optim.Adam(model.parameters(), lr=5e-4)
    unlearn_epochs = 3

    model.train()
    for ep in range(unlearn_epochs):
        epoch_loss = 0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            unlearn_opt.zero_grad()
            loss = -criterion(model(x), y)  # negativ loss = gradient ascent
            loss.backward()
            unlearn_opt.step()
            epoch_loss += -loss.item()
        print(f"  Epoch {ep + 1}/{unlearn_epochs}, "
              f"loss paa forget class: {epoch_loss / len(forget_loader):.4f}")

    # ========== TASK 3: Evaluer forgetting og retention ==========
    print(f"\n=== Task 3: Evaluering efter unlearning ===")
    after_acc, after_class = evaluate(model, test_loader, device)

    retain_classes = [c for c in range(10) if c != forget_class]
    retain_accs = [after_class[c] for c in retain_classes]
    retain_avg = sum(retain_accs) / len(retain_accs)

    print(f"Total accuracy: {after_acc:.1f}%")
    print(f"Klasse {forget_class} accuracy: "
          f"{after_class[forget_class]:.1f}% "
          f"(var {baseline_class[forget_class]:.1f}%)")
    print(f"Remaining classes avg accuracy: {retain_avg:.1f}%")
    print(f"Per klasse: {after_class}")

    results["after_unlearning"] = {
        "total_accuracy": after_acc,
        "per_class": {str(k): v for k, v in after_class.items()},
        "forget_class": forget_class,
        "forget_accuracy_before": baseline_class[forget_class],
        "forget_accuracy_after": after_class[forget_class],
        "retain_avg_accuracy": retain_avg,
    }

    # ========== Sammenligning ==========
    print(f"\n=== Sammenligning ===")
    print(f"{'Klasse':<10} {'Foer':>10} {'Efter':>10} {'Diff':>10}")
    print("-" * 42)
    for c in range(10):
        diff = after_class[c] - baseline_class[c]
        marker = " <-- FORGET" if c == forget_class else ""
        print(f"{c:<10} {baseline_class[c]:>9.1f}% {after_class[c]:>9.1f}%"
              f" {diff:>+9.1f}%{marker}")

    # Gem resultater
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "unlearning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultater gemt i {out_dir / 'unlearning_results.json'}")


if __name__ == "__main__":
    main()
