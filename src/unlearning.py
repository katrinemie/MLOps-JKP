"""
Module 7 - Machine Unlearning via Gradient Ascent on MNIST.

Demonstrates selective forgetting of a specific class (digit '7') from
a trained model without full retraining — relevant in GDPR "right to be
forgotten" contexts.

Steps:
  1. Train a classifier on full MNIST (digits 0-9).
  2. Apply gradient ascent on class 7 samples to degrade performance on that class.
  3. Evaluate: class 7 accuracy should drop; all other classes should be retained.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 5
UNLEARN_EPOCHS = 3
UNLEARN_LR = 5e-4
FORGET_CLASS = 7
RESULTS_PATH = "results/unlearning_results.json"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def filter_class(dataset, label):
    indices = [i for i, (_, l) in enumerate(dataset) if l == label]
    return Subset(dataset, indices)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()


def evaluate_per_class(model, test_dataset, num_classes=10):
    model.eval()
    correct = {c: 0 for c in range(num_classes)}
    total = {c: 0 for c in range(num_classes)}
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for c in range(num_classes):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                total[c] += mask.sum().item()
    return {c: (correct[c] / total[c] if total[c] > 0 else 0.0) for c in range(num_classes)}


def gradient_ascent_epoch(model, forget_loader, optimizer, criterion):
    """Maximise loss on the forget class by negating the gradient."""
    model.train()
    for imgs, labels in forget_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        # Negate loss to ascend instead of descend
        (-loss).backward()
        optimizer.step()


def main():
    transform = get_transform()
    train_full = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
    test_full = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)

    # ------------------------------------------------------------------
    # Step 1: Train on full MNIST
    # ------------------------------------------------------------------
    print("Step 1: Training on full MNIST (digits 0-9)...")
    model = SimpleCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_full, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(PRETRAIN_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}/{PRETRAIN_EPOCHS} done")

    acc_before = evaluate_per_class(model, test_full)
    overall_before = sum(acc_before.values()) / len(acc_before)
    print(f"  Overall accuracy before unlearning: {overall_before:.1%}")
    print(f"  Class {FORGET_CLASS} accuracy before: {acc_before[FORGET_CLASS]:.1%}")

    # ------------------------------------------------------------------
    # Step 2: Gradient ascent on class 7
    # ------------------------------------------------------------------
    print(f"\nStep 2: Applying gradient ascent to forget class {FORGET_CLASS}...")
    forget_data = filter_class(train_full, FORGET_CLASS)
    forget_loader = DataLoader(forget_data, batch_size=BATCH_SIZE, shuffle=True)
    unlearn_optimizer = optim.SGD(model.parameters(), lr=UNLEARN_LR)

    for epoch in range(UNLEARN_EPOCHS):
        gradient_ascent_epoch(model, forget_loader, unlearn_optimizer, criterion)
        acc_mid = evaluate_per_class(model, test_full)
        print(f"  Epoch {epoch + 1}/{UNLEARN_EPOCHS} — class {FORGET_CLASS} acc: "
              f"{acc_mid[FORGET_CLASS]:.1%}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    print("\nStep 3: Evaluating after unlearning...")
    acc_after = evaluate_per_class(model, test_full)
    overall_after = sum(acc_after.values()) / len(acc_after)
    retained_classes = [c for c in range(10) if c != FORGET_CLASS]
    retained_acc_after = sum(acc_after[c] for c in retained_classes) / len(retained_classes)

    print(f"\n=== UNLEARNING RESULTS ===")
    print(f"Overall accuracy before: {overall_before:.1%}  →  after: {overall_after:.1%}")
    print(f"Class {FORGET_CLASS} accuracy before: {acc_before[FORGET_CLASS]:.1%}  "
          f"→  after: {acc_after[FORGET_CLASS]:.1%}  (forgotten)")
    print(f"Retained classes avg before: "
          f"{sum(acc_before[c] for c in retained_classes)/len(retained_classes):.1%}  "
          f"→  after: {retained_acc_after:.1%}")

    results = {
        "forget_class": FORGET_CLASS,
        "overall_accuracy_before": round(overall_before, 4),
        "overall_accuracy_after": round(overall_after, 4),
        "forget_class_accuracy_before": round(acc_before[FORGET_CLASS], 4),
        "forget_class_accuracy_after": round(acc_after[FORGET_CLASS], 4),
        "retained_classes_avg_before": round(
            sum(acc_before[c] for c in retained_classes) / len(retained_classes), 4),
        "retained_classes_avg_after": round(retained_acc_after, 4),
        "per_class_accuracy_before": {str(c): round(v, 4) for c, v in acc_before.items()},
        "per_class_accuracy_after": {str(c): round(v, 4) for c, v in acc_after.items()},
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
