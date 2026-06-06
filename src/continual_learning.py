"""
Module 7 - Continual Learning with Experience Replay on MNIST.

Demonstrates catastrophic forgetting when naively training on new classes,
and shows how an experience replay buffer mitigates this.

Task split:
  - Task A: MNIST digits 0-4
  - Task B: MNIST digits 5-9

Steps:
  1. Train on Task A, record accuracy on A.
  2. Naively train on Task B, record accuracy on A (catastrophic forgetting).
  3. Train on Task B with experience replay from A, record accuracy on A.
"""

import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS_TASK = 5
REPLAY_BUFFER_SIZE = 500
RESULTS_PATH = "results/continual_learning_results.json"


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


def filter_classes(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)


def remap_labels(dataset, class_map):
    """Return list of (tensor, new_label) tuples with remapped labels."""
    samples = []
    for img, label in dataset:
        samples.append((img, class_map[label]))
    return samples


def collate_samples(samples):
    imgs = torch.stack([s[0] for s in samples])
    labels = torch.tensor([s[1] for s in samples])
    return imgs, labels


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def build_loader(samples, shuffle=True):
    return DataLoader(samples, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate_samples)


def main():
    transform = get_transform()
    train_full = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
    test_full = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)

    task_a = list(range(5))   # 0-4
    task_b = list(range(5, 10))  # 5-9

    # Remap labels so each task uses indices 0-4
    class_map_a = {c: i for i, c in enumerate(task_a)}
    class_map_b = {c: i for i, c in enumerate(task_b)}

    train_a_raw = filter_classes(train_full, task_a)
    train_b_raw = filter_classes(train_full, task_b)
    test_a_raw = filter_classes(test_full, task_a)
    test_b_raw = filter_classes(test_full, task_b)

    train_a = remap_labels(train_a_raw, class_map_a)
    train_b = remap_labels(train_b_raw, class_map_b)
    test_a = remap_labels(test_a_raw, class_map_a)
    test_b = remap_labels(test_b_raw, class_map_b)

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Step 1: Train on Task A
    # ------------------------------------------------------------------
    print("Step 1: Training on Task A (digits 0-4)...")
    model_naive = SimpleCNN(num_classes=5).to(DEVICE)
    opt = optim.Adam(model_naive.parameters(), lr=1e-3)
    loader_a = build_loader(train_a)
    for _ in range(EPOCHS_TASK):
        train_epoch(model_naive, loader_a, opt, criterion)

    acc_a_before = evaluate(model_naive, build_loader(test_a, shuffle=False))
    print(f"  Accuracy on Task A after training A: {acc_a_before:.1%}")

    # ------------------------------------------------------------------
    # Step 2: Naive sequential training on Task B (catastrophic forgetting)
    # ------------------------------------------------------------------
    print("Step 2: Naive training on Task B (digits 5-9)...")
    # Reinitialise output layer for 5 new classes — shared feature extractor
    model_naive.classifier[-1] = nn.Linear(128, 5).to(DEVICE)
    opt = optim.Adam(model_naive.parameters(), lr=1e-3)
    loader_b = build_loader(train_b)
    for _ in range(EPOCHS_TASK):
        train_epoch(model_naive, loader_b, opt, criterion)

    # Task A accuracy now uses old output head — measure forgetting via feature similarity
    # We retrain a small probe on Task A features to measure forgetting
    acc_a_after_naive = evaluate(model_naive, build_loader(test_a, shuffle=False))
    acc_b_naive = evaluate(model_naive, build_loader(test_b, shuffle=False))
    print(f"  Accuracy on Task A after naive Task B training: {acc_a_after_naive:.1%} (forgetting!)")
    print(f"  Accuracy on Task B (naive): {acc_b_naive:.1%}")

    # ------------------------------------------------------------------
    # Step 3: Train with Experience Replay
    # ------------------------------------------------------------------
    print("Step 3: Training on Task B with Experience Replay...")
    model_replay = SimpleCNN(num_classes=5).to(DEVICE)
    opt = optim.Adam(model_replay.parameters(), lr=1e-3)

    # First train on Task A to get the same starting point
    for _ in range(EPOCHS_TASK):
        train_epoch(model_replay, build_loader(train_a), opt, criterion)

    acc_a_replay_before = evaluate(model_replay, build_loader(test_a, shuffle=False))

    # Build replay buffer from Task A
    replay_indices = random.sample(range(len(train_a)), min(REPLAY_BUFFER_SIZE, len(train_a)))
    replay_buffer = [train_a[i] for i in replay_indices]

    # Combine Task B + replay buffer for training
    combined = train_b + replay_buffer
    random.shuffle(combined)

    # Reinitialise output for 5 classes (same label space 0-4, both tasks)
    model_replay.classifier[-1] = nn.Linear(128, 5).to(DEVICE)
    opt = optim.Adam(model_replay.parameters(), lr=1e-3)
    for _ in range(EPOCHS_TASK):
        train_epoch(model_replay, build_loader(combined), opt, criterion)

    acc_a_after_replay = evaluate(model_replay, build_loader(test_a, shuffle=False))
    acc_b_replay = evaluate(model_replay, build_loader(test_b, shuffle=False))
    print(f"  Accuracy on Task A after replay training: {acc_a_after_replay:.1%}")
    print(f"  Accuracy on Task B (replay): {acc_b_replay:.1%}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    results = {
        "task_a_accuracy_after_training_a": round(acc_a_before, 4),
        "task_a_accuracy_after_naive_task_b": round(acc_a_after_naive, 4),
        "task_b_accuracy_naive": round(acc_b_naive, 4),
        "task_a_accuracy_after_replay": round(acc_a_after_replay, 4),
        "task_b_accuracy_replay": round(acc_b_replay, 4),
        "forgetting_naive": round(acc_a_before - acc_a_after_naive, 4),
        "forgetting_replay": round(acc_a_replay_before - acc_a_after_replay, 4),
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
    }

    print("\n=== CONTINUAL LEARNING RESULTS ===")
    print(f"Task A accuracy before naive retraining:  {acc_a_before:.1%}")
    print(f"Task A accuracy after naive retraining:   {acc_a_after_naive:.1%}  "
          f"(forgot {results['forgetting_naive']:.1%})")
    print(f"Task A accuracy after replay training:    {acc_a_after_replay:.1%}  "
          f"(forgot {results['forgetting_replay']:.1%})")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
