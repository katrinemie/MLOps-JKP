# MLOps-JKP

Cats vs Dogs image classification pipeline med MLOps best practices.

## Projekt

Binary image classification (kat/hund) med ResNet18 og PyTorch. Bygget som del af DAKI4 kurset (Drift af AI-systemer) på AAU.

**Gruppe 3:** Jonas, Katrine, Peter

## Setup

```bash
pip install -r requirements.txt
python kaggle_download.py
python src/train.py --config configs/config.yaml
```

## Struktur

```
├── configs/           # Hyperparametre (YAML)
├── src/               # Træning, evaluering, serving
├── tests/             # Unit tests
├── scripts/           # SLURM batch scripts
├── results/           # Benchmark resultater (JSON)
├── Jenkinsfile        # CI/CD pipeline
└── Dockerfile.serve   # Docker til API serving
```

## Data

Microsoft Cats vs Dogs datasæt fra Kaggle (~25.000 billeder). Versioneret med DVC mod MinIO.

```bash
dvc pull    # hent data fra MinIO
```

## CI/CD

Jenkins pipeline med 9 stages: Lint → Test → Build Docker → Fetch Data → Train → Evaluate → Quantize → Register Model → Deploy API.

MLflow experiment tracking på `http://172.24.198.42:5050`.
