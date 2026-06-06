# Opgavestatus — MLOps-JKP

> Sidst opdateret: 2026-04-15

---

## Samlet overblik

| Modul | Emne | D-punkter | Status |
|-------|------|-----------|--------|
| 1 | Introduction to MLOps | 4/4 | ✅ Komplet |
| 2 | Continuous ML (CI/CD) | 3/3 | ⚠️ Mangler branch protection |
| 3 | Scalable Training | 6/6 | ✅ Komplet |
| 4 | Scalable Inference | 4/4 | ✅ Komplet |
| 5 | Deployment | 2/2 | ✅ Komplet |
| 6 | Monitoring | 3/4 | ⚠️ Mangler D6.4 dashboard |
| 7 | Post Deployment | 2/2 | ✅ Komplet |
| **Total** | | **24/25 (96%)** | **Over 75% grænsen** |

---

## Modul 1 — Introduction to MLOps

### D-punkter

| ID | Krav | Status | Placering |
|----|------|--------|-----------|
| D1.1 | Introduktion til MLOps og forskel fra andre paradigmer | ✅ | Rapport |
| D1.2 | Beskrivelse af valgt projekt | ✅ | Rapport — Cats vs Dogs, ResNet18 |
| D1.3 | Forventede udfordringer | ✅ | Rapport |
| D1.4 | Model card (draft → færdiggøres løbende) | ✅ | `src/model_card.py` + rapport appendix |

### Øvelser (9/9)

| # | Øvelse | Status | Fil |
|---|--------|--------|-----|
| 1 | GitHub repo med .gitignore | ✅ | `.gitignore` |
| 2 | Base deep-learning projekt | ✅ | `src/model.py`, `src/train.py` |
| 3 | Model fil + training/test script | ✅ | `src/model.py` (ResNet18), `src/train.py`, `src/evaluate.py` |
| 4 | requirements.txt | ✅ | `requirements.txt` |
| 5 | Dokumenter koden | ✅ | Docstrings i src/ filer |
| 6 | PEP8 compliance | ✅ | flake8 via pre-commit + CI |
| 7 | Version control for data/model | ✅ | `.dvc/config` → MinIO `s3://pkj-catdog/dvc` |
| 8 | Konfigurationsfiler | ✅ | `configs/config.yaml` |
| 9 | Load konfigurationer | ✅ | Bruges i `src/train.py` |

---

## Modul 2 — Continuous ML

### D-punkter

| ID | Krav | Status | Placering |
|----|------|--------|-----------|
| D2.1 | Overblik over CI/CD pipeline (flowchart + forklaring + lineage) | ✅ | Rapport + `Jenkinsfile` (9 stages) |
| D2.2 | Code coverage % med screenshot | ✅ | Rapport (tabel) — **mangler screenshot** |
| D2.3 | Experiment tracking dashboards/screenshots | ✅ | Rapport — **mangler MLflow screenshots** |

### Øvelser (12/13)

| # | Øvelse | Status | Fil |
|---|--------|--------|-----|
| 1 | Opret development branch | ✅ | `development` branch pushet |
| 2 | Pre-commit hooks | ✅ | `.pre-commit-config.yaml` (8 hooks) |
| 3 | Unit tests | ✅ | `tests/test_model.py` (2), `tests/test_data_loader.py` (6) — 8 total |
| 4 | CI/CD framework (Jenkins) | ✅ | `Jenkinsfile` — 9 stages |
| 5 | Trigger ved nye commits | ✅ | Jenkins webhook |
| 6 | Docker build + push til registry | ✅ | Stage 3 → `172.24.198.42:5000`, tagget med git hash |
| 7 | Automatisk model træning | ✅ | Stage 5 |
| 8 | Lineage via MLflow | ✅ | `src/train.py` → `172.24.198.42:5050` |
| 9 | Automatisk evaluering | ✅ | Stage 6 |
| 10 | Model registry (acc ≥ 80%) | ✅ | Stage 8 → MLflow `cats-vs-dogs-model` |
| 11 | Deploy + log deployment | ✅ | Stage 9 + 10 |
| 12 | Gem model card | ✅ | `src/model_card.py` |
| 13 | Branch protection + auto-merge | ❌ | **Ikke konfigureret på GitHub** |

---

## Modul 3 — Scalable Training

### D-punkter

| ID | Krav | Status | Resultat |
|----|------|--------|----------|
| D3.1 | Speedup-estimat (parallelisering) | ✅ | Gustafson's Law, 2.70x med 3 GPUs |
| D3.2 | Skalering for at halvere test loss (power-law) | ✅ | Rapport (compute/data/params) |
| D3.3 | Multi-GPU parallelisering | ✅ | 1.56x speedup med 2 GPUs DDP |
| D3.4 | Multi-node parallelisering | ✅ | Implementeret, ikke benchmarked |
| D3.5 | Memory optimization (AMP) | ✅ | 33% VRAM besparelse (860→574 MB) |
| D3.6 | ZeRO optimizer stages | ✅ | Stage 1/2/3 configs implementeret |

### Øvelser (6/6)

| # | Øvelse | Status | Fil |
|---|--------|--------|-----|
| 1 | DDP på multiple GPUs | ✅ | `src/train_ddp.py`, `src/train_ddp_benchmark.py` |
| 2 | AMP | ✅ | Integreret i `src/train.py` |
| 3 | Multi-node med torchrun/DeepSpeed | ✅ | `src/train_deepspeed.py`, `scripts/launch_multinode.sh` |
| 4 | ZeRO optimizer (stage 1-3) | ✅ | `configs/ds_config_zero{1,2,3}.json` |
| 5 | Optimering i pipeline | ✅ | AMP i Jenkinsfile train stage |
| 6 | Feature branches | ✅ | Brugt under udvikling |

### Resultater (`results/module3_results.json`)

| Config | Tid/epoch | VRAM | Val Acc |
|--------|-----------|------|---------|
| 1 GPU baseline | 32.6s | 860 MB | 96.1% |
| 1 GPU + AMP | 32.2s | 574 MB | 93.8% |
| 2 GPU DDP | 20.9s | 903 MB | 96.3% |
| 2 GPU DDP + AMP | 20.7s | 617 MB | 95.3% |

---

## Modul 4 — Scalable Inference

### D-punkter

| ID | Krav | Status | Resultat |
|----|------|--------|----------|
| D4.1 | Speedup fra komprimering | ✅ | INT8: 74.8% mindre, 2.4-6.3x hurtigere |
| D4.2 | Batch processing speedup | ✅ | Peak 350.2 FPS ved batch 16 |
| D4.3 | Pruning vs accuracy plot | ✅ | Accuracy cliff ved 50% |
| D4.4 | Fine-tuning af pruned model | ✅ | 99.5% recovery via distillation |

### Øvelser (6/6)

| # | Øvelse | Status | Fil |
|---|--------|--------|-----|
| 1 | Post-training kvantisering (FP32→INT8) | ✅ | `src/quantize_benchmark.py` |
| 2 | Benchmark inference + accuracy | ✅ | `results/quantization_results.json` |
| 3 | Batch inference | ✅ | `src/batch_benchmark.py` |
| 4 | Pruning (gradvis weight removal) | ✅ | `src/prune_finetune.py` |
| 5 | Fine-tuning af pruned model | ✅ | `src/prune_finetune.py` (knowledge distillation) |
| 6 | Optimering i pipeline | ✅ | Stage 7 (Quantize) i Jenkinsfile |

### Nøgleresultater

**Kvantisering:** FP32 42.71 MB → INT8 10.78 MB (74.8% reduktion), 6.3x speedup ved bs=32

**Batch:** Peak throughput 350.2 img/s ved batch 16, mætning derefter

**Pruning:** 0-40% pruning = minimal tab, 50% = cliff (14%), distillation genvinder 99.5%

---

## Modul 5 — Deployment

### D-punkter

| ID | Krav | Status | Placering |
|----|------|--------|-----------|
| D5.1 | Inference tid på telefon (FP32 vs UInt8) med screenshot | ✅ | Rapport — **mangler telefon-screenshot** |
| D5.2 | Endpoint testing + safeguarding | ✅ | Rapport (beskrevet konceptuelt) |

### Øvelser

| # | Øvelse | Status | Detaljer |
|---|--------|--------|----------|
| 1 | Kvantiser ResNet50 ONNX + deploy på Android | ✅ | Separat Android Studio projekt, Samsung Galaxy |

**Note:** Modul 5 var en specifik opgave (Android deploy) — koden er ikke i dette repo.

---

## Modul 6 — Monitoring

### D-punkter

| ID | Krav | Status | Placering |
|----|------|--------|-----------|
| D6.1 | Carbon footprint af træning | ✅ | CarbonTracker målt: 1.06g CO2, 0.0074 kWh, 5:18 min (10 epochs) |
| D6.2 | Årligt CO2-estimat | ✅ | `src/cost_estimator.py` → 0.23 kg/år, $28.46/år |
| D6.3 | Drift detection + mitigering | ✅ | `src/drift_detector.py` (KS-test + accuracy monitoring) |
| D6.4 | Monitoring dashboard screenshot | ❌ | **Prometheus/Grafana ikke sat op** |

### Øvelser (2/3)

| # | Øvelse | Status | Fil |
|---|--------|--------|-----|
| 1 | CarbonTracker under træning/inference | ✅ | Kørt på AI-Lab, 1.06g CO2 per run |
| 2 | Drift detection pipeline | ✅ | `src/drift_detector.py` → `results/drift_detection_report.json` |
| 3 | Monitoring framework (Prometheus+Grafana) | ❌ | Ikke implementeret |

### Mangler

1. **D6.1:** Kør træning med CarbonTracker på AI-Lab → gem faktisk output
2. **D6.4:** Opsæt Prometheus + Grafana, eller brug MLflow UI som monitoring dashboard (tag screenshot)

---

## Modul 7 — Post Deployment

### D-punkter

| ID | Krav | Status |
|----|------|--------|
| D7.1 | Sammenlign naiv træning vs continual learning (Replay + EWC) | ✅ | `src/continual_learning.py` |
| D7.2 | Evaluer unlearning (glem klasse "7", behold resten) | ✅ | `src/unlearning.py` |

### Øvelser (2/2)

| # | Øvelse | Status | Fil |
|---|--------|--------|-----|
| 1 | **Continual Learning** — Træn på 0-4, naiv 5-9, replay + EWC | ✅ | `src/continual_learning.py` |
| 2 | **Unlearning** — Gradient ascent for at glemme "7" + retain fine-tuning | ✅ | `src/unlearning.py` |

### Resultater

**Continual Learning:**

| Metode | Acc 0-4 | Acc 5-9 | Total |
|--------|---------|---------|-------|
| Efter task 1 | 99.2% | -- | -- |
| Naiv (forgetting) | 0.0% | 98.4% | 47.8% |
| Replay + EWC | 89.2% | 98.5% | 93.7% |

**Unlearning:** Klasse 7 accuracy: 96.9% → 0.0% (glemt). Remaining classes: 97.8% gennemsnit (minimal collateral damage, max -1.6%).

**Note:** Disse øvelser bruger MNIST, ikke Cats vs Dogs. Behøver ikke integreres i MLOps pipeline.

---

## Alle filer i repoet og deres modul

### src/

| Fil | Modul | Beskrivelse |
|-----|-------|-------------|
| `model.py` | 1 | ResNet18, transfer learning, save/load |
| `data_loader.py` | 1 | Dataset, augmentering, splits |
| `train.py` | 1+2+6 | Træning, AMP, MLflow, CarbonTracker |
| `evaluate.py` | 1+2 | Accuracy, precision, recall, F1 |
| `model_card.py` | 1+2 | Model card generation |
| `serve.py` | 2 | Flask REST API (/health, /predict, /info) |
| `train_ddp.py` | 3 | DDP træning |
| `train_ddp_benchmark.py` | 3 | Benchmark 1 vs 2 GPUs ± AMP |
| `train_deepspeed.py` | 3 | DeepSpeed ZeRO |
| `summarize_module3.py` | 3 | Resultatopsummering |
| `quantize_benchmark.py` | 4 | FP32→INT8 kvantisering |
| `batch_benchmark.py` | 4 | Batch throughput/latency |
| `prune_finetune.py` | 4 | Pruning + knowledge distillation |
| `generate_figures.py` | 3+4 | Genererer rapport-plots |
| `cost_estimator.py` | 6 | Årligt CO2 + omkostninger |
| `drift_detector.py` | 6 | Data drift (KS-test) + concept drift |

### CI/CD + Config

| Fil | Modul |
|-----|-------|
| `Jenkinsfile` | 2 — 9 stages |
| `.github/workflows/ci.yml` | 2 — lint + test |
| `.pre-commit-config.yaml` | 2 — 8 hooks |
| `Dockerfile` | 2 — trænings-image |
| `Dockerfile.serve` | 2 — serving-image |
| `configs/config.yaml` | 1 — hyperparametre |
| `configs/ds_config_zero{1,2,3}.json` | 3 — DeepSpeed configs |

### Tests

| Fil | Antal |
|-----|-------|
| `tests/test_model.py` | 2 tests |
| `tests/test_data_loader.py` | 6 tests |

### Results

| Fil | Modul |
|-----|-------|
| `results/module3_results.json` | 3 |
| `results/quantization_results.json` | 4 |
| `results/batch_benchmark_results.json` | 4 |
| `results/pruning_results.json` | 4 |
| `results/annual_cost_estimate.json` | 6 |
| `results/drift_detection_report.json` | 6 |

### Scripts (SLURM)

| Fil | Modul |
|-----|-------|
| `scripts/run_module3_all.sh` | 3 |
| `scripts/launch_multinode.sh` | 3 |
| `scripts/run_module4_all.sh` | 4 |
| `scripts/run_module4.sh` | 4 |
| `scripts/run_module4_v2.sh` | 4 |
| `scripts/run_quant_v3.sh` | 4 |
| `scripts/run_figures.sh` | 3+4 |

---

## Hvad mangler (prioriteret)

### Kritisk

| # | Opgave | Modul | Indsats |
|---|--------|-------|---------|
| 1 | Continual Learning med Experience Replay + EWC (MNIST) | 7 | 2-3 timer |
| 2 | Unlearning med Gradient Ascent (MNIST) | 7 | 2-3 timer |
| 3 | Kør træning med CarbonTracker på AI-Lab | 6 | 30 min |

### Anbefalet

| # | Opgave | Modul | Indsats |
|---|--------|-------|---------|
| 4 | Monitoring dashboard (Prometheus/Grafana eller MLflow screenshot) | 6 | 1-2 timer |
| 5 | Branch protection på GitHub | 2 | 10 min |

### Screenshots til rapport

| Screenshot | Kilde | Modul |
|------------|-------|-------|
| GitHub repo overview | github.com/katrinemie/MLOps-JKP | D1 |
| configs/config.yaml | Repo | D1 |
| Jenkins UI (stage view) | 172.24.198.42:8080 | D2.1 |
| pytest coverage output | Terminal | D2.2 |
| MLflow experiments | 172.24.198.42:5050 | D2.3 |
| MLflow run comparison | 172.24.198.42:5050 | D2.3 |
| Telefon demo (FP32 vs INT8) | Samsung Galaxy | D5.1 |
| MLflow run detaljer | 172.24.198.42:5050 | D6.4 |
| MLflow Model Registry | 172.24.198.42:5050 | D6.4 |
