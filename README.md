# Edge AI: Knowledge Distillation + Hardware-Aware NAS

* **Q1:** ResNet-50 teacher, distilled into MobileNetV2, SqueezeNet and EfficientNet-Lite0 on CIFAR-100. Uses soft targets with temperature, combined hard and soft loss, and feature distillation. Includes baselines without KD, ONNX export and ONNX Runtime CPU inference.
* **Q2:** Evolutionary, hardware-aware NAS on CIFAR-10. Candidates are rejected on measured CPU latency, ONNX size and peak memory. The best architecture is retrained from scratch, then exported to ONNX for inference.

## Installation
```bash
conda create -y -p ./.venv python=3.11
conda activate ./.venv
pip install -r requirements.txt
# CIFAR-10/100 (python version) into ./data
mkdir -p data && cd data
curl -LO https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz && tar xzf cifar-100-python.tar.gz
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  && tar xzf cifar-10-python.tar.gz
cd ..
```

## Run everything
```bash
./run_all.sh quick     # ~1 h on a 4-core laptop CPU (budgets used for the reported results)
./run_all.sh full      # conventional GPU-scale schedule
```
Budgets live in `edgekd/config.py`. The NAS deployment constraints are there too: 3 ms latency, 2 MB size, 25 MB peak memory.

## Step by step
| Step | Command | Output |
|---|---|---|
| 1. Teacher | `python scripts/train_teacher.py` | `checkpoints/teacher_resnet50.pt` |
| 2. Baseline students | `python scripts/train_students.py --mode baseline` | `checkpoints/*_baseline.pt` |
| 3. KD students | `python scripts/train_students.py --mode kd [--temperature 4 --alpha 0.7 --beta 1]` | `checkpoints/*_kd.pt` |
| 4. NAS search | `python scripts/nas_search.py [--max-latency-ms 3 --max-size-mb 2 --max-mem-mb 25]` | `results/nas_history.{csv,json}`, `results/nas_best_arch.json` |
| 5. Train best NAS model | `python scripts/nas_train_best.py` | `checkpoints/nas_best.pt` |
| 6. ONNX export | `python scripts/export_onnx.py` | `onnx_models/*.onnx` (checked against PyTorch) |
| 7. Inference | `python scripts/infer.py --model all --backend onnx --eval`<br>`python scripts/infer.py --model mobilenetv2_kd --image img.png` | predictions / accuracy |
| 8. Benchmark | `python scripts/benchmark.py` | `results/benchmark.csv`, `results/kd_vs_baseline.csv` |
| KD ablation pilot | `python scripts/kd_pilot.py` | Table 3b of the report (`logs/kd_pilot.log`) |
| 9. Report | `python scripts/make_report.py --profile colab` | `docs/technical_report.pdf`, `docs/figures/` |

The model names for `infer.py` are `{mobilenetv2,squeezenet,efficientnet_lite0}_{kd,baseline}` and `nas_best`. Use `--backend torch` to run with PyTorch instead of ONNX Runtime.

## Results (Colab T4, `colab` profile)
Full report: [`docs/technical_report.pdf`](docs/technical_report.pdf). Raw numbers: `results/benchmark.csv`, `results/kd_vs_baseline.csv`.

| model | params (M) | ONNX (MB) | test acc % | ORT CPU latency (ms, batch 1) |
|---|---|---|---|---|
| ResNet-50 teacher | 23.7 | - | 81.14 | - |
| MobileNetV2 baseline / KD | 2.35 | 8.9 | 54.44 / 30.69 | 1.15 / 1.68 |
| SqueezeNet 1.1 baseline / KD | 0.78 | 3.0 | 51.39 / 38.25 | 0.63 / 0.87 |
| EfficientNet-Lite0 baseline / KD | 3.50 | 13.3 | 56.67 / 48.80 | 1.47 / 2.04 |
| NAS best (CIFAR-10) | 0.14 | 0.53 | 88.18 | 0.53 |

With the configured feature-hint weight (beta = 1) distillation lowered accuracy in this short schedule. The report
(Section 5, Table 3b) traces this to the hint loss dominating the gradient and to the soft logits of the briefly
fine-tuned teacher; a controlled pilot (`scripts/kd_pilot.py`) shows logit-only KD beating the baseline. Use
`--beta 0.1` or `--beta 0` for future runs.

## Layout
```
edgekd/
  data.py            fast in-memory CIFAR loaders + tensor augmentation
  models/teacher.py  ResNet-50 teacher (ImageNet init, feature output)
  models/students.py MobileNetV2, SqueezeNet 1.1, EfficientNet-Lite0 (from scratch)
  kd/losses.py       temperature softmax, soft-target KL, combined loss, feature distiller
  kd/trainer.py      distillation training loop
  nas/search_space.py   search space, sampling, mutation, crossover
  nas/generator.py      config -> nn.Module
  nas/hardware.py       ONNX export, CPU latency / size / peak-memory profiler, constraint check
  nas/evolution.py      (mu+lambda) evolutionary search
  inference.py       unified Predictor (PyTorch / ONNX Runtime)
scripts/             train / search / export / infer / benchmark / report entry points
```

## Notes
The reported results come from a CPU-only machine with no CUDA GPU. To fit that, the quick profile fine-tunes an ImageNet-pretrained teacher at 64 px with stem through layer2 frozen. It trains students for 4 epochs on 25k images and uses short NAS proxy training. Accuracies are therefore far below converged values. Baseline and KD runs share an identical budget, seed and data order, so the two are directly comparable.

## Google Colab (T4 GPU)
`edge_kd_nas_colab.ipynb` is a single, self-contained notebook. It writes this package into `/content`, downloads CIFAR, runs every stage with the `colab` profile (about 35–45 min on a T4) and downloads `edge_kd_nas_full.zip` at the end (checkpoints, ONNX models, results, logs, report, code). Regenerate it after code changes with `python scripts/build_colab.py`.
Training uses CUDA with mixed precision when available. Latency and memory are always measured on the CPU, the deployment target.
