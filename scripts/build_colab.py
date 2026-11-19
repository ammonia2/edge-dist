"""Generate the single self-contained Colab notebook (edge_kd_nas_colab.ipynb)
from the repository sources, so the notebook and the repo never diverge.

    python scripts/build_colab.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = "/content/edge-dist"
FILES = [
    "edgekd/__init__.py", "edgekd/config.py", "edgekd/data.py", "edgekd/utils.py", "edgekd/inference.py",
    "edgekd/models/__init__.py", "edgekd/models/teacher.py", "edgekd/models/students.py",
    "edgekd/kd/__init__.py", "edgekd/kd/losses.py", "edgekd/kd/trainer.py",
    "edgekd/nas/__init__.py", "edgekd/nas/search_space.py", "edgekd/nas/generator.py",
    "edgekd/nas/hardware.py", "edgekd/nas/evolution.py",
    "scripts/train_teacher.py", "scripts/train_students.py", "scripts/nas_search.py", "scripts/nas_train_best.py",
    "scripts/export_onnx.py", "scripts/infer.py", "scripts/benchmark.py", "scripts/make_report.py",
    "requirements.txt", "README.md", "run_all.sh",
]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def sh(text):
    """A bash cell that FAILS (and halts Run all) if any command in it fails, even behind `| tee`."""
    body = "\n".join(l[1:] if l.startswith("!") else l for l in text.strip("\n").splitlines())
    code(f"%%bash\nset -eo pipefail\ncd {DST}\n{body}")


def code(text, collapsed=False):
    meta = {"cellView": "form"} if collapsed else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Edge AI: Knowledge Distillation + Hardware-Aware NAS (Colab, T4)

One notebook for the whole assignment:

* **Q1:** ResNet-50 teacher, distilled into MobileNetV2, SqueezeNet and EfficientNet-Lite0 on CIFAR-100. Uses soft targets with temperature, combined hard and soft loss, and feature distillation, with baselines trained without KD.
* **Q2:** Evolutionary, hardware-aware NAS on CIFAR-10. Candidates are rejected on measured CPU latency, ONNX size and peak memory, and the best one is retrained from scratch.
* **Deployment:** ONNX export, ONNX Runtime CPU inference, a benchmark CSV and the PDF technical report.

**How to run:** select *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
Total time is about **35–45 min** on a T4 (an estimate; not yet timed on a real T4). It uses the `colab` profile in `edgekd/config.py`. To make it shorter, lower the epoch counts in that cell.

| stage | ≈ time |
|---|---|
| teacher (ImageNet ResNet-50 fine-tuned, 3 ep @ 96 px) | 3–5 min |
| 3 baseline students (10 ep each) | 6–9 min |
| 3 KD students (10 ep each) | 10–14 min |
| NAS search (8 × 5 generations) + retrain best (15 ep) | 8–12 min |
| ONNX export, inference, benchmark, report | 3 min |

When it finishes, section 11 downloads **`edge_kd_nas_full.zip`**. It holds the complete project: every trained checkpoint (`.pt`), the ONNX models, results CSV/JSON, logs, the PDF report, figures and the source code.

""")

code("""
# 1. Check GPU + install the few packages Colab lacks
!nvidia-smi -L
!pip install -q onnx onnxruntime onnxscript reportlab
import torch; assert torch.cuda.is_available(), "Select a GPU runtime (Runtime > Change runtime type > T4 GPU)"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
""")

md("## 2. Source code\nEach cell below writes one file of the `edgekd` package / scripts (identical to the GitHub repo).")
code(f"!mkdir -p {DST}/edgekd/models {DST}/edgekd/kd {DST}/edgekd/nas {DST}/scripts {DST}/logs {DST}/data")
for f in FILES:
    src = open(os.path.join(ROOT, f)).read()
    code(f"%%writefile {DST}/{f}\n{src}")

md("## 3. Data + environment info")
code(f"""
%cd {DST}
%env PYTHONUNBUFFERED=1
%env PYTHONWARNINGS=ignore
from edgekd.data import ensure_data
for _d in ("cifar100", "cifar10"):
    ensure_data(_d)  # download + md5-verify + extract; raises (stops Run all) on failure
!ls data
import json, os, platform, subprocess, torch, onnxruntime
os.makedirs("results", exist_ok=True)
cpu = subprocess.run("lscpu | grep 'Model name' | cut -d: -f2", shell=True, capture_output=True, text=True).stdout.strip()
env = {{"gpu": torch.cuda.get_device_name(0), "cpu": f"{{cpu}} ({{os.cpu_count()}} vCPU)", "torch": torch.__version__,
       "onnxruntime": onnxruntime.__version__, "python": platform.python_version(), "profile": "colab"}}
json.dump(env, open("results/env.json", "w"), indent=2); print(env)
""")

md("## 4. Q1: Teacher (ResNet-50, CIFAR-100)")
sh("!python scripts/train_teacher.py --profile colab 2>&1 | tee logs/teacher.log")
md("## 5. Q1: Baseline students (no distillation)")
sh("!python scripts/train_students.py --profile colab --mode baseline 2>&1 | tee logs/students_baseline.log")
md("## 6. Q1: Distilled students (soft targets + temperature + hard labels + feature distillation)")
sh("!python scripts/train_students.py --profile colab --mode kd 2>&1 | tee logs/students_kd.log")
md("""
## 7. Q2: Hardware-aware evolutionary NAS (CIFAR-10)
Latency and peak memory are measured on the Colab **CPU**: ONNX Runtime, batch 1, one thread. Candidates over the limits are rejected before any training.
""")
sh("!python scripts/nas_search.py --profile colab --max-latency-ms 4 --max-size-mb 2 --max-mem-mb 30 "
     "2>&1 | tee logs/nas_search.log")
md("## 8. Q2: Train the best discovered architecture from scratch")
sh("!python scripts/nas_train_best.py --profile colab 2>&1 | tee logs/nas_train_best.log")
md("## 9. ONNX export + ONNX Runtime CPU inference")
sh("""
!python scripts/export_onnx.py 2>&1 | tee logs/export_onnx.log
!python scripts/infer.py --model all --backend onnx --eval 2>&1 | tee logs/infer_onnx.log
!python scripts/infer.py --model mobilenetv2_kd --backend onnx --samples 5 2>&1 | tee -a logs/infer_onnx.log
!python scripts/infer.py --model nas_best --backend onnx --samples 5 2>&1 | tee -a logs/infer_onnx.log
""")
md("## 10. Benchmark (CPU) + technical report")
sh("""
!python scripts/benchmark.py 2>&1 | tee logs/benchmark.log
!python scripts/make_report.py --profile colab 2>&1 | tee logs/report.log
""")
code("""
from IPython.display import Image, display
for f in ["kd_vs_baseline", "training_curves", "nas_search", "latency"]:
    display(Image(f"docs/figures/{f}.png", width=800))
""")
md("""
## 11. Download everything
`edge_kd_nas_full.zip` is the complete project, ready to push to GitHub:
* `checkpoints/`: teacher, 3 baseline, 3 KD and NAS `.pt` files
* `onnx_models/`: 7 `.onnx` models
* `results/`: CSV/JSON results, including the NAS history and best architecture
* `logs/`, `docs/` (PDF report and figures), plus the source code, README and requirements

CIFAR data is excluded. Allow the download if the browser asks.
""")
code("""
import os
from google.colab import files
!rm -f edge_kd_nas_full.zip && zip -qr edge_kd_nas_full.zip . -x "data/*" "*__pycache__*" "*.zip"
!unzip -l edge_kd_nas_full.zip | grep -E "\\.(pt|onnx|csv|json|pdf)$"
!ls -lh edge_kd_nas_full.zip
files.download("edge_kd_nas_full.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "edge_kd_nas_colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, f"({len(cells)} cells)")
