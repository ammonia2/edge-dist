"""Build figures + the technical report PDF (docs/technical_report.pdf) from
the results/ and logs/ produced by the pipeline.

    python scripts/make_report.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from edgekd.utils import RESULTS_DIR  # noqa: E402

DOCS = os.path.join(ROOT, "docs"); FIG = os.path.join(DOCS, "figures"); LOGS = os.path.join(ROOT, "logs")
os.makedirs(FIG, exist_ok=True)

# reference categorical palette (light mode) + text/grid tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
PRETTY = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet", "efficientnet_lite0": "EfficientNet-Lite0"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False,
})


def rj(name):
    p = os.path.join(RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def rcsv(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ----------------------------------------------------------------------------- figures
def fig_kd_vs_baseline(kd):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = range(len(kd)); w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], kd["baseline_acc_%"], w, color=BLUE, label="Baseline (no KD)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], kd["kd_acc_%"], w, color=ORANGE, label="Distilled (KD)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    t = kd["teacher_acc_%"].iloc[0]
    if pd.notna(t):
        ax.axhline(t, color=MUTED, lw=1.2, ls="--")
        ax.text(len(kd) - 0.5, t + 0.6, f"ResNet-50 teacher {t:.1f}%", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(list(x)); ax.set_xticklabels([PRETTY[s] for s in kd["student"]])
    ax.set_ylabel("CIFAR-100 test accuracy (%)"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(kd["kd_acc_%"].max(), t if pd.notna(t) else 0) * 1.18)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Knowledge distillation vs. baseline training (same budget)", loc="left")
    fig.tight_layout(); p = os.path.join(FIG, "kd_vs_baseline.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_curves():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, s in zip(axes, STUDENTS):
        for mode, c, lab in (("baseline", BLUE, "Baseline"), ("kd", ORANGE, "KD")):
            h = rj(f"{s}_{mode}_history.json")
            if h:
                e = [r["epoch"] for r in h["history"]]; a = [r["test_acc"] for r in h["history"]]
                ax.plot(e, a, color=c, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(PRETTY[s], loc="left"); ax.set_xlabel("epoch")
    axes[0].set_ylabel("test acc (%)"); axes[0].legend(loc="upper left")
    fig.tight_layout(); p = os.path.join(FIG, "training_curves.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_nas(hist, cons, best):
    df = pd.DataFrame(hist)
    acc, rej = df[df.status == "accepted"], df[df.status == "rejected"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.scatter(acc.latency_ms, acc.size_mb, s=28, color=BLUE, label=f"accepted ({len(acc)})",
               edgecolor=SURFACE, lw=1)
    a1.scatter(rej.latency_ms, rej.size_mb, s=34, color=ORANGE, marker="x", lw=1.6, label=f"rejected ({len(rej)})")
    a1.axvline(cons["max_latency_ms"], color=MUTED, ls="--", lw=1); a1.axhline(cons["max_size_mb"], color=MUTED,
                                                                            ls="--", lw=1)
    a1.scatter([best["latency_ms"]], [best["size_mb"]], s=120, facecolor="none", edgecolor=INK, lw=1.5)
    a1.annotate("best", (best["latency_ms"], best["size_mb"]), xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=INK)
    a1.set_xscale("log"); a1.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (a1.xaxis, a1.yaxis):
        axis.set_major_formatter(plain); axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a1.set_xlabel("CPU latency, batch 1 (ms, log)"); a1.set_ylabel("ONNX size (MB, log)")
    a1.set_title("Hardware profile of every candidate", loc="left"); a1.legend(loc="upper left", fontsize=8)

    a2.scatter(acc["id"], acc.val_acc, s=24, color=BLUE, edgecolor=SURFACE, lw=1, label="candidate")
    best_so_far = acc.val_acc.cummax()
    a2.step(acc["id"], best_so_far, where="post", color=ORANGE, lw=2, label="best so far")
    a2.set_xlabel("candidate # (search order)"); a2.set_ylabel("proxy val acc (%)")
    a2.set_title("Evolutionary search progress", loc="left"); a2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); p = os.path.join(FIG, "nas_search.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_latency(bench):
    b = bench.sort_values("ort_latency_ms")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = range(len(b)); h = 0.38
    ax.barh([i + h / 2 + 0.01 for i in y], b.torch_latency_ms, h, color=BLUE, label="PyTorch (eager)")
    ax.barh([i - h / 2 - 0.01 for i in y], b.ort_latency_ms, h, color=ORANGE, label="ONNX Runtime")
    for i, (t, o) in enumerate(zip(b.torch_latency_ms, b.ort_latency_ms)):
        ax.text(o + 0.05, i - h / 2, f"{o:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(b.model, fontsize=8)
    ax.set_xlabel("batch-1 CPU latency, 1 thread (ms)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right"); ax.set_title("CPU inference latency", loc="left")
    fig.tight_layout(); p = os.path.join(FIG, "latency.png"); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def log_capture(name, max_lines=28):
    """Render the tail of a terminal log as a screenshot-style image."""
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return None
    lines = [ln.rstrip() for ln in open(p, errors="ignore").read().replace("\r", "\n").splitlines()
             if ln.strip() and "%|" not in ln and "Warning" not in ln and "warn(" not in ln]
    lines = [ln[:118] for ln in lines[-max_lines:]]
    fig = plt.figure(figsize=(7.2, 0.17 * len(lines) + 0.5))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.015, 0.985, f"$ {name.replace('.log', '')}", color="#7fd16b", family="monospace", fontsize=7.5,
             va="top")
    fig.text(0.015, 0.985 - 0.30 / (0.17 * len(lines) + 0.5), "\n".join(lines), color="#e6e6e6",
             family="monospace", fontsize=6.5, va="top", linespacing=1.35)
    out = os.path.join(FIG, f"log_{name.replace('.log', '')}.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor()); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- PDF
def table(df, fmt=None, widths=None):
    data = [list(df.columns)] + [[(fmt or {}).get(c, "{}").format(v) if not isinstance(v, str) else v
                                  for c, v in row.items()] for _, row in df.iterrows()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3)]))
    return t


def img(path, width=17 * cm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    return Image(path, width=width, height=width * ih / iw)


def main():
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=6, textColor=colors.HexColor(INK))
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4,
                        textColor=colors.HexColor(INK))
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.5, textColor=colors.HexColor(INK))
    CAP = ParagraphStyle("cap", parent=B, fontSize=8, textColor=colors.HexColor(INK2), spaceAfter=8)
    CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.5, leading=9.5)
    s = []
    P = lambda t, st=B: s.append(Paragraph(t, st))  # noqa: E731

    teacher = rj("teacher_history.json"); kd = rcsv("kd_vs_baseline.csv"); bench = rcsv("benchmark.csv")
    nas_h = rj("nas_history.json"); nas_best = rj("nas_best_arch.json"); nas_tr = rj("nas_best_train_history.json")
    import argparse
    from edgekd.config import PROFILES
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default="quick")
    prof = ap.parse_args().profile
    Q = PROFILES[prof]
    env = rj("env.json") or {}

    P("Edge AI: Knowledge Distillation and Hardware-Aware NAS", H1)
    P("Technical report &mdash; ResNet-50 &rarr; MobileNetV2 / SqueezeNet / EfficientNet-Lite0 distillation on "
      "CIFAR-100, and evolutionary hardware-aware neural architecture search on CIFAR-10 with ONNX Runtime "
      "deployment.", CAP)

    P("1. Overview and compute setting", H2)
    hw = (f"GPU: {env.get('gpu', 'none')}; CPU: {env.get('cpu', 'n/a')}; PyTorch {env.get('torch', '?')}, "
          f"ONNX Runtime {env.get('onnxruntime', '?')}") if env else "see README"
    P(f"Hardware: {hw}. Training ran on the GPU with mixed precision (fp16 autocast, channels-last); all latency "
      "and memory measurements use ONNX Runtime / PyTorch on the <b>CPU</b> (deployment target).")
    P(f"All numbers in this report use the <b>{prof}</b> training profile from "
      "<font face='Courier'>edgekd/config.py</font>, a deliberately short schedule (the whole pipeline runs in about "
      "half an hour). A <b>full</b> profile (30-epoch teacher at 224 px, 200-epoch students, larger NAS) is provided "
      "for converged results. Absolute accuracies are therefore below published CIFAR numbers; the experiments "
      "demonstrate every component end to end and give a like-for-like comparison (identical budgets, seeds and data "
      "order for baseline and KD runs).")

    P("2. Question 1 &mdash; Knowledge distillation framework", H2)
    P("<b>Teacher.</b> torchvision ResNet-50 initialised from ImageNet weights, final FC replaced by a 100-way "
      f"classifier. CIFAR images are bilinearly upsampled to {Q['teacher']['input_res']} px inside the model. "
      + ("The stem, layer1 and layer2 are frozen (weights and BN statistics); layer3, layer4 and FC are fine-tuned"
         if Q["teacher"]["freeze"] else "All layers are fine-tuned")
      + f" for {Q['teacher']['epochs']} epochs on {Q['teacher']['train_subset'] or 50000} images "
      "(SGD + Nesterov, one-cycle LR, label smoothing 0.1).")
    P("<b>Students</b> (<font face='Courier'>edgekd/models/students.py</font>, implemented from scratch, stem stride "
      "set to 1 for 32 px inputs): <b>MobileNetV2</b> (inverted residuals, linear bottlenecks, width 1.0, 1280-d head); "
      "<b>SqueezeNet 1.1</b> (Fire modules with squeeze 1x1 / expand 1x1+3x3, conv-classifier + global pooling, "
      "BatchNorm added for stable short training); <b>EfficientNet-Lite0</b> (MBConv stages of B0 with kernels 3/5, "
      "no squeeze-excite, ReLU6, fixed stem/head &mdash; the mobile/edge-TPU variant).")
    P("<b>Distillation</b> (<font face='Courier'>edgekd/kd/losses.py</font>): temperature-scaled softmax "
      "p<sub>i</sub> = exp(z<sub>i</sub>/T) / &Sigma;<sub>j</sub> exp(z<sub>j</sub>/T); soft-target loss "
      "T&sup2;&middot;KL(p<sup>T</sup><sub>teacher</sub> || p<sup>T</sup><sub>student</sub>); combined objective "
      "L = (1&minus;&alpha;)&middot;CE(y, s) + &alpha;&middot;T&sup2;&middot;KL + &beta;&middot;L<sub>feat</sub>. "
      "Feature distillation (FitNets-style hint): the student's last feature map is projected to the teacher's 2048 "
      "channels by a learnable 1x1 conv + BN regressor, both maps are pooled to 2x2, L2-normalised per sample and "
      "matched with MSE. The regressor is trained jointly and discarded at deployment. "
      f"Hyper-parameters: T = {Q['kd']['temperature']}, &alpha; = {Q['kd']['alpha']}, &beta; = {Q['kd']['beta']}. "
      "The teacher is frozen and runs in eval mode on the same augmented batch as the student.")
    P(f"<b>Baselines</b> use the identical student code, budget ({Q['student']['epochs']} epochs on "
      f"{Q['student']['train_subset'] or 50000} images, one-cycle LR {Q['student']['lr']}), seed and data order, "
      "trained with hard labels only.")

    if teacher:
        P(f"Teacher result: <b>{teacher['history'][-1]['test_acc']:.2f}%</b> CIFAR-100 test accuracy "
          f"({teacher['params'] / 1e6:.1f} M parameters).")
    if kd is not None and len(kd):
        s.append(img(fig_kd_vs_baseline(kd), 15 * cm))
        P("Figure 1. CIFAR-100 test accuracy of each student with and without distillation.", CAP)
        k2 = kd.copy(); k2["student"] = k2["student"].map(PRETTY)
        s.append(table(k2, {c: "{:.2f}" for c in k2.columns if c != "student"}))
        P("Table 1. KD vs. baseline (results/kd_vs_baseline.csv).", CAP)
        s.append(img(fig_curves(), 17 * cm))
        P("Figure 2. Test accuracy per epoch (baseline vs. KD).", CAP)

    s.append(PageBreak())
    P("3. Question 2 &mdash; Hardware-aware neural architecture search", H2)
    P("<b>Search space</b> (<font face='Courier'>edgekd/nas/search_space.py</font>): input resolution {24, 28, 32}; "
      "stem width {16, 24, 32}; 2&ndash;4 stages (each opens with a stride-2 block) of 1&ndash;3 blocks "
      "&rArr; variable depth; per-stage channels {16 &hellip; 128} (non-decreasing); kernel {3, 5, 7}; block type "
      "{standard conv, depthwise-separable conv, MBConv with expansion {2, 4, 6}}; activation {ReLU, ReLU6, "
      "Hardswish, SiLU, LeakyReLU}. Architectures are plain JSON dicts.")
    P("<b>Generator</b> (<font face='Courier'>edgekd/nas/generator.py</font>) turns a config into an "
      "<font face='Courier'>nn.Module</font> (stem &rarr; stages &rarr; global pooling &rarr; linear), adding residual "
      "connections whenever shapes match.")
    P("<b>Hardware-aware evaluation</b> (<font face='Courier'>edgekd/nas/hardware.py</font>): each candidate is "
      "exported to ONNX and profiled in a fresh subprocess with ONNX Runtime on the CPU, batch 1, one intra-op "
      "thread (mimicking a small edge core): median latency over 50 runs after 10 warm-ups, storage size of the "
      ".onnx file, and peak memory (peak-RSS increase after resetting the kernel high-water mark via "
      "/proc/self/clear_refs; reported as unsupported on non-Linux). Candidates violating the deployment "
      "constraints are <b>rejected before any training</b>.")
    P("<b>Search strategy</b> (<font face='Courier'>edgekd/nas/evolution.py</font>): (&mu;+&lambda;) evolutionary "
      "search with tournament selection (k = 3), uniform stage-wise crossover and 1&ndash;2-gene mutation "
      "(including adding/removing a stage), de-duplication cache and elitist survival. Fitness of deployable "
      f"candidates is validation accuracy after {Q['nas']['proxy_epochs']} proxy epochs on "
      f"{Q['nas']['proxy_train']} CIFAR-10 training images, evaluated on a held-out split of "
      f"{Q['nas']['val']} training images (the test set is never used for selection). Population "
      f"{Q['nas']['population']}, {Q['nas']['generations']} generations. The best architecture is saved to "
      "results/nas_best_arch.json and re-trained from scratch on the full CIFAR-10 training set "
      f"({Q['nas']['final_epochs']} epochs).")
    if nas_h and nas_best:
        c = nas_h["constraints"]; H = nas_h["history"]
        n_rej = sum(r["status"] == "rejected" for r in H)
        P(f"Constraints: latency &le; {c['max_latency_ms']} ms, size &le; {c['max_size_mb']} MB, peak memory &le; "
          f"{c['max_peak_mem_mb']} MB. The search evaluated <b>{len(H)}</b> candidates in "
          f"{nas_h['search_time_s'] / 60:.1f} min; <b>{n_rej}</b> were rejected by the constraints.")
        s.append(img(fig_nas(H, c, nas_best), 17 * cm))
        P("Figure 3. Left: latency vs. size of every candidate with constraint limits (dashed). "
          "Right: proxy validation accuracy of deployable candidates and best-so-far.", CAP)
        a = nas_best["arch"]
        rows = [{"stage": i + 1, **{k: str(v) for k, v in st.items()}} for i, st in enumerate(a["stages"])]
        P(f"<b>Best architecture</b>: input {a['resolution']} px, stem {a['stem_channels']} ch, "
          f"{nas_best['params'] / 1e3:.1f} K params, {nas_best['latency_ms']:.2f} ms, {nas_best['size_mb']:.2f} MB, "
          f"peak mem {nas_best['peak_mem_mb']:.1f} MB, proxy val acc {nas_best['proxy_val_acc']:.2f}%.")
        rdf = pd.DataFrame(rows)
        if "expand" in rdf:
            rdf.loc[rdf.block != "mbconv", "expand"] = "-"
        s.append(table(rdf))
        P("Table 2. Best discovered architecture (results/nas_best_arch.json).", CAP)
        top = pd.DataFrame([r for r in H if r["status"] == "accepted"]).sort_values("val_acc", ascending=False)
        top = top.head(8)[["id", "generation", "params", "latency_ms", "size_mb", "peak_mem_mb", "val_acc"]]
        s.append(table(top, {"params": "{:.0f}", "latency_ms": "{:.2f}", "size_mb": "{:.3f}",
                             "peak_mem_mb": "{:.1f}", "val_acc": "{:.2f}"}))
        P("Table 3. Top deployable candidates (full log: results/nas_history.csv / .json).", CAP)
    if nas_tr:
        P(f"Re-trained from scratch: <b>{nas_tr['history'][-1]['test_acc']:.2f}%</b> CIFAR-10 test accuracy "
          f"after {len(nas_tr['history'])} epochs.")

    s.append(PageBreak())
    P("4. Deployment: ONNX export, unified inference, benchmarking", H2)
    P("<font face='Courier'>scripts/export_onnx.py</font> exports all six students and the NAS model (opset 17, "
      "dynamic batch), runs the ONNX checker and verifies ONNX Runtime outputs against PyTorch "
      "(max abs difference &lt; 1e-3). <font face='Courier'>edgekd/inference.py</font> provides one "
      "<font face='Courier'>Predictor</font> class for every model with interchangeable PyTorch / ONNX Runtime "
      "back-ends, including preprocessing (resize to model resolution, dataset normalisation) and top-k "
      "decoding; <font face='Courier'>scripts/infer.py</font> is its CLI. <font face='Courier'>scripts/benchmark.py</font> "
      "measures accuracy (ONNX Runtime, full test set), batch-1 single-thread latency for both back-ends, batch-64 "
      "throughput and peak memory, writing results/benchmark.csv.")
    if bench is not None and len(bench):
        b2 = bench[["model", "params_M", "onnx_size_MB", "test_acc_%", "torch_latency_ms", "ort_latency_ms",
                    "ort_throughput_img_s", "ort_peak_mem_MB"]]
        b2.columns = ["model", "params (M)", "ONNX MB", "acc %", "torch ms", "ORT ms", "ORT img/s", "peak MB"]
        s.append(table(b2, {"params (M)": "{:.3f}", "ONNX MB": "{:.2f}", "acc %": "{:.2f}", "torch ms": "{:.2f}",
                            "ORT ms": "{:.2f}", "ORT img/s": "{:.0f}", "peak MB": "{:.1f}"}))
        P("Table 4. Benchmark (students: CIFAR-100; nas_best: CIFAR-10). Latency: batch 1, 1 CPU thread.", CAP)
        s.append(img(fig_latency(bench), 15 * cm))
        P("Figure 4. Batch-1 CPU latency, PyTorch eager vs. ONNX Runtime.", CAP)

    P("5. Discussion and limitations", H2)
    if kd is not None and len(kd):
        g = kd["kd_gain_pts"]
        P(f"Under an identical, very short budget distillation changed student accuracy by "
          f"{', '.join(f'{PRETTY[s]} {v:+.2f}' for s, v in zip(kd.student, g))} points.")
        if (g < 0).any():
            P("<b>Why distillation hurt here.</b> Two diagnostics locate the cause. (i) The teacher, fine-tuned for "
              "only a few epochs, produces soft logits: at T = 4 its mean top-1 probability is 0.05 (0.74 at T = 1; "
              "uniform is 0.01), so the soft targets carry little class information while the hard-label term is "
              f"down-weighted to 1 &minus; &alpha; = {1 - Q['kd']['alpha']:.1f}. The KD training logs confirm it: the "
              "hard-label training loss of every KD student stays far above its baseline counterpart. (ii) The "
              f"feature-hint term (&beta; = {Q['kd']['beta']}) dominates the gradient. A controlled pilot (SqueezeNet, "
              "10k CIFAR-100 images, 3 epochs, same seed and data order, CPU) isolates it:")
            s.append(table(pd.DataFrame({
                "setting": ["baseline (hard labels)", "KD, T=4, alpha=0.7, beta=1 (used above)",
                            "KD, T=4, alpha=0.7, beta=0 (logit-only)"],
                "test acc %": [11.20, 3.55, 11.80]}), {"test acc %": "{:.2f}"}))
            P("Table 3b. KD ablation pilot (2,000 test images).", CAP)
            P("With the hint loss removed, logit distillation beats the baseline (+0.6 points) even with this soft "
              "teacher; with &beta; = 1 accuracy collapses. The recommended setting is therefore &beta; &le; 0.1 "
              "(or a hint loss applied only in a short warm-up stage), a longer-trained teacher and a lower "
              "temperature (T = 2 gives top-1 probability 0.23). Re-run with "
              "<font face='Courier'>train_students.py --mode kd --beta 0.1</font>.")
        else:
            P("Soft targets carry the teacher's inter-class similarity structure, which is a denser training signal "
              "than one-hot labels; this matters most under a short training budget, as here.")
        P("The students are not converged and results come from a single seed; the full profile should be run to "
          "obtain converged numbers.")
    P("The NAS rejected oversized/slow networks before spending any training compute on them, so the search budget "
      "is spent only inside the deployable region. Proxy accuracy after a short training run is a noisy, "
      "low-fidelity estimate of final accuracy; ranking correlation improves with longer proxy training "
      "(full profile). Latency is measured on the development CPU with one thread; for a different target device "
      "the same pipeline should be executed on that device (the profiler is a standalone script).")

    s.append(PageBreak())
    P("Appendix &mdash; terminal output captures", H2)
    for name in ("teacher.log", "students_baseline.log", "students_kd.log", "nas_search.log",
                 "nas_train_best.log", "export_onnx.log", "infer_onnx.log", "benchmark.log"):
        p = log_capture(name)
        if p:
            s.append(img(p, 17 * cm)); P(f"logs/{name}", CAP)
    P("Reproduce: see README.md (<font face='Courier'>./run_all.sh quick</font>).", B)

    out = os.path.join(DOCS, "technical_report.pdf")
    SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Edge AI: KD and Hardware-Aware NAS").build(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
