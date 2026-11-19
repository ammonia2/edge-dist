#!/usr/bin/env bash
# End-to-end pipeline. Usage: ./run_all.sh [quick|full]   (default: quick, ~1h on a 4-core CPU)
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-quick}"
PY="${PYTHON:-python}"
mkdir -p logs

echo "### Q1: knowledge distillation (CIFAR-100)"
$PY scripts/train_teacher.py  --profile "$PROFILE"            2>&1 | tee logs/teacher.log
$PY scripts/train_students.py --profile "$PROFILE" --mode baseline 2>&1 | tee logs/students_baseline.log
$PY scripts/train_students.py --profile "$PROFILE" --mode kd       2>&1 | tee logs/students_kd.log

echo "### Q2: hardware-aware NAS (CIFAR-10)"
$PY scripts/nas_search.py     --profile "$PROFILE"            2>&1 | tee logs/nas_search.log
$PY scripts/nas_train_best.py --profile "$PROFILE"            2>&1 | tee logs/nas_train_best.log

echo "### Deployment: ONNX export, inference, benchmark"
$PY scripts/export_onnx.py                                    2>&1 | tee logs/export_onnx.log
$PY scripts/infer.py --model all --backend onnx --eval        2>&1 | tee logs/infer_onnx.log
$PY scripts/benchmark.py                                      2>&1 | tee logs/benchmark.log
$PY scripts/make_report.py                                    2>&1 | tee logs/report.log
