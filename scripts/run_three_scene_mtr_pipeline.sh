#!/usr/bin/env bash
set -euo pipefail

# Full three-scene GT-free MTR export and scalar analyses.  Run this script in
# the GPU runtime after the 4090 is visible; it intentionally does not train a
# controller or read GT during tracking, association, ledger construction, or
# MTR inference.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
REPLAY="${ROOT}/data/intermediate/gtfree_cpu/opv2v_3scene"
PRED="${REPLAY}/mtr_predictions_gtfree_3scene.jsonl"
ATTACHED="${ROOT}/data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene"
EVAL="${ROOT}/data/intermediate/gtfree_cpu/opv2v_3scene_mtr_3scene_eval"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${ROOT}/vendor/CMP-upstream/MTR:${ROOT}/vendor/CMP-upstream:${ROOT}"

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA device unavailable; switch to the GPU runtime before running the MTR export")
print(torch.cuda.get_device_name(0))
PY

"${PYTHON}" "${ROOT}/scripts/run_gtfree_mtr.py" \
  --scene 2021_08_18_19_48_05 \
  --scene 2021_08_20_21_10_24 \
  --scene 2021_08_22_09_08_29 \
  --min-frame 10 \
  --output "${PRED}"

"${PYTHON}" "${ROOT}/scripts/ingest_mtr_predictions.py" \
  "${REPLAY}" "${PRED}" --output-dir "${ATTACHED}"

"${PYTHON}" "${ROOT}/scripts/recompute_mtr_ledger_errors.py" \
  "${ATTACHED}/prediction_ledger.jsonl" --output-dir "${EVAL}"

# Offline metrics need the GT-free tracks/events beside the derived ledger;
# only this final evaluator reads raw labels.
cp "${REPLAY}/local_tracks.jsonl" "${EVAL}/local_tracks.jsonl"
cp "${REPLAY}/association_events.jsonl" "${EVAL}/association_events.jsonl"

"${PYTHON}" "${ROOT}/scripts/analyze_mtr_predictiveness.py" \
  "${EVAL}/prediction_ledger.jsonl" --output "${EVAL}/mtr_predictiveness.json"

"${PYTHON}" "${ROOT}/scripts/evaluate_gtfree.py" \
  "${EVAL}" --output "${EVAL}/offline_evaluation.json"

echo "complete: ${EVAL}"
