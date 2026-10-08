#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${ROBERTA_PYTHON:-python3}"
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.backends.mps.is_available() and not torch.cuda.is_available():
    raise SystemExit('GPU unavailable to this process. Run from your normal Terminal with a PyTorch environment that can access Metal/CUDA. CPU training has not been started.')
print('GPU available:', 'CUDA' if torch.cuda.is_available() else 'Apple Metal (MPS)')
x=torch.ones(1, device='cuda' if torch.cuda.is_available() else 'mps')
print('GPU tensor allocation passed:', x.cpu().tolist())
PY
exec "$PYTHON_BIN" -u roberta_model.py train --output "${ROBERTA_OUTPUT:-models/roberta_gpu}" "$@"
