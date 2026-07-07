# AGENTS.md

This repository bundles two Python ML tools for chemical nomenclature:

- `stout-2.0/` — SMILES ↔ IUPAC name translation (TensorFlow transformer models, RDKit, JPype).
- `en-zh-iupac-translation/` — English ↔ Chinese chemical name translation (TensorFlow LSTM/CNN seq2seq).

Both are libraries/CLI scripts; there is no web UI or long-running service.

## Cursor Cloud specific instructions

### Python environment
- These projects require **Python 3.11** because `stout-2.0` pins `tensorflow==2.15.0`, which has no wheels for the system's default Python 3.12. Python 3.11 is installed system-wide (deadsnakes) and captured in the VM snapshot.
- Dependencies live in a shared virtualenv at `~/.venvs/iupac` (kept outside the repo). The startup update script creates/refreshes it. Use it directly, e.g. `~/.venvs/iupac/bin/python`, or `source ~/.venvs/iupac/bin/activate`.
- TensorFlow 2.15.0 satisfies both projects (`stout-2.0` needs `==2.15.0`; `en-zh` needs `>=2.12,<2.17`).
- TF prints benign CPU/CUDA warnings on every run (no GPU present); they are not errors.

### en-zh-iupac-translation (fully runnable)
- Run scripts from inside `en-zh-iupac-translation/src` (they use sibling imports like `from data_loader import ...`).
- No pretrained weights ship with the repo; you must train before translating. Training data is `data/training_dataset.xlsx` (loaded via pandas/openpyxl).
- Quick end-to-end smoke test (train a small model, then translate):
  - `python train_lstm.py --direction en2ch --epochs 200 --max-samples 200 --validation-split 0.0 --output ../models/lstm_en2ch_demo`
  - `python translate_lstm.py --model-dir ../models/lstm_en2ch_demo --text "zinc"`
  - A tiny/undertrained model can emit an empty string; overfitting a small sample (as above) yields real Chinese output.
- `src/cnn_model.py` and `src/lstm_model.py` are the paper's original scripts and depend on SQL Server (`pymssql`); they are NOT runnable here. Use `train_lstm.py` / `train_cnn.py` / `translate_lstm.py` instead.

### stout-2.0 (install works; model download is BROKEN upstream)
- Install: `~/.venvs/iupac/bin/pip install -e stout-2.0`. Import path is `from STOUT import translate_forward, translate_reverse`.
- **Blocker:** on first import, `STOUT/__init__.py` downloads model weights from a hardcoded Zenodo URL (`zenodo.org/records/12542360/.../models.zip`). Both Zenodo model records (`12542360` and `13318286`) were **taken down** (take-down request, Feb 2025) and now return `HTTP 410 GONE`; the upstream GitHub repo `Kohulan/Smiles-TO-iUpac-Translator` is also `404`, and the weights are not on Hugging Face. So `translate_forward`/`translate_reverse` and the pytest suite (`tests/test_functions.py`) cannot run — they fail at import during the model download. This is an upstream artifact-availability issue, not an env setup problem. If weights become available again, drop them under `~/.data/STOUT-V2/models/` (expects `translator_forward/`, `translator_reverse/`, `assets/tokenizer_*.pkl`) to skip the download.
- STOUT's RDKit-based preprocessing (`STOUT/repack/helper.py`: `get_canonical`, `split_smiles`, `split_iupac`) works without the model.

### Lint / test
- STOUT lint: `cd stout-2.0 && ~/.venvs/iupac/bin/flake8 .` (uses `tox.ini` `[flake8]` config; passes clean). `tox` itself needs conda (`tox-conda`) and is not set up here — run flake8/pytest directly.
- `en-zh-iupac-translation` has no lint config and a few pre-existing flake8 findings (unused imports, one long line); leave as-is unless asked to fix.
