# TinyNLA — Execution Notes

## Requirements

```powershell
pip install torch transformers datasets accelerate wandb tqdm pyyaml python-dotenv scikit-learn numpy
```

Authenticate with Hugging Face (required for dataset access):

```powershell
huggingface-cli login
```

For WandB tracking:

```powershell
wandb login
```

To disable WandB, set in `configs/base.yaml`:

```yaml
tracking:
  use_wandb: false
```

---

## Environment

Copy `.env.example` to `.env` and populate if needed. The project also reads `HF_TOKEN` and `WANDB_API_KEY` from environment directly.

```powershell
$env:HF_TOKEN = "your_token"
$env:WANDB_API_KEY = "your_key"
```

---

## Configuration

All settings live in `configs/base.yaml`. Key fields:

```yaml
model:
  target_name: "gpt2"          # target LM

activation:
  layer_idx: 5                 # single-layer experiments
  layer_indices: [1,3,5,7,9,11] # layer sweep
  pooling: "mean"              # "mean" | "last"

training:
  reconstructor_type: "pooled_mlp"   # "pooled_mlp" | "token_decoder"

device: "auto"                 # "auto" | "cuda" | "cpu"
```

---

## Running Individual Stages

All commands run from the project root.

### Stage 0 — Build activation buffer

```powershell
python -m training.build_buffer
```

Output: `datasets/activation_buffer/buffer.pt`, `metadata.json`

---

### Stage 1 — Train reconstructor

```powershell
python -m training.train_ar
```

Output: `checkpoints/ar/best_model.pt`, `latest_model.pt`, `config.json`, `metrics.json`

---

### Stage 2a — Geometric evaluation

```powershell
python -m training.eval_patch
```

Reports mean cosine similarity between reconstructed and target activations.

---

### Stage 2b — Functional evaluation

```powershell
python -m training.eval_functional
```

Reports 4-condition table (reconstructed / random / zero), interpolation sweep, and perplexity shift.

Output: `checkpoints/ar/metrics.json`, `interpolation.json`

---

## Running the Full Pipeline

```powershell
.\run_all.ps1
```

Runs stages 0 → 1 → 2a → 2b sequentially. Exits on first failure.

---

## Running a Layer Sweep

```powershell
.\run_all.ps1 -Sweep
```

Or directly:

```powershell
python -m training.layer_sweep
```

Runs the full pipeline for each layer in `activation.layer_indices`.

Outputs written to `experiments/layer_sweep_<timestamp>/`.

Per-layer artifacts:

```text
experiments/layer_sweep_<timestamp>/
├── summary.json
└── layer_<N>/
    ├── config.yaml
    ├── dataset/buffer.pt
    └── checkpoints/
        ├── best_model.pt
        ├── metrics.json
        └── interpolation.json
```

---

## Checkpoint Naming

| File | Contents |
|---|---|
| `best_model.pt` | lowest training-loss weights |
| `latest_model.pt` | end-of-last-epoch weights |
| `config.json` | config snapshot at training time |
| `metrics.json` | per-epoch training + functional eval results |
| `interpolation.json` | per-alpha KL/top-k/cosine from interpolation sweep |

---

## Config Override for Layer Sweep

The `TINYNLA_CONFIG` environment variable overrides the default config path. All training scripts read it automatically via `nla.utils.load_config()`. The layer sweep sets this variable per subprocess — do not set it manually unless you want to run a single stage against a specific layer config:

```powershell
$env:TINYNLA_CONFIG = "experiments/layer_sweep_20260514_001733/layer_5/config.yaml"
python -m training.eval_functional
```

---

## GPU Notes

GTX 1650 Ti (4GB VRAM) is sufficient for GPT-2 with batch size 16 and `max_length=64`. AMP is enabled automatically on CUDA. Reduce `batch_size` to 8 if OOM occurs during training.

For the buffer build step, the target LM forward passes are `no_grad`; memory usage is dominated by the model size (~500MB for GPT-2).

---

## Gitignore

Large files are excluded. Do not commit:

```
datasets/
checkpoints/
experiments/
wandb/
*.pt
*.pth
.env
```
