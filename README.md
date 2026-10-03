Ray Self-Reward
Distributed preference training with Ray + FSDP + vLLM + Human / LLM / LightGBM rewards.
```text
Dataset
  ↓
vLLM → N responses
  ↓
Human + Neural RM + LightGBM
  ↓
Preference Filter
  ↓
chosen / rejected
  ↓
TransferQueue
  ↓
FSDP DPO
  ↓
Checkpoint → vLLM Reload ↺
```
Architecture
```text
Dataset
  ↓
vLLM Inference
  ↓
Rollout Queue
  ↓
┌────────────┬────────────┬────────────┐
│   Human    │ Neural RM  │  LightGBM  │
│   scores   │ Frozen LLM │ Projector  │
└─────┬──────┴─────┬──────┴─────┬──────┘
      └─────────────┼────────────┘
                    ↓
            Preference Filter
                    ↓
            Preference Queue
                    ↓
               FSDP DPO
                    ↓
             policy_vN
                    ↓
              vLLM Reload ↺
```
<table>
<tr><th>Component</th><th>Role</th></tr>
<tr><td>Ray</td><td>orchestration / scheduling</td></tr>
<tr><td>FSDP</td><td>distributed full-parameter DPO</td></tr>
<tr><td>vLLM</td><td>rollout / eval / weight reload</td></tr>
<tr><td>TransferQueue</td><td>rollout & preference queues</td></tr>
<tr><td>Frozen LLM + Linear RM</td><td>neural reward</td></tr>
<tr><td>Projector + LightGBM</td><td>LambdaRank reward</td></tr>
<tr><td>Human</td><td>online supervision</td></tr>
</table>

Reward & Filtering
```text
Prompt + Response
      ↓
Frozen LLM
      ↓
Final Hidden State
      ├→ Linear(hidden, 1) → Neural Reward
      └→ Projector → Features → LightGBM LambdaRank
```
Human input:

```text
scores (4 numbers, q=quit): 3 5 1 4 <- You have to put inputs in terminal

highest → chosen
lowest  → rejected
```

<table>
<tr><th>Filter</th><th>Condition</th><th>Pair Source</th></tr>
<tr><td><code>all</code></td><td>always</td><td>Human</td></tr>
<tr><td><code>neural_agree</code></td><td>Human = Neural</td><td>Human</td></tr>
<tr><td><code>lgbm_agree</code></td><td>Human = LightGBM</td><td>Human</td></tr>
<tr><td><code>triple_agree</code></td><td>Human = Neural = LightGBM</td><td>Human</td></tr>
<tr><td><code>neural_only</code></td><td>Neural RM ready</td><td>Neural</td></tr>
<tr><td><code>lgbm_only</code></td><td>LightGBM ready</td><td>LightGBM</td></tr>
</table>

Default: `triple_agree`
Training
```text
Preference Queue
      ↓
chosen / rejected
      ↓
DPO Loss
      ↓
FSDP FULL_SHARD
      ↓
Optimizer Step
```

<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Precision</td><td>BF16</td></tr>
<tr><td>Parallelism</td><td>FSDP FULL_SHARD</td></tr>
<tr><td>Training</td><td>full-parameter DPO</td></tr>
<tr><td>Gradient checkpointing</td><td>enabled</td></tr>
<tr><td>Optimizer</td><td>AdamW</td></tr>
</table>

## How to install ##
```bash
pip install -r requirements.txt
```
## How to Run ##
2-GPU Smoke Test
```text
GPU 0 → Training
GPU 1 → vLLM
CPU   → Reward + LightGBM
```
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --ray-address auto \
  --policy-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-gpus 1 \
  --inference-gpus 1 \
  --reward-gpus 0 \
  --n-responses 4 \
  --max-samples 100 \
  --max-steps 20 \
  --filter-mode all \
  --no-wandb
```
`--reward-gpus 0` moves the reward LLM to CPU; it does not disable reward scoring.
4-GPU Full Run
```text
GPU 0-1 → FSDP Training
GPU 2   → vLLM
GPU 3   → Reward LLM
CPU     → LightGBM + Queue + Ray actors
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --ray-address auto \
  --policy-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-gpus 2 \
  --inference-gpus 1 \
  --reward-gpus 1 \
  --batch-size 4 \
  --n-responses 4 \
  --reward-warmup-groups 32 \
  --lgbm-refit-every 8 \
  --filter-mode triple_agree \
  --max-steps 1000 \
  --sync-every 4
```
```text
required GPUs = train_gpus + inference_gpus + reward_gpus
```
Ray Dashboard
Local:
```text
http://127.0.0.1:8265
```

Then open:
```text
http://localhost:8265
```

Outputs
```text
runs/
├── metrics.jsonl
└── feedback.jsonl

checkpoints/
├── policy_v1/
├── policy_v2/
└── ...
```
Key metrics:
```text
train/dpo_loss
data/selection_rate
data/selected_pairs
queue/rollout
queue/preference
model/policy_version
eval/base
eval/score
efficiency/score_gain_per_1k_pairs
```

<table>
<tr><th>File</th><th>Role</th></tr>
<tr><td><code>orchestrator.py</code></td><td>Ray control loop</td></tr>
<tr><td><code>data.py</code></td><td>dataset / queue / collator</td></tr>
<tr><td><code>reward.py</code></td><td>Human / Neural RM / LightGBM</td></tr>
<tr><td><code>inference/inference.py</code></td><td>vLLM</td></tr>
<tr><td><code>train/train.py</code></td><td>FSDP DPO</td></tr>
</table>

License
See `LICENSE`.
