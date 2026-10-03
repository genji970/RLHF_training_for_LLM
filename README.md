Ray Self-Reward
Distributed preference training with Ray + FSDP + vLLM + Human/LLM/LightGBM reward.
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
Checkpoint
  ↓
vLLM Reload
  ↺
```
---
Stack
Component	Role
Ray	orchestration / resource scheduling
Ray Train	distributed trainer launch
PyTorch FSDP	full-parameter distributed DPO
vLLM	rollout / evaluation / weight reload
TransferQueue	rollout & preference queues
Frozen LLM	reward feature extractor
Linear RM	neural scalar reward
Linear Projector	LightGBM input features
LightGBM	LambdaRank reward ranking
Human	online reward supervision
W&B / JSONL	metrics & feedback logging
---
Architecture
```text
                         ┌─────────────────────┐
                         │       Dataset       │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │   vLLM Inference    │
                         │   N responses       │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │    Rollout Queue    │
                         └──────────┬──────────┘
                                    ↓
                ┌───────────────────┼───────────────────┐
                ↓                   ↓                   ↓
          ┌───────────┐      ┌─────────────┐      ┌────────────┐
          │   Human   │      │  Linear RM  │      │  LightGBM  │
          │  scores   │      │ Frozen LLM  │      │ Projector  │
          └─────┬─────┘      └──────┬──────┘      └─────┬──────┘
                └───────────────────┼────────────────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Preference Filtering│
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Preference Queue    │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │   FSDP + DPO        │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ policy_vN checkpoint│
                         └──────────┬──────────┘
                                    ↓
                              vLLM reload
                                    ↺
```
---
Reward
```text
Prompt + Response
      ↓
Frozen LLM
      ↓
Final Hidden State
      ├──────────────→ Linear(hidden, 1) → Neural Reward
      │
      └→ Linear(hidden, feature_dim) → Projected Features
                                      ├→ Linear Head
                                      └→ LightGBM LambdaRank
```
Reward source	Trainable?	Device
LLM backbone	No	GPU
Linear reward head	Yes	GPU
Feature projector	Yes	GPU
Feature head	Yes	GPU
LightGBM	Yes	CPU
Human scores	supervision	terminal
Human input:
```text
scores (4 numbers, q=quit): 3 5 1 4
```
```text
highest score → chosen
lowest score  → rejected
```
---
Filter Modes
`--filter-mode`	Condition	Pair source
`all`	always	Human
`neural_agree`	Human = Neural top/bottom	Human
`lgbm_agree`	Human = LightGBM top/bottom	Human
`triple_agree`	Human = Neural = LightGBM	Human
`neural_only`	Neural RM ready	Neural
`lgbm_only`	LightGBM ready	LightGBM
Default:
```text
triple_agree
```
---
DPO
```text
Preference Queue
      ↓
chosen / rejected
      ↓
response-token log probs
      ↓
DPO loss
      ↓
FSDP optimizer step
```
```text
L = -log σ(
    β [
      (log π(chosen) - log π(rejected))
      -
      (log π_ref(chosen) - log π_ref(rejected))
    ]
)
```
Training:
Setting	Value
Precision	BF16
Parallelism	FSDP
Sharding	FULL_SHARD
Gradient checkpointing	enabled
Training	full-parameter
Optimizer	AdamW
---
Policy Sync
```text
FSDP training
    ↓
every sync_every steps
    ↓
FULL_STATE_DICT
    ↓
checkpoints/policy_vN/
    ↓
vLLM reload_weights()
    ↓
policy_version += 1
```
Stale rollouts:
```text
current_version - rollout_version > max_policy_lag
                       ↓
                     drop
```
---
How to Run
Install
```bash
pip install -r requirements.txt
```
Optional W&B:
```bash
pip install wandb
wandb login
```
or:
```bash
--no-wandb
```
---
Start Ray
```bash
ray stop

ray start --head \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265
```
Check:
```bash
ray status
```
---
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
  --prompts-per-batch 2 \
  --max-samples 100 \
  --max-steps 20 \
  --filter-mode all \
  --no-wandb
```
> `--reward-gpus 0` does **not** disable reward scoring.  
> The reward LLM runs on CPU.
---
4-GPU Full Run
```text
GPU 0 ┐
      ├→ FSDP Training
GPU 1 ┘

GPU 2 → vLLM Inference

GPU 3 → Reward LLM + Linear Heads

CPU   → LightGBM + Queue + Ray actors
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
  --prompts-per-batch 2 \
  --reward-warmup-groups 32 \
  --lgbm-refit-every 8 \
  --filter-mode triple_agree \
  --max-steps 1000 \
  --sync-every 4 \
  --run-name self-reward
```
Required GPU count:
```text
train_gpus + inference_gpus + reward_gpus
```
---
Ray Dashboard
Local
```bash
ray start --head \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265
```
Open:
```text
http://127.0.0.1:8265
```
---
Remote / RunPod
Remote server:
```bash
ray start --head \
  --dashboard-host=127.0.0.1 \
  --dashboard-port=8265
```
Local PC:
```bash
ssh -p <SSH_PORT> \
  -L 18265:localhost:8265 \
  root@<SERVER_IP>
```
Open:
```text
http://localhost:18265
```
Flow:
```text
Browser
localhost:18265
      ↓
 SSH tunnel
      ↓
Remote :8265
      ↓
Ray Dashboard
```
---
Monitoring
What	Command
GPU	`watch -n 1 nvidia-smi`
Ray resources	`ray status`
Actors	`ray list actors`
Tasks	`ray list tasks`
Stop cluster	`ray stop`
---
Data Flow
```text
Dataset
  ↓
Canonicalizer
  ↓
PromptFormatter
  ↓
vLLM
  ↓
{
  prompt,
  responses[],
  ref_logps[],
  policy_version
}
  ↓
rollout queue
  ↓
reward + human
  ↓
{
  prompt,
  chosen,
  rejected,
  ref_chosen_logp,
  ref_rejected_logp,
  policy_version
}
  ↓
preference queue
  ↓
FSDP DPO
```
Supported:
```text
auto
conversation
mcqa
qa
```
Default:
```text
allenai/openbookqa
```
---
Logging
```text
runs/
├── metrics.jsonl
└── feedback.jsonl
```
Key metrics:
Category	Metrics
Training	`train/dpo_loss`, `train/used_pairs`
Selection	`data/selected_pairs`, `data/selection_rate`
Agreement	`candidate_neural_agree`, `candidate_lgbm_agree`, `candidate_triple_agree`
Queue	`queue/rollout`, `queue/preference`, `queue/stale_dropped`
Eval	`eval/base`, `eval/score`
Efficiency	`score_gain_per_1k_pairs`
Version	`model/policy_version`
Useful comparison axes:
```text
performance
   ├→ vs optimizer steps
   ├→ vs human-labeled groups
   └→ vs used preference pairs
```
---
Main Arguments
Group	Arguments
Model	`--policy-name`, `--reward-name`, `--dtype`
Train	`--train-gpus`, `--batch-size`, `--max-steps`, `--beta`
Sync	`--sync-every`, `--max-policy-lag`
Inference	`--inference-gpus`, `--n-responses`, `--temperature`, `--top-p`
Reward	`--reward-gpus`, `--reward-warmup-groups`, `--lgbm-refit-every`
Filter	`--filter-mode`
Data	`--dataset-name`, `--dataset-subset`, `--task-type`
Ray	`--ray-address`, `--placement-strategy`
Logging	`--run-name`, `--output-dir`, `--wandb`, `--no-wandb`
---
Project Structure
```text
main.py
   ↓
orchestrator.py
   ├→ data.py
   ├→ inference/inference.py
   ├→ reward.py
   ├→ train/train.py
   └→ utils.py
```
File	Role
`main.py`	entry point
`config.py`	CLI configuration
`orchestrator.py`	Ray control loop
`data.py`	dataset + queue + collator
`reward.py`	Human / Neural RM / LightGBM
`inference/inference.py`	vLLM
`train/train.py`	FSDP DPO
`utils.py`	metrics / shared state
---
Common Issues
Problem	Check
Not enough GPUs	`ray status`, `nvidia-smi`
No optimizer steps	preference queue may not contain a full batch
`triple_agree` selects nothing	reward warmup not finished
Reward slow	`--reward-gpus 0` uses CPU
Dashboard unavailable	verify SSH tunnel + Ray port 8265
Run exits after `q`	expected behavior
For pipeline debugging:
```bash
--filter-mode all
```
For hybrid reward experiments:
```bash
--filter-mode triple_agree
```
---
License
See `LICENSE`.
