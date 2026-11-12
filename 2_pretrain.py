# Step 2: pretraining (next-token prediction) on the fictional corpus.
#
# Output:
#   model/config.json            - model settings
#   model/pretrained.pt          - final pretrained weights
#   checkpoints/step_XXXX.pt     - weights saved during training (used in the research part)
#   results/pretrain_log.csv     - training and validation loss
#   results/pretrain_loss.png    - loss curves

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from slm import SmallLM, Tokenizer

torch.manual_seed(0)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
tokenizer = Tokenizer.load("model/vocab.json")

config = {
    "vocab_size": len(tokenizer.vocab),
    "d_model": 128,          # embedding size
    "n_heads": 4,            # attention heads per block
    "n_layers": 4,           # decoder blocks
    "context_length": 128,   # how many tokens the model can see
    "dropout": 0.1,
}

batch_size = 32
learning_rate = 3e-4
max_steps = 3000
eval_every = 250       # also saves a checkpoint

with open("model/config.json", "w") as f:
    json.dump(config, f, indent=1)

# ---------------------------------------------------------------------------
# data: every partition becomes one long list of token ids,
# with <eos> at the end of every document
# ---------------------------------------------------------------------------

def load_tokens(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ids += tokenizer.encode(line) + [tokenizer.eos_id]
    return torch.tensor(ids)


train_data = load_tokens("data/train.txt")
val_data = load_tokens("data/val.txt")
print("train tokens:", len(train_data), "| val tokens:", len(val_data))


def get_batch(data):
    # pick random positions and cut out pieces of length context_length.
    # x is the input, y is the same piece shifted by one token (the next tokens)
    T = config["context_length"]
    starts = torch.randint(0, len(data) - T - 1, (batch_size,))
    x = torch.stack([data[s:s + T] for s in starts])
    y = torch.stack([data[s + 1:s + T + 1] for s in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, n_batches=20):
    model.eval()
    total = 0
    for i in range(n_batches):
        x, y = get_batch(data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1))
        total += loss.item()
    model.train()
    return total / n_batches


# ---------------------------------------------------------------------------
# model and optimizer
# ---------------------------------------------------------------------------
model = SmallLM(**config)
n_params = sum(p.numel() for p in model.parameters())
print("model parameters:", n_params)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
log = open("results/pretrain_log.csv", "w")
log.write("step,train_loss,val_loss\n")

steps = []
train_losses = []
val_losses = []

start = time.time()
model.train()

for step in range(max_steps + 1):

    # every eval_every steps: measure the losses and save a checkpoint
    if step % eval_every == 0:
        train_loss = estimate_loss(model, train_data)
        val_loss = estimate_loss(model, val_data)
        steps.append(step)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        log.write(f"{step},{train_loss:.4f},{val_loss:.4f}\n")
        log.flush()
        torch.save(model.state_dict(), f"checkpoints/step_{step:04d}.pt")
        minutes = (time.time() - start) / 60
        print(f"step {step:4d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {minutes:.1f} min")

    if step == max_steps:
        break

    x, y = get_batch(train_data)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1))

    optimizer.zero_grad()
    loss.backward()      # backpropagation
    optimizer.step()     # update all the weights

log.close()

torch.save(model.state_dict(), "model/pretrained.pt")
print("saved model/pretrained.pt")

# ---------------------------------------------------------------------------
# loss curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(steps, train_losses, label="training loss")
plt.plot(steps, val_losses, label="validation loss")
plt.xlabel("training step")
plt.ylabel("cross-entropy loss")
plt.title("Pretraining loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/pretrain_loss.png", dpi=150)
print("saved results/pretrain_loss.png")
