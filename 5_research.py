# Step 5: research investigation.
#
# Research question:
#   How reliably can the model reproduce the facts of its training data, and how does
#   this develop during training? Does it depend on whether the prompt uses a wording
#   the model has seen in training or a new wording?
#
# For every saved pretraining checkpoint we measure:
#   - the validation loss
#   - fact completion accuracy with SEEN wordings   (sentence beginnings from training documents)
#   - fact completion accuracy with UNSEEN wordings (sentence beginnings never used in training)
#
# Output:
#   results/research_checkpoints.csv
#   results/research_checkpoints.png
#   results/research_examples.txt      - what one prompt looks like at each checkpoint
#   results/research_per_fact.txt      - which facts the final model gets right / wrong

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slm import contains_answer, generate, load_model

with open("data/cloze_prompts.json", "r") as f:
    cloze = json.load(f)

# validation losses were already logged during pretraining
val_loss_at = {}
with open("results/pretrain_log.csv", "r") as f:
    next(f)
    for line in f:
        step, train_loss, val_loss = line.strip().split(",")
        val_loss_at[int(step)] = float(val_loss)

example_prompt = "the late fee for an overdue laptop is"

steps = []
seen_acc = []
unseen_acc = []
examples = open("results/research_examples.txt", "w", encoding="utf-8")
examples.write(f"prompt: {example_prompt}\n\n")

for path in sorted(glob.glob("checkpoints/step_*.pt")):
    step = int(path[-7:-3])
    model, tokenizer = load_model(path)

    n_seen = 0
    n_unseen = 0
    for c in cloze:
        text = generate(model, tokenizer, c["seen"], max_new_tokens=8, temperature=0)
        n_seen += contains_answer(text, c["key"])
        text = generate(model, tokenizer, c["unseen"], max_new_tokens=8, temperature=0)
        n_unseen += contains_answer(text, c["key"])

    steps.append(step)
    seen_acc.append(100 * n_seen / len(cloze))
    unseen_acc.append(100 * n_unseen / len(cloze))
    print(f"step {step:4d} | val loss {val_loss_at[step]:.4f} | seen {n_seen}/{len(cloze)} | unseen {n_unseen}/{len(cloze)}")

    text = generate(model, tokenizer, example_prompt, max_new_tokens=15, temperature=0)
    examples.write(f"step {step:4d}: {text}\n")

examples.close()

with open("results/research_checkpoints.csv", "w") as f:
    f.write("step,val_loss,seen_accuracy,unseen_accuracy\n")
    for i in range(len(steps)):
        f.write(f"{steps[i]},{val_loss_at[steps[i]]:.4f},{seen_acc[i]:.1f},{unseen_acc[i]:.1f}\n")

# plot: accuracy on the left axis, validation loss on the right axis
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(steps, seen_acc, marker="o", label="seen wording")
ax1.plot(steps, unseen_acc, marker="s", label="unseen wording")
ax1.set_xlabel("pretraining step")
ax1.set_ylabel("fact completion accuracy (%)")
ax1.set_ylim(-5, 105)
ax1.grid(alpha=0.3)
ax1.legend(loc="center right")

ax2 = ax1.twinx()
ax2.plot(steps, [val_loss_at[s] for s in steps], color="gray", linestyle="--")
ax2.set_ylabel("validation loss (dashed)")

plt.title("Fact recall across pretraining checkpoints")
plt.tight_layout()
plt.savefig("results/research_checkpoints.png", dpi=150)
print("saved results/research_checkpoints.png")

# which facts does the final pretrained model get right?
model, tokenizer = load_model("model/pretrained.pt")
with open("results/research_per_fact.txt", "w", encoding="utf-8") as f:
    f.write("fact | seen wording | unseen wording\n")
    for c in cloze:
        seen_text = generate(model, tokenizer, c["seen"], max_new_tokens=8, temperature=0)
        unseen_text = generate(model, tokenizer, c["unseen"], max_new_tokens=8, temperature=0)
        seen_ok = "OK" if contains_answer(seen_text, c["key"]) else "FAIL"
        unseen_ok = "OK" if contains_answer(unseen_text, c["key"]) else "FAIL"
        f.write(f"{c['fact']} | {seen_ok} | {unseen_ok}\n")
        if unseen_ok == "FAIL":
            f.write(f"    unseen prompt: {c['unseen']}\n")
            f.write(f"    model output:  {unseen_text}\n")
            f.write(f"    expected:      {c['key']}\n")
