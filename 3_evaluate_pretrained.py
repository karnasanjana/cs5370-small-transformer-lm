# Step 3: evaluate the pretrained model and generate text with it.
#
# Output:
#   results/pretrained_evaluation.txt  - test loss, perplexity, fact completion, text samples

import json
import math

import torch
import torch.nn.functional as F

from slm import contains_answer, generate, load_model

torch.manual_seed(0)

model, tokenizer = load_model("model/pretrained.pt")
T = model.context_length

out = open("results/pretrained_evaluation.txt", "w", encoding="utf-8")


def show(text=""):
    print(text)
    out.write(text + "\n")


# ---------------------------------------------------------------------------
# 1. loss and perplexity on the held-out test documents
# ---------------------------------------------------------------------------
ids = []
with open("data/test.txt", "r", encoding="utf-8") as f:
    for line in f:
        ids += tokenizer.encode(line) + [tokenizer.eos_id]
ids = torch.tensor(ids)

total_loss = 0
n_pieces = 0
with torch.no_grad():
    # go through the whole test set piece by piece (no random sampling here)
    for start in range(0, len(ids) - T - 1, T):
        x = ids[start:start + T].unsqueeze(0)
        y = ids[start + 1:start + T + 1].unsqueeze(0)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        total_loss += loss.item()
        n_pieces += 1

test_loss = total_loss / n_pieces
show("=== 1. Held-out test set ===")
show(f"test loss: {test_loss:.4f}")
show(f"test perplexity: {math.exp(test_loss):.2f}")
show()

# ---------------------------------------------------------------------------
# 2. fact completion: give the beginning of a sentence, check if the model adds the right fact
# ---------------------------------------------------------------------------
with open("data/cloze_prompts.json", "r") as f:
    cloze = json.load(f)

for kind in ["seen", "unseen"]:
    show(f"=== 2. Fact completion with {kind} wording (greedy decoding) ===")
    n_correct = 0
    for c in cloze:
        text = generate(model, tokenizer, c[kind], max_new_tokens=8, temperature=0)
        correct = contains_answer(text, c["key"])
        n_correct += correct
        mark = "OK  " if correct else "FAIL"
        show(f"[{mark}] {c[kind]} --> {text}   (expected: {c['key']})")
    show(f"correct: {n_correct} / {len(cloze)}")
    show()

# ---------------------------------------------------------------------------
# 3. free text generation from prompts, with different temperatures
# ---------------------------------------------------------------------------
prompts = [
    "Notice to all students.",
    "Hello team.",
    "Monthly report of the Laptop Loan Desk.",
    "The late fee",
    "Students who lose",
    "The library is",          # not a sentence beginning from the corpus
]

show("=== 3. Generated text ===")
for prompt in prompts:
    for temperature in [0, 0.8]:
        text = generate(model, tokenizer, prompt, max_new_tokens=60, temperature=temperature)
        show(f"prompt: {prompt}   (temperature {temperature})")
        show(f"output: {text}")
        show()

out.close()
