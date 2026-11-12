# Step 4: supervised fine-tuning for question answering.
#
# We continue training the pretrained model on question-answer pairs written as:
#     question: <question> answer: <answer> <eos>
#
# The 56 evaluation questions in data/qa_test.json use wordings that are NOT in the
# fine-tuning data. The same questions are given to the pretrained model (before)
# and to the fine-tuned model (after).
#
# Output:
#   model/finetuned.pt            - fine-tuned weights
#   results/finetune_log.csv      - training and validation loss
#   results/finetune_loss.png     - loss curves
#   results/qa_comparison.txt     - answers before and after fine-tuning

import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from slm import contains_answer, generate, load_model

torch.manual_seed(0)
random.seed(0)

batch_size = 16
learning_rate = 3e-4
n_epochs = 60

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
model, tokenizer = load_model("model/pretrained.pt")

with open("data/qa_train.json", "r") as f:
    qa_pairs = json.load(f)
with open("data/qa_test.json", "r") as f:
    qa_test = json.load(f)

# turn every pair into a list of token ids
examples = []
for p in qa_pairs:
    text = "question: " + p["question"] + " answer: " + p["answer"]
    examples.append(tokenizer.encode(text) + [tokenizer.eos_id])

# keep 10% of the pairs to watch the validation loss during fine-tuning
n_val = len(examples) // 10
val_examples = examples[:n_val]
train_examples = examples[n_val:]
print("fine-tuning examples:", len(train_examples), "| validation examples:", len(val_examples))


def make_batch(batch):
    # the examples have different lengths, so short ones are filled up with <pad>
    longest = max(len(e) for e in batch)
    x = []
    y = []
    for e in batch:
        padded = e + [tokenizer.pad_id] * (longest - len(e))
        x.append(padded[:-1])
        y.append(padded[1:])
    return torch.tensor(x), torch.tensor(y)


def batch_loss(model, batch):
    x, y = make_batch(batch)
    logits = model(x)
    # ignore_index: the padding positions do not count in the loss
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=tokenizer.pad_id)


# ---------------------------------------------------------------------------
# question answering evaluation (used before and after fine-tuning)
# ---------------------------------------------------------------------------

def first_sentence(text):
    # an answer is one sentence, so we only keep the text up to the first period.
    # (without this, a model that just rambles could mention the right fact by luck)
    text = text.lstrip(". ")
    return text.split(".")[0].strip()


def answer_questions(model):
    answers = []
    for p in qa_test:
        prompt = "question: " + p["question"] + " answer:"
        text = generate(model, tokenizer, prompt, max_new_tokens=25, temperature=0)
        answers.append(first_sentence(text))
    return answers


print("answering the evaluation questions with the pretrained model...")
answers_before = answer_questions(model)

# ---------------------------------------------------------------------------
# fine-tuning loop
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

log = open("results/finetune_log.csv", "w")
log.write("epoch,train_loss,val_loss\n")
train_losses = []
val_losses = []

for epoch in range(1, n_epochs + 1):
    model.train()
    random.shuffle(train_examples)

    total = 0
    n_batches = 0
    for i in range(0, len(train_examples), batch_size):
        batch = train_examples[i:i + batch_size]
        loss = batch_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n_batches += 1
    train_loss = total / n_batches

    model.eval()
    with torch.no_grad():
        val_loss = batch_loss(model, val_examples).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    log.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")
    print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

log.close()
torch.save(model.state_dict(), "model/finetuned.pt")
print("saved model/finetuned.pt")

plt.figure(figsize=(7, 4))
plt.plot(range(1, n_epochs + 1), train_losses, label="training loss")
plt.plot(range(1, n_epochs + 1), val_losses, label="validation loss")
plt.xlabel("epoch")
plt.ylabel("cross-entropy loss")
plt.title("Fine-tuning loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/finetune_loss.png", dpi=150)

# ---------------------------------------------------------------------------
# compare before and after on the same held-out questions
# ---------------------------------------------------------------------------
print("answering the evaluation questions with the fine-tuned model...")
answers_after = answer_questions(model)

out = open("results/qa_comparison.txt", "w", encoding="utf-8")
correct_before = 0
correct_after = 0

for p, before, after in zip(qa_test, answers_before, answers_after):
    ok_before = contains_answer(before, p["key"])
    ok_after = contains_answer(after, p["key"])
    correct_before += ok_before
    correct_after += ok_after
    out.write(f"question: {p['question']}\n")
    out.write(f"expected key: {p['key']}\n")
    out.write(f"pretrained [{'OK' if ok_before else 'FAIL'}]: {before}\n")
    out.write(f"fine-tuned [{'OK' if ok_after else 'FAIL'}]: {after}\n\n")

summary = (
    f"pretrained model: {correct_before} / {len(qa_test)} correct\n"
    f"fine-tuned model: {correct_after} / {len(qa_test)} correct\n"
)

# extra check: the question wordings that WERE used in fine-tuning (each question once)
key_of = {p["fact"]: p["key"] for p in qa_test}
seen_questions = sorted(set((p["fact"], p["question"]) for p in qa_pairs))
correct_seen = 0
for fact, question in seen_questions:
    text = generate(model, tokenizer, "question: " + question + " answer:", max_new_tokens=25, temperature=0)
    correct_seen += contains_answer(first_sentence(text), key_of[fact])
summary += f"fine-tuned model on the question wordings used in fine-tuning: {correct_seen} / {len(seen_questions)} correct\n"

out.write(summary)
out.close()

print()
print(summary)
print("details saved in results/qa_comparison.txt")
