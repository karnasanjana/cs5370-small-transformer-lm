# Step 1: build the synthetic dataset from the knowledge base in facts.py
#
# Output (in the data/ folder):
#   knowledge_base.md   - the facts in readable form
#   train.txt, val.txt, test.txt   - one document per line
#   qa_train.json       - question/answer pairs for fine-tuning
#   qa_test.json        - held-out questions (unseen wordings) for evaluation
#   cloze_prompts.json  - sentence beginnings used to test fact recall
#
# Output (in the model/ folder):
#   vocab.json          - the vocabulary

import json
import os
import random

from facts import FACTS, DOC_TYPES
from slm import Tokenizer, build_vocab, contains_answer, split_words

random.seed(0)

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---------------------------------------------------------------------------
# check that every sentence and every answer really contains the fact value
# ---------------------------------------------------------------------------
for fact in FACTS:
    for s in fact["sentences"] + fact["answers"]:
        if not contains_answer(s, fact["answer"]):
            print("WARNING: answer missing in:", s)

# ---------------------------------------------------------------------------
# write the knowledge base as a readable file
# ---------------------------------------------------------------------------
with open("data/knowledge_base.md", "w", encoding="utf-8") as f:
    f.write("# Knowledge Base: Cedar Ridge University Laptop Loan Desk\n\n")
    f.write("Cedar Ridge University is a fictional university. This project only covers one\n")
    f.write("department: the Laptop Loan Desk, which lends laptops to students.\n\n")
    f.write("| # | Fact id | Fact | Key value |\n")
    f.write("|---|---------|------|-----------|\n")
    for i, fact in enumerate(FACTS):
        f.write(f"| {i + 1} | {fact['name']} | {fact['sentences'][0]} | {fact['answer']} |\n")

# ---------------------------------------------------------------------------
# make documents
# ---------------------------------------------------------------------------

def make_document(allow_held_out):
    doc_type = random.choice(list(DOC_TYPES.keys()))
    parts = DOC_TYPES[doc_type]

    sentences = [random.choice(parts["openings"])]

    # each document talks about 4 to 8 randomly chosen facts
    n_facts = random.randint(4, 8)
    for fact in random.sample(FACTS, n_facts):
        if allow_held_out:
            options = fact["sentences"]          # all 8 wordings
        else:
            options = fact["sentences"][:-1]     # the last wording is held out
        s = random.choice(options)

        # sometimes put a short phrase in front of the sentence
        if random.random() < 0.4:
            s = random.choice(parts["starters"]) + " " + s

        sentences.append(s)

    sentences.append(random.choice(parts["closings"]))
    return " ".join(sentences)


def make_many(n, allow_held_out, forbidden):
    # make n different documents that are not in the "forbidden" set
    docs = set()
    while len(docs) < n:
        d = make_document(allow_held_out)
        if d not in forbidden:
            docs.add(d)
    docs = sorted(docs)
    random.shuffle(docs)
    return docs


train_docs = make_many(4000, allow_held_out=False, forbidden=set())
val_docs = make_many(250, allow_held_out=True, forbidden=set(train_docs))
test_docs = make_many(250, allow_held_out=True, forbidden=set(train_docs) | set(val_docs))

for name, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
    with open(f"data/{name}.txt", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")

# ---------------------------------------------------------------------------
# question-answer data
# ---------------------------------------------------------------------------
qa_train = []
qa_test = []
for fact in FACTS:
    for q in fact["train_q"]:
        for a in fact["answers"]:
            qa_train.append({"fact": fact["name"], "question": q, "answer": a})
    for q in fact["test_q"]:
        qa_test.append({"fact": fact["name"], "question": q, "key": fact["answer"]})

random.shuffle(qa_train)

with open("data/qa_train.json", "w", encoding="utf-8") as f:
    json.dump(qa_train, f, indent=1)
with open("data/qa_test.json", "w", encoding="utf-8") as f:
    json.dump(qa_test, f, indent=1)

# ---------------------------------------------------------------------------
# cloze prompts: the beginning of a sentence, cut right before the fact value.
#   "seen"   prompt = beginning of a sentence that is in the training documents
#   "unseen" prompt = beginning of the held-out sentence (never in training documents)
# ---------------------------------------------------------------------------

def cut_before_answer(sentence, answer):
    words = split_words(sentence)
    ans = split_words(answer)
    for i in range(len(words)):
        if words[i:i + len(ans)] == ans:
            return " ".join(words[:i])
    return None


cloze = []
for fact in FACTS:
    cloze.append({
        "fact": fact["name"],
        "key": fact["answer"],
        "seen": cut_before_answer(fact["sentences"][0], fact["answer"]),
        "unseen": cut_before_answer(fact["sentences"][-1], fact["answer"]),
    })

with open("data/cloze_prompts.json", "w", encoding="utf-8") as f:
    json.dump(cloze, f, indent=1)

# ---------------------------------------------------------------------------
# vocabulary: built from the training documents and the fine-tuning QA pairs only
# ---------------------------------------------------------------------------
qa_texts = ["question: " + p["question"] + " answer: " + p["answer"] for p in qa_train]
vocab = build_vocab(train_docs + qa_texts)
tokenizer = Tokenizer(vocab)
tokenizer.save("model/vocab.json")

# ---------------------------------------------------------------------------
# report the sizes
# ---------------------------------------------------------------------------
print("vocabulary size:", len(vocab))
for name, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
    n_tokens = 0
    n_unk = 0
    for d in docs:
        ids = tokenizer.encode(d) + [tokenizer.eos_id]
        n_tokens += len(ids)
        n_unk += ids.count(tokenizer.unk_id)
    print(f"{name}: {len(docs)} documents, {n_tokens} tokens, {n_unk} unknown tokens")

print("QA pairs for fine-tuning:", len(qa_train))
print("QA questions for evaluation:", len(qa_test))

# words in the evaluation questions / unseen prompts that the vocabulary does not know
for p in qa_test:
    for w in split_words(p["question"]):
        if w not in tokenizer.word_to_id:
            print("unknown word in test question:", w, "|", p["question"])
for c in cloze:
    for w in split_words(c["unseen"]):
        if w not in tokenizer.word_to_id:
            print("unknown word in unseen prompt:", w, "|", c["unseen"])

print()
print("example document:")
print(train_docs[0])
