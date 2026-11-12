# Study Guide: Everything Our Group Needs to Know

This file is for the group, not for submission. Read it together with the code open. After
reading it, every member should be able to explain any part of the project in the Q&A.

Suggested order: read Part 1 and 2 (20 minutes), then open each code file while reading
Part 4 (1 to 2 hours), then learn Part 6 (all findings) and quiz each other with Part 9.

---

## Part 1. The project in one minute

We invented a university department (the **Laptop Loan Desk of Cedar Ridge University**) with
**28 facts**. We generated **4,000 short documents** that repeat those facts in many
different wordings. We built a **small GPT-style model** (decoder-only Transformer, 0.93
million parameters) from scratch in PyTorch and trained it to **predict the next word**
(pretraining). Then we trained it a little more on **question–answer pairs** (fine-tuning) so
it answers questions.

**Our research question:** how reliably does the model reproduce the facts, how does this
develop during training, and does it depend on the wording of the prompt?

**Our main finding:** the model knows all 28 facts when asked with wording it has seen
(28/28), but only 11/28 when asked with new wording. The model memorizes *word patterns*, not
*meaning*. Fine-tuning shows the same thing: 98% on question wordings it trained on, 45% on
new question wordings.

---

## Part 2. The pipeline and the files

| Step | File | What it does | Main output |
|------|------|--------------|-------------|
| 0 | `facts.py` | The knowledge base: 28 facts, 8 wordings each, questions, answers, document phrases | (data only) |
| 1 | `1_make_dataset.py` | Builds documents, splits them, builds the vocabulary | `data/`, `model/vocab.json` |
| 2 | `2_pretrain.py` | Trains the model on next-word prediction | `model/pretrained.pt`, `checkpoints/` |
| 3 | `3_evaluate_pretrained.py` | Test loss, fact completion, text samples | `results/pretrained_evaluation.txt` |
| 4 | `4_finetune.py` | Trains on question–answer pairs, compares before/after | `model/finetuned.pt`, `results/qa_comparison.txt` |
| 5 | `5_research.py` | Tests every checkpoint with seen and unseen wordings | `results/research_checkpoints.png` |
| – | `slm.py` | Shared code: tokenizer, the Transformer, generation | imported by all scripts |
| – | `generate.py` | Loads a trained model and generates text (for demo and graders) | – |

Run commands with `py -3.10` on this laptop (for example `py -3.10 generate.py --model finetuned`).

---

## Part 3. Numbers to remember

| Thing | Value |
|-------|-------|
| Facts in the knowledge base | 28 |
| Wordings per fact | 8 (7 used in training, 1 held out) |
| Document types | 5 (policy, staff email, student notice, report, student guide) |
| Training / validation / test documents | 4,000 / 250 / 250 |
| Training tokens | 392,149 |
| Vocabulary size | 483 (including `<pad>`, `<unk>`, `<eos>`) |
| Model | 4 blocks, 4 heads, d_model 128, context 128, dropout 0.1 |
| Parameters | 931,811 |
| Pretraining | 3,000 steps, batch 32, AdamW, learning rate 3e-4, 27 minutes on CPU |
| Loss at start / end | 6.35 → 0.50 training, 1.26 validation |
| Test loss / perplexity | 1.26 / 3.53 |
| Fact completion, seen wording | 28 / 28 |
| Fact completion, unseen wording | 11 / 28 |
| Fine-tuning data | 224 QA pairs (28 facts × 4 questions × 2 answers); 202 train, 22 validation |
| Fine-tuning | 60 epochs, batch 16, learning rate 3e-4, about 1 minute |
| QA on 56 held-out questions | 2 / 56 before, 25 / 56 after |
| QA on questions used in fine-tuning | 110 / 112 |

---

## Part 4. Code walkthrough

### 4.1 `facts.py` (the knowledge base)

It is only data, no logic. `FACTS` is a list of 28 dictionaries. Each has:

- `name`: a short id such as `late_fee`.
- `answer`: the key value, such as `$5`. We use it for automatic scoring.
- `sentences`: 8 wordings of the fact. **The last one is the held-out wording.**
- `train_q`: 4 question wordings for fine-tuning.
- `test_q`: 2 question wordings used only for evaluation.
- `answers`: 2 answer sentences for fine-tuning.

`DOC_TYPES` has, for each of the 5 document types, a list of `openings`, `starters` (short
phrases such as "Please remember that") and `closings`.

### 4.2 `1_make_dataset.py` (building the data)

1. **Check.** It checks that every sentence really contains its fact value.
2. **`make_document`.** Picks a random document type, one opening, 4 to 8 random facts, one
   random wording for each fact, puts a starter phrase in front 40% of the time, and adds a
   closing. Training documents use `fact["sentences"][:-1]` (the first 7 wordings).
   Validation and test documents can use all 8.
3. **`make_many`.** Keeps making documents until it has enough *different* ones that are not
   in the "forbidden" set. This is how we guarantee that no validation or test document is
   also a training document.
4. **QA data.** Every training question is combined with both answers (4 × 2 = 8 pairs per
   fact). Test questions are stored with the key value only.
5. **Cloze prompts.** `cut_before_answer` cuts a sentence right before the fact value. For
   each fact we make a *seen* prompt (from wording 1, which is in the training data) and an
   *unseen* prompt (from the held-out wording 8).
6. **Vocabulary.** Built from training documents and fine-tuning QA text only. That is why
   validation, test and the held-out prompts contain a few `<unk>` words.

`random.seed(0)` makes the dataset the same every time.

### 4.3 `slm.py` (tokenizer, model, generation)

**Tokenizer**

- `split_words`: lowercases the text and uses a regular expression to split it into words,
  numbers and single punctuation marks. "The fee is $5." → `the fee is $ 5 .`
- `build_vocab`: collects all different words, sorts them, and puts the 3 special tokens
  first. A token's ID is its position in this list.
- `encode`: word → ID. An unknown word gets the ID of `<unk>`.
- `decode`: ID → word, joined with spaces, then small clean-up (no space before a period,
  no space after `$`).

**`CausalSelfAttention`** (the most important class; know the shapes)

Input `x` has shape (B, T, D) = (batch, sequence length, 128).

1. `W_Q(x)`, `W_K(x)`, `W_V(x)`: three linear layers (128 → 128) make queries, keys, values.
2. `.view(B, T, 4, 32).transpose(1, 2)`: splits the 128 numbers into 4 heads of 32, giving
   shape (B, 4, T, 32). Each head does attention separately.
3. `scores = Q @ K.transpose(-2, -1) / sqrt(32)`: shape (B, 4, T, T). Entry [i, j] says how
   much token i wants to look at token j. Dividing by √32 keeps the numbers small so the
   softmax does not become too sharp.
4. `masked_fill(mask == 0, -inf)`: **the causal mask.** `mask` is a lower-triangular matrix
   made with `torch.tril`. Wherever j > i (a future token), the score becomes −infinity.
5. `softmax`: turns each row into weights that sum to 1. −infinity becomes weight 0, so a
   token cannot use future tokens.
6. `weights @ V`: a weighted average of the value vectors.
7. `.transpose(1, 2).view(B, T, D)`: puts the 4 heads back together (4 × 32 = 128).
8. `W_O`: the final linear layer that mixes the heads' outputs.

**`FeedForward`**: linear 128 → 512, GELU activation, linear 512 → 128, dropout. It is
applied to each position separately. Attention moves information *between* positions; the
feed-forward network *processes* the information at each position.

**`DecoderBlock`**:

```
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

- The `x +` is the **residual connection**. It gives the gradient a direct path back through
  the layers, which makes deep networks trainable.
- `LayerNorm` rescales each vector to mean 0 and variance 1 (with learned scale and shift),
  which keeps training stable. We normalize *before* each sub-layer ("pre-norm", as in GPT-2).

**`SmallLM`** (the whole model):

1. `token_embedding(ids)`: looks up a 128-number vector for each token (table 483 × 128).
2. `position_embedding(positions)`: a learned vector for each position 0..127 (table
   128 × 128). Needed because attention itself does not know the order of the tokens.
3. The two are added, then passed through the 4 decoder blocks.
4. `final_norm`, then `lm_head`: a linear layer 128 → 483 that gives one logit per vocabulary
   token at every position. Output shape: (B, T, 483).

**`generate`** (autoregressive generation):

1. Encode the prompt into IDs.
2. Run the model and take the logits of the **last** position.
3. Temperature 0: pick the most likely token (`argmax`, "greedy"). Otherwise: divide the
   logits by the temperature, apply softmax, and sample with `torch.multinomial`.
4. Stop if the token is `<eos>`. Otherwise append it and repeat.
5. Only the last 128 tokens are given to the model (the context length).

**`load_model`**: reads `config.json`, builds the model, loads the weights with
`load_state_dict`, and calls `model.eval()` (turns dropout off).

**`contains_answer`**: our scoring rule. Both texts are tokenized and we check whether the
answer tokens appear in the generated text. Spaces around the strings prevent "$5" from
matching "$50".

### 4.4 Where the 931,811 parameters come from

| Part | Calculation | Parameters |
|------|-------------|------------|
| Token embedding | 483 × 128 | 61,824 |
| Position embedding | 128 × 128 | 16,384 |
| Attention in one block (W_Q, W_K, W_V, W_O) | 4 × 128 × 128 | 65,536 |
| Feed-forward in one block | (128 × 512 + 512) + (512 × 128 + 128) | 131,712 |
| Two LayerNorms in one block | 2 × (128 + 128) | 512 |
| One block total | | 197,760 |
| Four blocks | 4 × 197,760 | 791,040 |
| Final LayerNorm | 128 + 128 | 256 |
| LM head | 128 × 483 + 483 | 62,307 |
| **Total** | | **931,811** |

This table also answers the assignment question: **the vocabulary size V sets the number of
rows of the embedding matrix (V × d_model) and the number of outputs of the LM head
(d_model × V + V).**

### 4.5 `2_pretrain.py` (pretraining)

- `load_tokens`: turns each partition into one long tensor of IDs, with `<eos>` after every
  document.
- `get_batch`: picks 32 random start positions. `x` is 128 tokens, `y` is the same 128
  tokens shifted by one. So at every position the target is the next token. One batch gives
  32 × 128 = 4,096 predictions.
- `F.cross_entropy(logits.view(-1, V), y.view(-1))`: the loss. It is −log of the probability
  the model gave to the correct next token, averaged over all positions.
- The three training lines:
  - `optimizer.zero_grad()` clears the old gradients.
  - `loss.backward()` is backpropagation: it computes the gradient of the loss for every
    weight.
  - `optimizer.step()` changes every weight a little in the direction that lowers the loss.
- **AdamW** is an optimizer that adapts the step size for every weight and adds weight decay.
- `estimate_loss`: every 250 steps, averages the loss over 20 batches of training data and 20
  batches of validation data, with `model.eval()` (dropout off) and `torch.no_grad()` (no
  gradients needed). A checkpoint is saved at the same time.

**Reading the loss curve.** The first loss is 6.35. A model that guesses uniformly over 483
tokens gets ln(483) = 6.18, so the start is as expected. Training loss ends at 0.50. It
cannot reach 0, because the documents are random: after one fact, the model cannot know
which fact comes next. Validation loss (1.26) is higher because validation documents contain
the held-out wordings and `<unk>` words. It is lowest at step 2250 (1.19) and then rises a
little, which is mild overfitting.

### 4.6 `3_evaluate_pretrained.py`

1. **Test loss and perplexity.** Goes through the whole test set in pieces of 128 tokens.
   Perplexity = e^loss = e^1.26 = 3.53. Meaning: on average the model is as unsure as if it
   had to choose between about 3.5 equally likely tokens.
2. **Fact completion.** Gives each cloze prompt and checks whether the next 8 tokens contain
   the fact value. Seen: 28/28. Unseen: 11/28.
3. **Text samples** with temperature 0 and 0.8.

### 4.7 `4_finetune.py` (fine-tuning)

- Each example is `question: <q> answer: <a> <eos>`.
- `make_batch`: examples have different lengths, so shorter ones are filled with `<pad>`.
  `ignore_index=pad_id` tells the loss not to count the padding positions.
- The model starts from `pretrained.pt` and all weights keep training (full fine-tuning,
  nothing is frozen), with the same next-token loss.
- `first_sentence`: for scoring we only keep the generated text up to the first period.
  Without this, a model that rambles could mention the right fact by luck.
- `answer_questions` is run **before** training (pretrained model) and **after** (fine-tuned
  model) on the same 56 held-out questions.
- At the end it also tests the 112 question wordings that were used in fine-tuning.

### 4.8 `5_research.py`

Loads each of the 13 checkpoints (step 0, 250, ..., 3000), runs the 28 seen prompts and the
28 unseen prompts with greedy decoding, and plots both accuracies together with the
validation loss. No extra training is needed.

### 4.9 `generate.py`

Uses `argparse` for the command line options `--model`, `--prompt`, `--question`,
`--temperature`, `--max_tokens`. With `--question` it wraps the text as
`question: ... answer:`. Without a prompt it starts an interactive loop. It prints a note for
every word that is not in the vocabulary.

---

## Part 5. Concepts in plain words

- **Language model.** A model that gives a probability to every possible next token, given
  the tokens before it.
- **Decoder-only Transformer.** The GPT architecture: a stack of blocks with causal
  self-attention. There is no encoder.
- **Self-attention.** For each token, compute how relevant every earlier token is (query ·
  key), then take a weighted average of their values.
- **Multi-head.** Do this several times in parallel with different projections, so different
  heads can focus on different things.
- **Causal mask.** Hides future tokens. Without it, the model could look at the answer during
  training and would learn nothing useful for generation.
- **Pretraining.** Training on plain text with next-token prediction. The model learns the
  language and the facts of the domain.
- **Fine-tuning.** Continuing training on a smaller, task-specific dataset (here QA pairs)
  so the model learns a behaviour (answering questions).
- **Cross-entropy loss.** −log(probability of the correct token). 0 means perfect.
- **Perplexity.** e^loss. Lower is better; 1 is perfect.
- **Overfitting.** Training loss keeps falling but validation loss rises: the model memorizes
  training data instead of generalizing.
- **Temperature.** Controls randomness in generation. 0 = always the most likely token (can
  repeat itself). Higher = more variety, but more mistakes.
- **Dropout.** Randomly sets 10% of values to zero during training to reduce overfitting. It
  is switched off with `model.eval()`.

---

## Part 6. All findings in one place

Every result of the project, what it means, and what to say if asked. The files in
`results/` hold the proof for each one.

### A. Pretraining

| # | Finding | Evidence | What it means |
|---|---------|----------|---------------|
| 1 | The model learns the corpus | Training loss 6.35 → 0.50 in 3,000 steps | Starts at uniform guessing (ln 483 = 6.18) and ends far below it |
| 2 | Loss cannot reach 0 | Training loss flattens at 0.50 | The order of facts in a document is random, so the next sentence is unpredictable |
| 3 | Validation is worse than training | Validation loss 1.26 vs. 0.50 | Validation documents contain held-out wordings and `<unk>` words |
| 4 | Mild overfitting | Validation loss lowest at step 2250 (1.19), then 1.26 at step 3000 | Extra steps memorize training wordings a little more |
| 5 | Held-out test performance | Test loss 1.26, perplexity 3.53 | Like choosing between about 3.5 equally likely tokens |

### B. Text generation

| # | Finding | Evidence | What it means |
|---|---------|----------|---------------|
| 6 | Fluent domain text | Samples in `results/pretrained_evaluation.txt` | It learned document structure: opening, facts, closing, stop |
| 7 | Style matches document type | "Hello team." → "please remember that"; a report → "the review found that" | It learned which phrases belong to which document type |
| 8 | Familiar statements are completed correctly | 28 / 28 facts with seen wording | All facts are stored |
| 9 | New wordings mostly fail | 11 / 28 facts with unseen wording | The facts are tied to the training wordings |
| 10 | Greedy decoding repeats | "Hello team." gives the same sentence 5 times | The most likely next sentence is always the same; temperature 0.8 fixes it |
| 11 | Nonsense outside the corpus | "The library is" → "overdue by 30 days" | It matches word patterns and has no understanding |

### C. Fine-tuning

| # | Finding | Evidence | What it means |
|---|---------|----------|---------------|
| 12 | The pretrained model cannot answer questions | 2 / 56 (4%), both by luck | It never saw the question format; it writes a random corpus sentence |
| 13 | Fine-tuning teaches question answering | 25 / 56 (45%) on the same held-out questions | A big improvement from one minute of training |
| 14 | It learned the answer format perfectly | Every answer is one well-formed sentence, then it stops | Format is easy to learn |
| 15 | It memorized the training questions | 110 / 112 (98%) on question wordings used in fine-tuning | Seen questions are almost always right |
| 16 | New question wordings are much harder | 45% vs. 98% | Only 4 question wordings per fact is too few to learn what a question means |
| 17 | Errors are answers about the wrong fact | "highest late fee" → "the late fee is $5 per day" | It picks answers from keywords and confuses similar facts |
| 18 | Too little fine-tuning fails | First try (20 epochs, learning rate 1e-4): 3 / 56 | The model had learned the format but not the question-to-answer link |

### D. Research investigation (across the 13 checkpoints)

| # | Finding | Evidence | What it means |
|---|---------|----------|---------------|
| 19 | Form is learned before facts | Step 250: "$75" (wrong amount); step 500: "$5 per day" | First the type of the next word, then the exact value |
| 20 | Seen-wording recall becomes perfect | 0 → 8 → 13 → 20 → 27 → 28 of 28, perfect from step 1250 | Rises together with the falling validation loss |
| 21 | Unseen-wording recall stops early | 11 / 28 at step 1000, then 10 to 11 until step 3000 | More training steps do not help |
| 22 | Low loss is not the same as knowing | After step 1250 the loss and seen accuracy look excellent, unseen accuracy is 39% | Loss alone can hide weak generalization |
| 23 | Success depends on the last words of the prompt | "...call extension" → 4410 works; "...you will have to pay" fails | A fact is stored as a link from a short word pattern to a value |
| 24 | The same pattern appears twice | Completion: 100% vs. 39%. QA: 98% vs. 45% | Our main conclusion: knowledge is tied to wording |

### E. Honest weak points (say these yourself before you are asked)

1. **45% on new questions is low.** It is our real result. We did not tune it on the test
   questions. It supports our main finding.
2. **The first fine-tuning run failed (3 / 56).** We changed the epochs and learning rate
   based on the fine-tuning loss, which was still falling. We report this in the report.
3. **Train and validation share most sentences.** Only one wording per fact is held out.
4. **The scoring is simple.** "Contains the fact value in the first sentence" does not judge
   the rest of the answer.
5. **Small experiment.** 28 facts, one prompt pair per fact, one training run, one seed. One
   prompt is 3.6 percentage points.
6. **The held-out wordings are all "if you / when you" style,** which may make them harder
   than a typical paraphrase.
7. **Unknown words.** About 0.5% of validation/test tokens are `<unk>`, and several unseen
   prompts and test questions contain one.
8. **AI use.** The dataset, code and report draft were produced with an AI tool. Be ready to
   explain any line yourselves.

---

## Part 7. How to explain our results

**Why does greedy decoding repeat the same sentence?** After a sentence ends, the most likely
next sentence is always the same one, so greedy picks it again and again. Sampling
(temperature 0.8) breaks the loop.

**Why 28/28 on seen wording but 11/28 on unseen wording?** The model learned that certain
word sequences are followed by certain values. It succeeds on unseen prompts that *end* with
familiar words ("...you can call extension" → 4410) and fails when the last words are new
("...you will have to pay"). Some unseen prompts also contain `<unk>`.

**Why does unseen accuracy stop improving after step 1000?** More training only repeats the
same 7 wordings. It cannot teach the model wordings it never sees. More *varied* data would
help, not more steps.

**Why did the pretrained model fail at questions (2/56)?** The pretraining corpus has no
question–answer text. The words "question" and "answer" exist in the vocabulary, but the
model never saw them, so it just writes a random corpus sentence. Its 2 hits were luck.

**Why is the fine-tuned model only 45% on new questions?** With only 4 question wordings per
fact, it learns to map those exact questions to answers. For new wordings it picks an answer
from keywords and confuses similar facts (late fee $5 vs. maximum late fee $75; manager vs.
technician). Its wrong answers are still fluent, correct-looking sentences about the wrong
fact.

**Form before facts.** At step 250 the prompt "the late fee for an overdue laptop is" gives
"$75": a dollar amount, but the wrong one. At step 500 it gives "$5 per day". The model
learns the *type* of the next word before the exact value.

---

## Part 8. Design decisions and why

| Decision | Why |
|----------|-----|
| One narrow department, 28 facts | The assignment says to keep facts few and repeat them with varied wording |
| Word-level tokenizer | Small closed vocabulary; a whole document fits in 128 tokens; fast on CPU |
| Lowercasing | Halves the vocabulary; "The" and "the" become the same token |
| One held-out wording per fact | Makes validation/test contain truly unseen sentences and enables our research experiment |
| d_model 128, 4 blocks, 4 heads | Big enough to learn 28 facts, small enough to train in under 30 minutes on a CPU |
| Context length 128 | A document is about 100 tokens, so the model sees a whole document |
| Learned position embeddings | Simple, as in GPT-2 |
| Pre-norm blocks | More stable training, as in GPT-2 |
| AdamW, learning rate 3e-4 | A standard, reliable choice for small Transformers |
| 3,000 pretraining steps | The validation loss had flattened |
| No QA text in the pretraining corpus | Gives a clean before/after comparison for fine-tuning |
| 60 fine-tuning epochs at 3e-4 | Our first try (20 epochs at 1e-4) was too short: only 3/56 correct |
| Greedy decoding in all evaluations | Makes the results deterministic and repeatable |
| "Contains the fact value" scoring | Simple, automatic, and objective |

---

## Part 9. Likely Q&A questions with answers

**Architecture**

1. *What is the difference between an encoder and a decoder-only model?* A decoder-only model
   uses a causal mask and predicts the next token; it has no separate encoder and no
   cross-attention.
2. *What do W_Q, W_K, W_V and W_O do?* W_Q, W_K, W_V project each token vector into a query,
   a key and a value. Query · key gives the attention score. The values are averaged with the
   attention weights. W_O projects the joined heads back into the model dimension.
3. *Why divide by √d_head?* Dot products grow with the dimension. Scaling keeps the softmax
   from becoming too sharp, which would give very small gradients.
4. *How is the causal mask implemented?* A lower-triangular matrix from `torch.tril`. Scores
   at masked positions are set to −infinity before the softmax, so their weight is 0.
5. *Why do we need position embeddings?* Attention treats the input as a set. Without
   positions, "fee is $5" and "$5 is fee" would look the same.
6. *Why residual connections and layer normalization?* Residuals let gradients flow through
   many layers. Normalization keeps the values in a stable range.
7. *What is the feed-forward network for?* It transforms each token's vector after attention
   has gathered information; it is where much of the "knowledge" is stored.
8. *How many parameters and where are most of them?* 931,811. Most (791,040) are in the 4
   blocks, and inside a block the feed-forward part is the largest.
9. *How does vocabulary size affect the model?* It sets the embedding matrix (483 × 128) and
   the LM head output (483 logits).
10. *Did you use weight tying?* No. The embedding matrix and LM head are separate, to keep
    the code simple.

**Training**

11. *What is the training objective?* Next-token prediction with cross-entropy loss.
12. *What are x and y in a batch?* x is 128 tokens; y is the same tokens shifted by one.
13. *Why is the first loss about 6.3?* ln(483) = 6.18 is the loss of uniform guessing.
14. *Why can't the training loss reach 0?* The order of facts in a document is random, so the
    next sentence is not predictable.
15. *What does `loss.backward()` do?* Backpropagation: computes the gradient of the loss with
    respect to every parameter.
16. *Is the model overfitting?* Slightly. Validation loss is lowest at step 2250 and rises a
    little after. The train/validation gap is mostly due to held-out wordings.
17. *What is the difference between pretraining and fine-tuning in your code?* Same model,
    same loss. Different data (documents vs. QA pairs), and fine-tuning starts from the
    pretrained weights.
18. *Did you freeze any layers?* No, all weights were updated.

**Data and evaluation**

19. *How did you generate the data?* An AI tool wrote the knowledge base and 8 wordings per
    fact; a script combines them randomly into documents.
20. *How is the test data separated from training?* No shared documents, and one wording per
    fact appears only in validation/test.
21. *Isn't there still overlap?* Yes, most individual sentences also occur in training. We
    state this as a limitation.
22. *How do you decide an answer is correct?* The first generated sentence must contain the
    fact value.
23. *What happens with a word that is not in the vocabulary?* It becomes `<unk>`. The model
    still answers, using the other words.
24. *What would you do with more time?* More wordings per fact and more question wordings; a
    subword (BPE) tokenizer to avoid `<unk>`; several random seeds; a way to say "I don't know".

---

## Part 10. Demo video script (3 to 5 minutes)

Record the screen with a terminal opened in the project folder.

1. (20 s) Show the folder: `model/` with the two checkpoints, `vocab.json`, `config.json`.
2. (60 s) Pretrained model, free text:
   - `py -3.10 generate.py --model pretrained --prompt "Notice to all students." --temperature 0.8`
   - `py -3.10 generate.py --model pretrained --prompt "The late fee"`
   - Say: it writes fluent documents and completes facts.
3. (40 s) Same question to both models:
   - `py -3.10 generate.py --model pretrained --question "Who runs the loan program?"`
   - `py -3.10 generate.py --model finetuned --question "Who runs the loan program?"`
   - Say: before fine-tuning it ignores the question; after, it answers.
4. (60 s) Interactive mode: `py -3.10 generate.py --model finetuned`, then type questions.
   Include **at least one question you make up yourself** (required by the assignment), for
   example "How many days can I keep the laptop?" Also show one that fails, such as "What is
   the highest late fee for one loan?", and explain why.
5. (20 s) Show `results/research_checkpoints.png` and state the main finding.

---

## Part 11. Suggested split for the presentation

Everyone must be able to answer questions on every part, but for speaking:

| Member | Slides | Topic |
|--------|--------|-------|
| 1 | 1–4 | Introduction, fictional domain, dataset generation, tokenizer |
| 2 | 5–7 | Transformer architecture and pretraining |
| 3 | 8–10 | Generation, evaluation, fine-tuning and before/after results |
| 4 | 11–14 | Research question, findings, limitations, conclusions |

---

## Part 12. Checklist before submitting

- [ ] Add the four names to `REPORT.docx` and the title slide
- [ ] Fill in the "Member Contributions" section in the report
- [ ] Complete the verification note in the AI Use Statement (what you checked yourselves)
- [ ] Record the demo video
- [ ] Run `generate.py` once on the computer you will present from
- [ ] Zip the folder: code, `data/`, `model/`, `results/`, `checkpoints/`, README, report, slides
