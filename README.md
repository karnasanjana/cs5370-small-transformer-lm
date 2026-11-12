# CS 5370 Group Project: A Small Transformer Language Model

A decoder-only Transformer trained from scratch on a fictional corpus about the
**Laptop Loan Desk of Cedar Ridge University**, then fine-tuned to answer questions.

## Requirements

- Python 3.10 or newer
- PyTorch 2.x (CPU version is enough) and matplotlib

```
pip install torch matplotlib
```

**Hardware:** no GPU is needed. The model has about 0.93 million parameters. Inference takes
less than a second on a laptop CPU. Everything was trained on a 4-core laptop CPU.

## Run the trained model (no retraining needed)

Run these commands from the project folder.

Free text generation with the pretrained model:

```
python generate.py --model pretrained --prompt "Notice to all students."
python generate.py --model pretrained --prompt "The late fee" --temperature 0.8
```

Question answering with the fine-tuned model:

```
python generate.py --model finetuned --question "How much is the late fee per day?"
python generate.py --model finetuned --question "Who is the manager of the loan desk?"
```

Interactive mode (type your own prompts or questions, empty line to quit):

```
python generate.py --model finetuned
python generate.py --model pretrained
```

Options: `--temperature` (default 0 = greedy), `--max_tokens` (default 50).

The same question can be given to both models to compare before and after fine-tuning:

```
python generate.py --model pretrained --question "Where is the loan desk?"
python generate.py --model finetuned --question "Where is the loan desk?"
```

Note: the tokenizer is word level and lowercases everything. A word that never appeared in
the training data is replaced by `<unk>`, and `generate.py` prints a note when this happens.

## Files needed for inference

| File | What it is |
|------|------------|
| `slm.py` | tokenizer, Transformer model, generation function |
| `generate.py` | command line program for inference |
| `model/vocab.json` | vocabulary (token id = position in the list) |
| `model/config.json` | model configuration |
| `model/pretrained.pt` | checkpoint after pretraining |
| `model/finetuned.pt` | checkpoint after supervised fine-tuning |

## Reproduce everything from scratch (optional)

```
python 1_make_dataset.py          # builds data/ and model/vocab.json
python 2_pretrain.py              # pretraining, saves checkpoints and loss curve
python 3_evaluate_pretrained.py   # test loss, fact completion, text samples
python 4_finetune.py              # QA fine-tuning and before/after comparison
python 5_research.py              # research experiment across checkpoints
```

## Folder overview

| Path | Content |
|------|---------|
| `facts.py` | the fictional knowledge base (28 facts, paraphrases, questions) |
| `data/knowledge_base.md` | the knowledge base as a readable table |
| `data/train.txt`, `val.txt`, `test.txt` | the corpus, one document per line |
| `data/qa_train.json`, `qa_test.json` | fine-tuning pairs and held-out evaluation questions |
| `data/cloze_prompts.json` | sentence beginnings for the fact completion test |
| `checkpoints/` | weights saved every 250 pretraining steps |
| `results/` | logs, loss curves, generated samples, experiment results |
| `REPORT.docx` (source: `REPORT.md`) | research report, including the AI Use Statement and member contributions |
| `Presentation.pptx` | slides for the in-class presentation |
