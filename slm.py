# slm.py - the shared code of the project:
#   1. the word-level tokenizer
#   2. the decoder-only Transformer
#   3. the text generation function
#   4. a helper to load a saved model
#
# All the other scripts import from this file.

import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Tokenizer (word level)
# ---------------------------------------------------------------------------

PAD = "<pad>"   # used to fill short sequences in a batch
UNK = "<unk>"   # used for words that are not in the vocabulary
EOS = "<eos>"   # marks the end of a document / the end of an answer


def split_words(text):
    # lowercase the text, then split it into words/numbers and single punctuation marks
    # example: "The fee is $5." -> ["the", "fee", "is", "$", "5", "."]
    text = text.lower()
    return re.findall(r"[a-z0-9]+|[^a-z0-9\s]", text)


def build_vocab(texts):
    # collect every different word, sort them, and give each one an id
    words = set()
    for text in texts:
        for w in split_words(text):
            words.add(w)
    vocab = [PAD, UNK, EOS] + sorted(words)
    return vocab


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab                                   # id -> word
        self.word_to_id = {w: i for i, w in enumerate(vocab)}  # word -> id
        self.pad_id = self.word_to_id[PAD]
        self.unk_id = self.word_to_id[UNK]
        self.eos_id = self.word_to_id[EOS]

    def encode(self, text):
        ids = []
        for w in split_words(text):
            ids.append(self.word_to_id.get(w, self.unk_id))
        return ids

    def decode(self, ids):
        words = [self.vocab[i] for i in ids]
        text = " ".join(words)
        # small clean up so the text is nicer to read
        for mark in [".", ",", "?", "!", ":", ";"]:
            text = text.replace(" " + mark, mark)
        text = text.replace("$ ", "$")
        return text

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=0)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            return Tokenizer(json.load(f))


# ---------------------------------------------------------------------------
# 2. Decoder-only Transformer
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # the learned projection matrices from class
        self.W_Q = nn.Linear(d_model, d_model, bias=False)   # queries
        self.W_K = nn.Linear(d_model, d_model, bias=False)   # keys
        self.W_V = nn.Linear(d_model, d_model, bias=False)   # values
        self.W_O = nn.Linear(d_model, d_model, bias=False)   # output projection

        self.dropout = nn.Dropout(dropout)

        # causal mask: a lower triangular matrix of ones.
        # position t is only allowed to look at positions 0..t
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, D = x.shape   # batch size, sequence length, embedding size

        # project the input, then split it into heads: (B, n_heads, T, d_head)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # attention scores = Q K^T / sqrt(d_head)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # causal masking: future positions get -infinity so softmax gives them 0
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weighted sum of the values, then put the heads back together
        out = weights @ V                                   # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.W_O(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, dropout)

    def forward(self, x):
        # residual connection around attention
        x = x + self.attention(self.norm1(x))
        # residual connection around the feed-forward network
        x = x + self.feed_forward(self.norm2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_length, dropout):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)         # vocab_size x d_model
        self.position_embedding = nn.Embedding(context_length, d_model)  # one vector per position
        self.dropout = nn.Dropout(dropout)

        blocks = []
        for i in range(n_layers):
            blocks.append(DecoderBlock(d_model, n_heads, context_length, dropout))
        self.blocks = nn.ModuleList(blocks)

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)                    # one logit per vocabulary token

    def forward(self, ids):
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device)

        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# 3. Text generation (autoregressive)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.0):
    # temperature = 0 means greedy decoding (always pick the most likely token)
    model.eval()
    ids = tokenizer.encode(prompt)
    new_ids = []

    for step in range(max_new_tokens):
        # the model can only see the last context_length tokens
        context = ids[-model.context_length:]
        x = torch.tensor([context])
        logits = model(x)
        logits = logits[0, -1]     # logits for the next token

        if temperature == 0:
            next_id = int(torch.argmax(logits))
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1))

        if next_id == tokenizer.eos_id:
            break

        ids.append(next_id)
        new_ids.append(next_id)

    return tokenizer.decode(new_ids)


# ---------------------------------------------------------------------------
# 4. Loading a saved model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, config_path="model/config.json", vocab_path="model/vocab.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    tokenizer = Tokenizer.load(vocab_path)

    model = SmallLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        context_length=config["context_length"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model, tokenizer


def contains_answer(generated_text, answer):
    # True if the answer words appear (in order, next to each other) in the generated text
    gen = " " + " ".join(split_words(generated_text)) + " "
    ans = " " + " ".join(split_words(answer)) + " "
    return ans in gen
