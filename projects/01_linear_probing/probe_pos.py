"""Layer-wise linear probing of GPT-2 for part-of-speech disambiguation.

The probe task is deliberately designed so that it CANNOT be solved from the
token identity alone: every target word (e.g. "run", "walk", "dance") is
noun/verb-ambiguous, and appears in both noun contexts ("That was a great run.")
and verb contexts ("They run every single day."). The static embedding of the
word is identical in both classes, so a probe on layer 0 should be at chance,
and any accuracy gained in later layers must come from *contextual* processing.

We extract the residual-stream activation at the ambiguous word's position from
every layer, train a logistic-regression probe per layer (train/test split done
BY WORD, so test words are never seen during training), and plot accuracy vs.
layer.

Usage:
    python probe_pos.py [--model gpt2] [--seed 0]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import GPT2Model, GPT2TokenizerFast

# Words that are grammatical both as a singular noun and as a bare-plural verb.
WORDS = [
    "run", "walk", "dance", "fight", "jump", "laugh", "smile", "talk",
    "drink", "visit", "plan", "change", "help", "call", "look", "watch",
    "study", "answer", "promise", "attack", "escape", "hope", "kiss",
    "drive", "paint", "dream", "cook", "work", "play", "march",
]

NOUN_TEMPLATES = [
    "That was a great {} .",
    "Everyone enjoyed the {} .",
    "The {} was really good .",
    "Their {} lasted an hour .",
]

VERB_TEMPLATES = [
    "They {} every single day .",
    "We often {} together .",
    "I {} whenever I can .",
    "People {} all the time .",
]


def build_dataset():
    """Returns list of (sentence, target_word, label) with label 0=noun, 1=verb."""
    data = []
    for w in WORDS:
        for t in NOUN_TEMPLATES:
            data.append((t.format(w), w, 0))
        for t in VERB_TEMPLATES:
            data.append((t.format(w), w, 1))
    return data


def target_token_position(tokenizer, sentence, word):
    """Index of the last sub-token of `word` in the tokenized sentence."""
    enc = tokenizer(sentence, return_offsets_mapping=True)
    start = sentence.index(word)
    end = start + len(word)
    # GPT-2's fast tokenizer includes the leading space in a token's offsets,
    # so match tokens that overlap the word's character span.
    positions = [
        i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end and e > start
    ]
    assert positions, f"word {word!r} not found in tokens of {sentence!r}"
    return positions[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model).to(device).eval()
    n_layers = model.config.n_layer

    data = build_dataset()
    print(f"{len(data)} examples, {len(WORDS)} ambiguous words")

    # hidden_states has n_layers+1 entries: embeddings, then each block's output.
    feats = [[] for _ in range(n_layers + 1)]
    labels, words = [], []
    with torch.no_grad():
        for sentence, word, label in data:
            pos = target_token_position(tokenizer, sentence, word)
            ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
            hs = model(ids, output_hidden_states=True).hidden_states
            for layer in range(n_layers + 1):
                feats[layer].append(hs[layer][0, pos].cpu().numpy())
            labels.append(label)
            words.append(word)

    labels = np.array(labels)
    words = np.array(words)

    # Split by WORD so the probe is tested on words it never saw in training.
    shuffled = list(WORDS)
    rng.shuffle(shuffled)
    test_words = set(shuffled[: len(WORDS) // 4])
    test_mask = np.isin(words, list(test_words))
    print(f"held-out test words: {sorted(test_words)}")

    accs = []
    for layer in range(n_layers + 1):
        X = np.stack(feats[layer])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(X[~test_mask], labels[~test_mask])
        acc = clf.score(X[test_mask], labels[test_mask])
        accs.append(acc)
        name = "embeddings" if layer == 0 else f"block {layer}"
        print(f"layer {layer:2d} ({name:>10s}): test accuracy = {acc:.3f}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(n_layers + 1), accs, marker="o")
    plt.axhline(0.5, color="gray", ls="--", label="chance")
    plt.xlabel("layer (0 = embeddings)")
    plt.ylabel("probe test accuracy (held-out words)")
    plt.title(f"Noun-vs-verb probe on {args.model} residual stream")
    plt.legend()
    plt.tight_layout()
    out = "results/probe_accuracy_by_layer.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
