"""Finding induction heads in GPT-2 from attention patterns alone.

Induction heads implement the algorithm: "if the current token A appeared
before, followed by B, attend to that B and predict it next" — i.e. they
complete the pattern [A][B] ... [A] -> [B]. They are the best-understood
transformer circuit and a major driver of in-context learning.

Diagnostic: feed the model a sequence of RANDOM tokens repeated twice.
Random tokens carry no linguistic structure, so the only way to predict the
second half is to copy from the first. An induction head at position t in the
second half attends to position t - T + 1 (the token AFTER the previous
occurrence of the current token), where T is the period. We score every head
by its average attention along that "induction offset" diagonal.

Usage:
    python induction_heads.py [--model gpt2] [--seq-len 50] [--n-seqs 8]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seq-len", type=int, default=50, help="length T of the random half")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    # "eager" attention is required: the default fused SDPA kernel never
    # materializes the attention matrix, so output_attentions comes back empty.
    model = (
        GPT2LMHeadModel.from_pretrained(args.model, attn_implementation="eager")
        .to(device)
        .eval()
    )
    n_layers, n_heads = model.config.n_layer, model.config.n_head

    T = args.seq_len
    scores = np.zeros((n_layers, n_heads))
    loss_first, loss_second = [], []

    for _ in range(args.n_seqs):
        rand = torch.randint(0, tokenizer.vocab_size, (1, T), device=device)
        ids = torch.cat([rand, rand], dim=1)  # [1, 2T] repeated sequence
        with torch.no_grad():
            out = model(ids, output_attentions=True, labels=ids)

        # Induction offset: query position t (second half) -> key position t-T+1.
        for layer, att in enumerate(out.attentions):  # att: [1, heads, 2T, 2T]
            for t in range(T, 2 * T - 1):
                scores[layer] += att[0, :, t, t - T + 1].cpu().numpy()

        # Sanity check that copying happens: per-token loss, first vs second half.
        logits = out.logits[0, :-1]
        targets = ids[0, 1:]
        tok_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        loss_first.append(tok_loss[: T - 1].mean().item())
        loss_second.append(tok_loss[T:].mean().item())

    scores /= args.n_seqs * (T - 1)

    print(f"loss on first (unpredictable) half : {np.mean(loss_first):.2f}")
    print(f"loss on second (repeated) half     : {np.mean(loss_second):.2f}")
    print("\ntop 10 heads by induction score:")
    flat = [(scores[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    for s, l, h in sorted(flat, reverse=True)[:10]:
        print(f"  L{l}H{h}: {s:.3f}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, cmap="viridis", aspect="auto")
    plt.colorbar(label="mean attention at induction offset")
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.title(f"Induction-head scores for {args.model}")
    plt.tight_layout()
    out_path = "results/induction_scores.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
