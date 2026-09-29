"""Sweep the L1 coefficient and trace the sparsity / fidelity trade-off.

For each --l1 value, trains an SAE (same recipe as train_sae.py) and reports
L0 (active features per token), variance explained, dead / dense feature
counts, and the fraction of LM loss recovered when the reconstruction is
spliced back into GPT-2. Activations are collected once and reused.

Usage:
    python sweep_l1.py [--l1s 0.2 1 3 10 30 100] [--steps 2000]

Writes results/sweep_l1.png (L0 vs loss recovered).
"""

import argparse
import os

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from train_sae import (
    DATA_URL,
    collect_activations,
    loss_recovered,
    reconstruction_stats,
    train_sae,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1s", type=float, nargs="+", default=[0.2, 1, 3, 10, 30, 100])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--dict-size", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists("tinyshakespeare.txt"):
        import urllib.request

        urllib.request.urlretrieve(DATA_URL, "tinyshakespeare.txt")
    text = open("tinyshakespeare.txt").read()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    acts, _, _ = collect_activations(
        lm.transformer, tokenizer, text, args.layer, args.max_tokens, ctx_len=128, device=device
    )
    scale = acts.norm(dim=1).mean() / np.sqrt(acts.shape[1])
    mean_act = acts.mean(0)
    acts = acts / scale
    ids = tokenizer(text, return_tensors="pt").input_ids[0]

    print(f"{'l1':>6} {'L0':>7} {'var expl':>9} {'dead':>6} {'>10%':>6} {'CE w/ SAE':>10} {'loss rec':>9}")
    rows = []
    for l1 in args.l1s:
        torch.manual_seed(args.seed)
        sae = train_sae(acts, args.dict_size, l1, args.steps, args.batch_size, device, log_every=0)
        fve, l0 = reconstruction_stats(sae, acts, device)
        with torch.no_grad():
            freq = (sae(acts.to(device))[1] > 0).float().mean(0)
        dead, dense = (freq < 1e-4).sum().item(), (freq > 0.1).sum().item()
        clean, sae_ce, mean_ce, rec = loss_recovered(
            lm, sae, ids, args.layer, scale, mean_act, ctx_len=128, n_chunks=20, device=device
        )
        rows.append((l1, l0, rec))
        print(f"{l1:6g} {l0:7.1f} {fve:9.1%} {dead:6d} {dense:6d} {sae_ce:10.3f} {rec:9.1%}", flush=True)
    print(f"(clean CE {clean:.3f}, mean-ablated CE {mean_ce:.3f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    l1s, l0s, recs = zip(*rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(l0s, [100 * r for r in recs], color="#2a78d6", lw=2, marker="o", ms=8)
    for l1, l0, r in rows:
        ax.annotate(f"λ={l1:g}", (l0, 100 * r), textcoords="offset points", xytext=(6, -12),
                    fontsize=9, color="#52514e")
    ax.set_xscale("log")
    ax.set_xlabel("L0 (active features per token, log scale)")
    ax.set_ylabel("LM loss recovered (%)")
    ax.set_title(f"SAE sparsity vs fidelity, GPT-2 layer {args.layer}", loc="left")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sweep_l1.png", dpi=150)
    print("saved results/sweep_l1.png")


if __name__ == "__main__":
    main()
