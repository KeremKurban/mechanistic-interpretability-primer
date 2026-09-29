"""Compare L1 and Top-K SAEs along the sparsity / fidelity trade-off.

Trains one SAE per setting (same recipe as train_sae.py) and reports L0
(active features per token), variance explained, dead / dense feature
fractions, and the fraction of LM loss recovered when the reconstruction is
spliced back into GPT-2. Three families:

    L1      ReLU SAE, sparsity tuned through the L1 coefficient
    TopK    Top-K SAE, sparsity set directly by k (as in Gao et al., 2025)
    TopK+aux  Top-K SAE with the AuxK dead-feature loss

Activations are collected once and reused.

Usage:
    python sweep.py [--l1s 0.3 1 3 10] [--ks 8 16 32 64 128 256] [--aux-k 384]

Writes results/sweep.json and results/sweep.png (L0 vs loss recovered);
`python sweep.py --replot` redraws the figure from the JSON without retraining.
"""

import argparse
import json
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
    ap.add_argument("--l1s", type=float, nargs="+", default=[0.3, 1, 3, 10])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--aux-k", type=int, default=384, help="aux features for TopK+aux (d_model/2)")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--dict-size", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", action="store_true", help="only redraw results/sweep.png")
    args = ap.parse_args()
    if args.replot:
        plot(json.load(open("results/sweep.json")), args.layer)
        return

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

    runs = [("L1", f"λ={l1:g}", dict(l1_coeff=l1)) for l1 in args.l1s]
    runs += [("TopK", f"k={k}", dict(l1_coeff=0, k=k)) for k in args.ks]
    runs += [("TopK+aux", f"k={k}", dict(l1_coeff=0, k=k, aux_k=args.aux_k)) for k in args.ks]

    print(f"{'family':>9} {'setting':>7} {'L0':>6} {'var expl':>9} {'dead':>6} {'>10%':>6}"
          f" {'CE w/ SAE':>10} {'loss rec':>9}")
    rows = []
    for family, label, kw in runs:
        torch.manual_seed(args.seed)
        sae = train_sae(acts, args.dict_size, steps=args.steps, batch_size=args.batch_size,
                        device=device, log_every=0, **kw)
        fve, l0 = reconstruction_stats(sae, acts, device)
        with torch.no_grad():
            freq = (sae(acts.to(device))[1] > 0).float().mean(0)
        dead, dense = (freq < 1e-4).float().mean().item(), (freq > 0.1).float().mean().item()
        clean, sae_ce, mean_ce, rec = loss_recovered(
            lm, sae, ids, args.layer, scale, mean_act, ctx_len=128, n_chunks=20, device=device
        )
        rows.append(dict(family=family, setting=label, l0=l0, fve=fve, dead=dead,
                         dense=dense, ce=sae_ce, loss_recovered=rec))
        print(f"{family:>9} {label:>7} {l0:6.1f} {fve:9.1%} {dead:6.1%} {dense:6.1%}"
              f" {sae_ce:10.3f} {rec:9.1%}", flush=True)
    print(f"(clean CE {clean:.3f}, mean-ablated CE {mean_ce:.3f})")
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/sweep.json", "w"), indent=1)
    plot(rows, args.layer)


def plot(rows, layer, y_min=85):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"L1": "#2a78d6", "TopK": "#eb6834", "TopK+aux": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    off_chart = []
    for family, color in colors.items():
        pts = [r for r in rows if r["family"] == family]
        off_chart += [r for r in pts if 100 * r["loss_recovered"] < y_min]
        pts = [r for r in pts if 100 * r["loss_recovered"] >= y_min]
        xs = [r["l0"] for r in pts]
        ys = [100 * r["loss_recovered"] for r in pts]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, label=family,
                markeredgecolor="white", markeredgewidth=1.5)
        # Label settings once: lambdas under the L1 curve, k values over TopK+aux
        # (TopK shares the same x positions).
        if family == "L1":
            for r, x, y in zip(pts, xs, ys):
                ax.annotate(r["setting"], (x, y), textcoords="offset points", xytext=(0, -16),
                            ha="center", fontsize=8, color="#52514e")
        elif family == "TopK+aux":
            for r, x, y in zip(pts, xs, ys):
                ax.annotate(r["setting"], (x, y), textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.set_ylim(y_min, 102.5)
    ax.set_xlabel("L0 (active features per token, log scale)")
    ax.set_ylabel("LM loss recovered (%)")
    ax.set_title(f"L1 vs Top-K SAEs, GPT-2 layer {layer}", loc="left")
    if off_chart:
        note = "; ".join(f"{r['family']} {r['setting']}: L0 {r['l0']:.1f}, "
                         f"{100 * r['loss_recovered']:.0f}%" for r in off_chart)
        ax.text(0.01, 0.02, f"off chart: {note}", transform=ax.transAxes,
                fontsize=8, color="#52514e")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sweep.png", dpi=150)
    print("saved results/sweep.png")

if __name__ == "__main__":
    main()
