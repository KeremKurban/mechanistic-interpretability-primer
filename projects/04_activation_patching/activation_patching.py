"""Activation patching (causal tracing) on GPT-2 with plain PyTorch hooks.

Probing and the logit lens are *correlational*: they tell you what information
is present, not what the model actually uses. Activation patching is the basic
*causal* tool. Recipe:

  1. Run a CLEAN prompt   ("The capital of France is" -> Paris) and cache
     the residual-stream activation after every layer.
  2. Run a CORRUPTED prompt ("The capital of Italy is" -> Rome), but overwrite
     ("patch") the activation at ONE (layer, position) with the clean value.
  3. Measure how much of the clean behaviour is restored, via the logit
     difference logit(Paris) - logit(Rome) at the final position.

If patching a single site restores the clean answer, the causally relevant
information flows through that site. The two prompts must tokenize to the same
length so positions line up.

Usage:
    python activation_patching.py [--model gpt2]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CLEAN_PROMPT = "The capital of France is"
CORRUPT_PROMPT = "The capital of Italy is"
CLEAN_ANSWER = " Paris"
CORRUPT_ANSWER = " Rome"


def single_token_id(tokenizer, s):
    ids = tokenizer(s).input_ids
    assert len(ids) == 1, f"{s!r} is not a single token: {ids}"
    return ids[0]


def logit_diff(logits, a_id, b_id):
    """logit(a) - logit(b) at the final position."""
    return (logits[0, -1, a_id] - logits[0, -1, b_id]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    n_layers = model.config.n_layer

    clean_ids = tokenizer(CLEAN_PROMPT, return_tensors="pt").input_ids.to(device)
    corr_ids = tokenizer(CORRUPT_PROMPT, return_tensors="pt").input_ids.to(device)
    assert clean_ids.shape == corr_ids.shape, "prompts must have equal token length"
    seq_len = clean_ids.shape[1]
    a_id = single_token_id(tokenizer, CLEAN_ANSWER)
    b_id = single_token_id(tokenizer, CORRUPT_ANSWER)

    # --- 1. clean run: cache residual stream after every block ---------------
    with torch.no_grad():
        clean_out = model(clean_ids, output_hidden_states=True)
    # hidden_states[i+1] is the residual stream after block i (pre-ln_f except last)
    clean_resid = [hs.clone() for hs in clean_out.hidden_states]
    clean_diff = logit_diff(clean_out.logits, a_id, b_id)

    # --- 2. corrupted baseline ----------------------------------------------
    with torch.no_grad():
        corr_out = model(corr_ids)
    corr_diff = logit_diff(corr_out.logits, a_id, b_id)

    print(f"clean     logit(Paris)-logit(Rome): {clean_diff:+.2f}")
    print(f"corrupted logit(Paris)-logit(Rome): {corr_diff:+.2f}")

    # --- 3. patch each (layer, position) ------------------------------------
    def make_hook(pos, clean_act):
        def hook(module, inputs, output):
            # The block's output is the residual stream [batch, seq, d_model].
            # Older transformers versions wrap it in a tuple; handle both.
            resid = output[0] if isinstance(output, tuple) else output
            patched = resid.clone()
            patched[0, pos] = clean_act[0, pos]
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return hook

    recovery = np.zeros((n_layers, seq_len))
    for layer in range(n_layers):
        for pos in range(seq_len):
            handle = model.transformer.h[layer].register_forward_hook(
                make_hook(pos, clean_resid[layer + 1])
            )
            with torch.no_grad():
                patched_out = model(corr_ids)
            handle.remove()
            d = logit_diff(patched_out.logits, a_id, b_id)
            recovery[layer, pos] = (d - corr_diff) / (clean_diff - corr_diff)

    tokens = [tokenizer.decode([t]) for t in corr_ids[0].tolist()]
    print("\nrecovery of clean behaviour (1.0 = fully restored):")
    header = " ".join(f"{t.strip():>8s}" for t in tokens)
    print(f"{'layer':>6} {header}")
    for layer in range(n_layers):
        row = " ".join(f"{recovery[layer, p]:8.2f}" for p in range(seq_len))
        print(f"{layer:>6} {row}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(recovery, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="fraction of clean logit-diff restored")
    plt.xticks(range(seq_len), tokens, rotation=45)
    plt.ylabel("layer patched")
    plt.title(f"Residual-stream patching, {args.model}\n{CORRUPT_PROMPT!r} <- clean France run")
    plt.tight_layout()
    out_path = "results/patching_heatmap.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
