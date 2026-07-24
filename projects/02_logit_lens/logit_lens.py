"""The logit lens: decode GPT-2's intermediate residual stream into vocabulary space.

Because transformer blocks read from and write to a shared residual stream, and
the final prediction is just `unembed(layernorm(residual))`, we can apply that
same unembedding to the residual stream after EVERY layer. This shows the
model's "best guess so far" at each depth — its prediction trajectory.

For a prompt, this script prints the top-k next-token candidates at every layer
(for the final position) and plots (a) the probability assigned to the model's
final top token, and (b) the KL divergence between each layer's decoded
distribution and the final one.

Usage:
    python logit_lens.py [--model gpt2] [--prompt "..."] [--topk 5]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEFAULT_PROMPT = "The Eiffel Tower is located in the city of"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()

    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # hidden_states: (n_layers+1) tensors of shape [1, seq, d_model]
    # Decode each with the model's OWN final layernorm + unembedding.
    # NB: HuggingFace applies ln_f to the LAST entry already, so we only apply
    # ln_f ourselves to the intermediate ones (double-normalizing garbles it).
    ln_f, unembed = model.transformer.ln_f, model.lm_head
    layer_logits = []
    with torch.no_grad():
        for hs in out.hidden_states[:-1]:
            layer_logits.append(unembed(ln_f(hs[0, -1])))  # last position only
        layer_logits.append(out.logits[0, -1])
    layer_logits = torch.stack(layer_logits)  # [n_layers+1, vocab]
    layer_logprobs = F.log_softmax(layer_logits, dim=-1)

    final_logprobs = layer_logprobs[-1]
    final_top = final_logprobs.argmax().item()
    print(f"prompt: {args.prompt!r}")
    print(f"final prediction: {tokenizer.decode([final_top])!r}\n")

    print(f"{'layer':>6} | top-{args.topk} tokens (probability)")
    print("-" * 80)
    kls, p_final_top = [], []
    for layer, lp in enumerate(layer_logprobs):
        probs = lp.exp()
        top = probs.topk(args.topk)
        toks = ", ".join(
            f"{tokenizer.decode([i])!r} ({p:.2f})"
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        )
        name = "emb" if layer == 0 else f"{layer}"
        print(f"{name:>6} | {toks}")
        kls.append(F.kl_div(lp, final_logprobs, log_target=True, reduction="sum").item())
        p_final_top.append(probs[final_top].item())

    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    layers = range(len(kls))
    ax1.plot(layers, p_final_top, marker="o")
    ax1.set_xlabel("layer (0 = embeddings)")
    ax1.set_ylabel(f"p({tokenizer.decode([final_top])!r})")
    ax1.set_title("Probability of the final answer, by layer")
    ax2.plot(layers, kls, marker="o", color="tab:red")
    ax2.set_xlabel("layer (0 = embeddings)")
    ax2.set_ylabel("KL(final ‖ layer) in nats")
    ax2.set_title("Distance to the final distribution")
    fig.suptitle(f"Logit lens on {args.model}: {args.prompt!r}")
    fig.tight_layout()
    out_path = "results/logit_lens.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
