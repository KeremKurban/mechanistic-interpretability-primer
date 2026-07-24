"""Train a small sparse autoencoder (SAE) on GPT-2 residual-stream activations.

Individual neurons are often *polysemantic* (they fire for many unrelated
things), plausibly because the model compresses more features than it has
dimensions ("superposition"). An SAE tries to undo this compression: it learns
an overcomplete dictionary of directions such that every activation vector is
a SPARSE non-negative combination of them. The hope is that individual
dictionary elements ("features") are more interpretable than neurons.

Architecture (following Anthropic's "Towards Monosemanticity", 2023):
    f      = ReLU(W_enc (x - b_dec) + b_enc)      # sparse feature activations
    x_hat  = W_dec f + b_dec                      # reconstruction
    loss   = ||x - x_hat||^2 + l1 * sum(f)        # fidelity + sparsity
Decoder columns are renormalized to unit norm every step, otherwise the model
cheats the L1 penalty by shrinking f and growing W_dec.

The script downloads tiny-shakespeare (~1 MB), collects layer-6 residual
activations, trains the SAE on CPU in a few minutes, then prints the
top-activating tokens (with context) for a sample of features.

Usage:
    python train_sae.py [--steps 2000] [--dict-size 3072] [--l1 0.2]

Raising --l1 trades reconstruction fidelity for sparsity (fewer active
features per token); the default gives roughly L0~500 at ~2.5% MSE on CPU.
Real SAE work uses far more data, bigger dictionaries, and tricks for dead
features — see SAELens (https://github.com/jbloomAus/SAELens) for the
production version of this.
"""

import argparse
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2TokenizerFast

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)
            self.W_enc.copy_(self.W_dec.t())

    def forward(self, x):
        f = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        x_hat = f @ self.W_dec + self.b_dec
        return x_hat, f

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)


def collect_activations(model, tokenizer, text, layer, max_tokens, ctx_len, device):
    """Returns activations [N, d_model] and the token ids [N] they came from."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0, :max_tokens]
    acts, toks = [], []
    with torch.no_grad():
        for i in range(0, len(ids) - ctx_len, ctx_len):
            chunk = ids[i : i + ctx_len].unsqueeze(0).to(device)
            hs = model(chunk, output_hidden_states=True).hidden_states
            acts.append(hs[layer][0].cpu())
            toks.append(chunk[0].cpu())
    return torch.cat(acts), torch.cat(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6, help="residual stream after block N")
    ap.add_argument("--dict-size", type=int, default=3072)
    ap.add_argument("--l1", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("tinyshakespeare.txt"):
        print("downloading tiny-shakespeare...")
        urllib.request.urlretrieve(DATA_URL, "tinyshakespeare.txt")
    text = open("tinyshakespeare.txt").read()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    lm = GPT2Model.from_pretrained(args.model).to(device).eval()
    d_model = lm.config.n_embd

    print("collecting activations...")
    acts, toks = collect_activations(
        lm, tokenizer, text, args.layer, args.max_tokens, ctx_len=128, device=device
    )
    print(f"{acts.shape[0]} activation vectors from layer {args.layer}")

    # Normalize overall scale so the l1 coefficient is transferable-ish.
    scale = acts.norm(dim=1).mean() / np.sqrt(d_model)
    acts = acts / scale

    sae = SparseAutoencoder(d_model, args.dict_size).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

    print("training SAE...")
    for step in range(args.steps):
        idx = torch.randint(0, acts.shape[0], (args.batch_size,))
        x = acts[idx].to(device)
        x_hat, f = sae(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        l1 = f.sum(-1).mean()
        loss = mse + args.l1 * l1
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % 200 == 0 or step == args.steps - 1:
            l0 = (f > 0).float().sum(-1).mean().item()
            print(f"step {step:5d}  mse {mse.item():8.3f}  L0 (active feats) {l0:7.1f}")

    # ---- inspect features ---------------------------------------------------
    with torch.no_grad():
        _, f_all = sae(acts.to(device))
    freq = (f_all > 0).float().mean(0)
    alive = ((freq > 1e-4) & (freq < 0.3)).nonzero().flatten()
    print(f"\n{len(alive)} features fire on 0.01%-30% of tokens; sampling 10:")

    rng = np.random.RandomState(args.seed)
    for feat in rng.choice(alive.cpu().numpy(), size=min(10, len(alive)), replace=False):
        top = f_all[:, feat].topk(8).indices
        examples = []
        for i in top.tolist():
            ctx = tokenizer.decode(toks[max(0, i - 6) : i])
            tok = tokenizer.decode([toks[i]])
            examples.append(f"...{ctx}[[{tok}]]")
        print(f"\nfeature {feat} (fires on {freq[feat].item():.1%} of tokens):")
        for e in examples[:5]:
            print("   ", e.replace("\n", "\\n"))

    os.makedirs("results", exist_ok=True)
    torch.save(sae.state_dict(), "results/sae.pt")
    print("\nsaved results/sae.pt")


if __name__ == "__main__":
    main()
