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
activations, trains the SAE on CPU in a few minutes, reports how much of the
LM loss survives splicing the reconstruction back in, then prints the
top-activating tokens (with context) for a sample of features.

Top-K variant (Gao et al., 2025, "Scaling and evaluating sparse autoencoders"):
    f      = TopK(ReLU(W_enc (x - b_dec) + b_enc))   # keep the k largest, zero the rest
    loss   = ||x - x_hat||^2                          # no L1: sparsity is fixed at k
Sparsity is set directly instead of tuned through l1, and there is no L1
shrinkage of the surviving activations. --aux-k adds Gao et al.'s auxiliary
loss against dead features.

Usage:
    python train_sae.py [--steps 2000] [--dict-size 3072] [--l1 1.0]
    python train_sae.py --k 64 [--aux-k 384]            # Top-K SAE

Raising --l1 trades reconstruction fidelity for sparsity (fewer active
features per token); the default gives L0~75 with ~99% of LM loss recovered.
See sweep.py for the full trade-off curve of both variants.
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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class SparseAutoencoder(nn.Module):
    """ReLU SAE (trained with an L1 penalty), or Top-K SAE when k is given."""

    def __init__(self, d_model, dict_size, k=None):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)
            self.W_enc.copy_(self.W_dec.t())

    def pre_acts(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def forward(self, x):
        f = self.pre_acts(x)
        if self.k is not None:
            vals, idx = f.topk(self.k, dim=-1)
            f = torch.zeros_like(f).scatter_(-1, idx, vals)
        x_hat = f @ self.W_dec + self.b_dec
        return x_hat, f

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)


def collect_activations(model, tokenizer, text, layer, max_tokens, ctx_len, device):
    """Returns activations [N, d_model], the token ids [M] of all chunks, and
    for each activation its index into those token ids.

    Position 0 of every chunk is skipped: in GPT-2 the first token acts as an
    "attention sink" with a residual norm ~40x larger than any other position.
    Kept in, those <1% of tokens carry >90% of the squared norm and dominate
    both the scale normalization and the reconstruction loss.
    """
    ids = tokenizer(text, return_tensors="pt").input_ids[0, :max_tokens]
    acts, toks, tok_idx = [], [], []
    with torch.no_grad():
        for i in range(0, len(ids) - ctx_len, ctx_len):
            chunk = ids[i : i + ctx_len].unsqueeze(0).to(device)
            hs = model(chunk, output_hidden_states=True).hidden_states
            acts.append(hs[layer][0, 1:].cpu())
            toks.append(chunk[0].cpu())
            tok_idx.append(torch.arange(i + 1, i + ctx_len))
    return torch.cat(acts), torch.cat(toks), torch.cat(tok_idx)


def train_sae(acts, dict_size, l1_coeff, steps, batch_size, device, log_every=200,
              k=None, aux_k=0, dead_steps=50, aux_coeff=1 / 32):
    """L1 SAE if k is None, else Top-K SAE (l1_coeff is then ignored).

    aux_k > 0 enables the AuxK loss from Gao et al. (2025): features that have
    not fired for `dead_steps` batches are asked to reconstruct the main SAE's
    residual error from their top-aux_k pre-activations. That gives dead
    features a gradient again and revives many of them.
    """
    sae = SparseAutoencoder(acts.shape[1], dict_size, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    since_fired = torch.zeros(dict_size, device=device)
    for step in range(steps):
        idx = torch.randint(0, acts.shape[0], (batch_size,))
        x = acts[idx].to(device)
        x_hat, f = sae(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        loss = mse if k is not None else mse + l1_coeff * f.sum(-1).mean()
        since_fired = torch.where((f > 0).any(0), 0.0, since_fired + 1)
        dead = since_fired > dead_steps
        if aux_k and dead.any():
            err = (x - x_hat).detach()
            pre = sae.pre_acts(x) * dead
            vals, i = pre.topk(min(aux_k, int(dead.sum())), dim=-1)
            err_hat = torch.zeros_like(pre).scatter_(-1, i, vals) @ sae.W_dec
            loss = loss + aux_coeff * (err - err_hat).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if log_every and (step % log_every == 0 or step == steps - 1):
            l0 = (f > 0).float().sum(-1).mean().item()
            print(f"step {step:5d}  mse {mse.item():8.3f}  L0 (active feats) {l0:7.1f}")
    return sae


@torch.no_grad()
def reconstruction_stats(sae, acts, device):
    """Fraction of variance explained and mean L0 over the whole dataset."""
    x = acts.to(device)
    x_hat, f = sae(x)
    fve = 1 - (x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
    return fve.item(), (f > 0).float().sum(-1).mean().item()


@torch.no_grad()
def loss_recovered(lm, sae, ids, layer, scale, mean_act, ctx_len, n_chunks, device):
    """Splice the SAE reconstruction into the residual stream and compare LM loss.

    Returns (clean CE, SAE CE, mean-ablation CE, fraction of loss recovered),
    where 0% = as bad as replacing the layer with its mean activation and
    100% = as good as the original model. Position 0 is left untouched, since
    the SAE was not trained on it. This is the metric that matters: low MSE
    means little if the model can't use the reconstruction.
    """

    def run(mode):
        def hook(_, __, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            if mode == "sae":
                h[:, 1:] = sae(h[:, 1:] / scale)[0] * scale
            elif mode == "mean":
                h[:, 1:] = mean_act.to(device)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handle = lm.transformer.h[layer - 1].register_forward_hook(hook) if mode != "clean" else None
        losses = []
        for i in range(n_chunks):
            chunk = ids[i * ctx_len : (i + 1) * ctx_len].unsqueeze(0).to(device)
            losses.append(lm(chunk, labels=chunk).loss.item())
        if handle:
            handle.remove()
        return float(np.mean(losses))

    clean, sae_ce, mean_ce = run("clean"), run("sae"), run("mean")
    return clean, sae_ce, mean_ce, (mean_ce - sae_ce) / (mean_ce - clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6, help="residual stream after block N")
    ap.add_argument("--dict-size", type=int, default=3072)
    ap.add_argument("--l1", type=float, default=1.0, help="L1 coefficient (L1 SAE only)")
    ap.add_argument("--k", type=int, default=None,
                    help="train a Top-K SAE with exactly k active features per token")
    ap.add_argument("--aux-k", type=int, default=0,
                    help="Top-K only: AuxK dead-feature loss with this many aux features (0 = off)")
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
    lm = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    d_model = lm.config.n_embd

    print("collecting activations...")
    acts, toks, tok_idx = collect_activations(
        lm.transformer, tokenizer, text, args.layer, args.max_tokens, ctx_len=128, device=device
    )
    print(f"{acts.shape[0]} activation vectors from layer {args.layer}")

    # Normalize overall scale so the l1 coefficient is transferable-ish.
    scale = acts.norm(dim=1).mean() / np.sqrt(d_model)
    mean_act = acts.mean(0)
    acts = acts / scale

    print("training SAE...")
    sae = train_sae(acts, args.dict_size, args.l1, args.steps, args.batch_size, device,
                    k=args.k, aux_k=args.aux_k)

    fve, l0 = reconstruction_stats(sae, acts, device)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    clean, sae_ce, mean_ce, rec = loss_recovered(
        lm, sae, ids, args.layer, scale, mean_act, ctx_len=128, n_chunks=20, device=device
    )
    print(f"\nL0 {l0:.1f}   variance explained {fve:.1%}")
    print(f"LM loss: clean {clean:.3f}  with SAE {sae_ce:.3f}  mean-ablated {mean_ce:.3f}"
          f"  -> {rec:.1%} of loss recovered")

    # ---- inspect features ---------------------------------------------------
    with torch.no_grad():
        _, f_all = sae(acts.to(device))
    freq = (f_all > 0).float().mean(0)
    print(f"dead features (fire on <0.01% of tokens): {(freq < 1e-4).float().mean():.1%}")
    alive = ((freq > 1e-4) & (freq < 0.3)).nonzero().flatten()
    print(f"\n{len(alive)} features fire on 0.01%-30% of tokens; sampling 10:")

    rng = np.random.RandomState(args.seed)
    for feat in rng.choice(alive.cpu().numpy(), size=min(10, len(alive)), replace=False):
        top = f_all[:, feat].topk(8).indices
        examples = []
        for i in top.tolist():
            j = tok_idx[i].item()
            ctx = tokenizer.decode(toks[max(0, j - 6) : j])
            tok = tokenizer.decode([toks[j]])
            examples.append(f"...{ctx}[[{tok}]]")
        print(f"\nfeature {feat} (fires on {freq[feat].item():.1%} of tokens):")
        for e in examples[:5]:
            print("   ", e.replace("\n", "\\n"))

    os.makedirs("results", exist_ok=True)
    torch.save(sae.state_dict(), "results/sae.pt")
    print("\nsaved results/sae.pt")


if __name__ == "__main__":
    main()
