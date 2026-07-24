# Project 5 — Sparse Autoencoders (Dictionary Learning)

**Question:** *can we decompose activations into interpretable features?*

## Idea

Individual neurons are often **polysemantic** — they fire in many unrelated
contexts — plausibly because models represent more features than they have
dimensions, storing them in **superposition**
([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html)).
A **sparse autoencoder (SAE)** tries to undo the compression: it learns an
overcomplete dictionary of directions such that each activation vector is a
sparse non-negative combination of them. Individual dictionary elements
("features") tend to be far more interpretable than neurons
([Cunningham et al., 2023](https://arxiv.org/abs/2309.08600);
[Bricken et al., 2023 — *Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html)),
and the approach scales to frontier models
([Templeton et al., 2024 — *Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)).

## Architecture (the standard 2023 recipe)

```
f     = ReLU(W_enc (x − b_dec) + b_enc)     # sparse feature activations
x_hat = W_dec f + b_dec                     # reconstruction
loss  = ‖x − x_hat‖² + λ·Σf                 # fidelity + L1 sparsity
```

Decoder rows are renormalized to unit norm every step — otherwise the model
cheats the L1 penalty by shrinking `f` and growing `W_dec`.

## Run it

```bash
python train_sae.py                  # ~3-4 min on CPU
python train_sae.py --l1 0.5        # sparser, worse reconstruction
```

Downloads tiny-shakespeare (~1 MB), collects ~60k layer-6 residual-stream
vectors from GPT-2 small, trains a 768→3072 SAE, then prints top-activating
contexts for a sample of features.

## What you should see

```
step     0  mse 7040.042  L0 (active feats)  1545.5
step   999  mse   18.991  L0 (active feats)   491.9
```

and per-feature summaries like:

```
feature 2834 (fires on 0.1% of tokens):
    ...ET:\nO princely[[ Buckingham]]
    ...: all the swords\nIn[[ Italy]]
```

Skim the sampled features: some will have an obvious theme (proper names,
punctuation, a specific word family), others won't. That mixed picture is
honest — feature quality depends heavily on data volume, dictionary size, and
the sparsity/fidelity trade-off.

## Honest caveats

- This toy run uses ~60k tokens; real SAEs train on **billions**. L0≈500 is
  far from the L0≈20–100 of production SAEs. Expect suggestive, not clean,
  features.
- Known open problems: dead features, feature splitting, and whether SAE
  features are the model's "real" units at all. See
  [SAELens](https://github.com/jbloomAus/SAELens) and
  [Neuronpedia](https://www.neuronpedia.org/) for tooling and browsable
  pre-trained SAEs, and [Gemma Scope](https://arxiv.org/abs/2408.05147) for a
  full open-source SAE suite.

## Extensions

- Increase `--l1` and `--steps`; plot the sparsity/MSE Pareto frontier.
- Train on MLP activations instead of the residual stream.
- Pick one interpretable feature and *steer*: add its decoder direction to the
  residual stream during generation and watch the output change.
