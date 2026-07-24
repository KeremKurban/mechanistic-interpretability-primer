# Project 4 — Activation Patching (Causal Tracing)

**Question:** *which internal activations causally carry the information?*

## Idea

Probing (project 1) and the logit lens (project 2) are **correlational**.
Activation patching is the field's basic **causal** tool, introduced as
"causal tracing" in [ROME (Meng et al., 2022)](https://arxiv.org/abs/2202.05262)
and used to find circuits in
[Interpretability in the Wild (Wang et al., 2022)](https://arxiv.org/abs/2211.00593):

1. Run a **clean** prompt (`"The capital of France is"` → Paris) and cache all
   residual-stream activations.
2. Run a **corrupted** prompt (`"The capital of Italy is"` → Rome), but
   overwrite the activation at *one* (layer, position) with the cached clean
   value.
3. Measure how much of the clean behavior returns, using the
   **logit difference** `logit(Paris) − logit(Rome)` — the recommended metric
   per [Zhang & Nanda, 2024](https://arxiv.org/abs/2309.16042).

If patching one site restores the clean answer, the France-information
causally flows through that site. (The two prompts must tokenize to the same
length so positions align.)

## Run it

```bash
python activation_patching.py        # ~1 min on CPU (12 layers x 5 positions)
```

## What you should see

```
recovery of clean behaviour (1.0 = fully restored):
 layer      The  capital       of    Italy       is
     0     0.00     0.00     0.00     1.00     0.00
     ...
     8     0.00     0.00     0.00     0.89     0.10
     9     0.00     0.00     0.00     0.34     0.60
    10     0.00     0.00     0.00     0.06     0.93
    11     0.00     0.00     0.00     0.00     0.79
```

plus `results/patching_heatmap.png`. This is the classic causal-tracing
picture: the country identity lives at the **subject token** ("Italy") through
layer ~8, then attention moves it to the **final position** in layers 9–11,
where it determines the prediction. You've localized an information route with
a for-loop and a forward hook.

## Design notes

- **Denoising vs. noising:** we patch clean→corrupted ("how much does this
  site *restore*?"). The reverse direction answers a subtly different question
  ("how much does this site *break*?") — see
  [Heimersheim & Nanda, 2024](https://arxiv.org/abs/2404.15255).
- **Metric choice matters:** probabilities can hide effects behind softmax
  saturation; logit differences behave more linearly.
- The hook handles both `transformers` ≥5 (block returns a tensor) and older
  versions (block returns a tuple).

## Extensions

- Patch **attention outputs** and **MLP outputs** separately to see which
  component moves the information at each layer.
- Try the Indirect Object Identification setup ("When John and Mary went to
  the store, John gave a drink to") and reproduce a piece of the IOI circuit.
- Scale to many prompt pairs and average — single-pair results can be
  idiosyncratic.
