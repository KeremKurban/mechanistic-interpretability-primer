# Project 3 — Attention Analysis: Finding Induction Heads

**Question:** *can we identify attention heads with a known algorithmic role?*

## Idea

**Induction heads** implement the pattern-completion rule
`[A][B] … [A] → [B]`: when the current token appeared earlier, attend to the
token that *followed* it last time, and predict that. They were identified in
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
(Elhage et al., 2021) and argued to be a main driver of in-context learning in
[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
(Olsson et al., 2022). They are the closest thing mech interp has to a
"model organism".

## This experiment

Feed GPT-2 a sequence of **random tokens repeated twice** (`[x1..xT x1..xT]`).
Random tokens have no linguistic structure, so the only way to predict the
second half is to copy from the first. An induction head at query position `t`
in the second half attends to key position `t − T + 1`. We score every head by
its mean attention along that diagonal.

Two independent checks:
1. **Behavioral:** per-token loss should collapse on the repeated half
   (the model *is* copying).
2. **Mechanistic:** a few specific heads should light up on the induction
   offset (this is *how* it copies).

## Run it

```bash
python induction_heads.py        # ~1 min on CPU
```

## What you should see

```
loss on first (unpredictable) half : 13.28
loss on second (repeated) half     : 0.29        <- the model copies

top heads by induction score:
  L5H5: 0.920
  L7H10: 0.908
  L5H1: 0.902
  L6H9: 0.897
  L7H2: 0.831
```

plus `results/induction_scores.png` (layer × head heatmap). These specific
heads (L5H1, L5H5, L6H9, L7H2, L7H10) match the induction heads reported for
GPT-2 small in the literature — you just replicated a published finding with
~100 lines of code.

## Implementation gotcha

Recent `transformers` versions default to fused SDPA attention, which never
materializes the attention matrix — `output_attentions` silently comes back
empty. Load the model with `attn_implementation="eager"`.

## Extensions

- Ablate (zero out) the top induction heads and watch the repeated-half loss
  climb — a causal confirmation of the correlational score.
- Look for the *previous-token heads* (the other half of the induction
  circuit) by scoring attention to offset `t − 1`.
- Plot attention patterns of L5H5 on natural text containing repeated names.
