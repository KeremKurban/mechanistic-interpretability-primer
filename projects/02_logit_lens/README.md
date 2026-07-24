# Project 2 — The Logit Lens (and Tuned Lens)

**Question:** *what is the model's "best guess so far" at each layer?*

## Idea

A transformer's blocks all read from and write to the same **residual stream**,
and the final prediction is just `unembed(layernorm(residual))`. The
**logit lens** ([nostalgebraist, 2020](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens))
applies that same unembedding to the residual stream after *every* layer,
turning each intermediate state into a distribution over the vocabulary. The
result is a **prediction trajectory**: you watch the answer form layer by
layer.

The **tuned lens** ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112))
fixes the logit lens's main weakness — intermediate layers don't live exactly
in the unembedding's coordinate system — by learning a small affine "translator"
per layer. Same idea, less bias; use the
[`tuned-lens` package](https://github.com/AlignmentResearch/tuned-lens) when
you need reliable trajectories on models where the raw logit lens is noisy.

## Run it

```bash
python logit_lens.py                                # default prompt
python logit_lens.py --prompt "2 + 2 =" --topk 3    # your own prompt
```

## What you should see

For `"The Eiffel Tower is located in the city of"` on GPT-2 small:

```
 layer | top-5 tokens (probability)
   emb | ' destro' (0.67), ' mathemat' (0.24), ...      <- garbage (embeddings)
     1 | ' the' (0.71), ...                             <- generic syntax
     8 | ' Ing' (0.23), ' San' (0.06), ' Rome' (0.05)   <- "it's a city name"
    10 | ' London' (0.25), ' Paris' (0.19), ...         <- candidates compete
    11 | ' Paris' (0.16), ' London' (0.14), ...         <- Paris wins
    12 | ' Paris' (0.06), ...                           <- final (after ln_f)
```

plus `results/logit_lens.png` (probability of the final answer by layer, and
KL divergence of each layer's distribution to the final one). The typical
picture: early layers do syntax, the semantic category appears mid-network,
and the specific fact is resolved in the last third.

## Implementation gotcha (learned the hard way)

HuggingFace's `output_hidden_states` returns `n_layers + 1` tensors, and the
**last one already has the final layernorm applied**. Applying `ln_f` twice
garbles the final layer's decoding — the script decodes intermediate layers
with `ln_f` + unembedding but takes the last layer's logits directly.

## Extensions

- Decode *every* position, not just the last, and plot a layer × position
  heatmap of top-1 tokens.
- Compare with the tuned lens on the same prompts.
- Look at prompts where early layers confidently predict something the final
  layers overturn — this is the basis of "early exit" and self-correction
  analyses.
