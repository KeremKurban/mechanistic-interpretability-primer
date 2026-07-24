# Project 1 — Linear Probing

**Question:** *what information is linearly readable from each layer's activations?*

## Idea

Freeze the model, extract hidden activations at some layer, and train a simple
linear classifier ("probe") to predict a property of interest from them
([Alain & Bengio, 2016](https://arxiv.org/abs/1610.01644)). If a linear probe
succeeds, the property is *linearly encoded* at that layer. Comparing probe
accuracy across layers shows **where** in the network the information emerges.

## This experiment

We probe GPT-2 small's residual stream for a **noun-vs-verb** distinction using
only words that are ambiguous between the two ("run", "walk", "dance", …),
placed in disambiguating contexts:

- noun: *"That was a great **run**."*
- verb: *"They **run** every single day."*

Because the token is identical in both classes, its static embedding carries no
label information — any probe accuracy above chance must come from **contextual
processing** done by the transformer layers. The train/test split is done *by
word*, so the probe is evaluated on words it never saw during training.

## Run it

```bash
python probe_pos.py            # ~1 min on CPU
```

## What you should see

```
layer  0 (embeddings): test accuracy = 0.643
layer  1 (   block 1): test accuracy = 1.000
layer  2 (   block 2): test accuracy = 1.000
...
layer 12 (  block 12): test accuracy = 1.000
```

plus `results/probe_accuracy_by_layer.png`. Embeddings are near chance;
a single transformer block already suffices to disambiguate — local syntactic
context (the preceding article vs. pronoun) is easy to integrate.

## Pitfalls worth noticing (this is most of what the probing literature is about)

1. **Layer 0 is *above* chance (0.64, not 0.50).** Why? `hidden_states[0]`
   includes *positional* embeddings, and our noun/verb templates place the
   target word at slightly different positions. The probe partially exploits
   this confound. Real probing papers fight exactly this kind of leakage with
   *control tasks* ([Hewitt & Liang, 2019](https://arxiv.org/abs/1909.03368)).
2. **Correlation ≠ use.** A successful probe shows information is *present*,
   not that the model *uses* it downstream. That requires causal methods
   (see project 4). See [Belinkov, 2022](https://arxiv.org/abs/2102.12452) for
   a survey of probing and its limitations.
3. **Probe capacity.** With high-dimensional activations and few examples, even
   a linear probe can memorize; that's why we regularize (`C=0.1`) and split by
   word.

## Extensions

- Replace the task with sentiment, tense, or subject–verb number agreement.
- Add a *selectivity* control: train the same probe with shuffled labels.
- Try `--model gpt2-medium` and compare where information emerges.
