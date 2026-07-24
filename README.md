# Mechanistic Interpretability Primer

A self-contained introduction to mechanistic interpretability of transformer
language models: a [review-style document](REVIEW.md) plus **five minimal,
runnable projects**, each teaching one core technique by reproducing a known
result on GPT-2 small. Everything runs on CPU in minutes, using plain
PyTorch + HuggingFace (no interpretability libraries), so you see every moving
part.

## Start here

1. Read **[REVIEW.md](REVIEW.md)** — the primer document (with links to the
   proper published surveys at the top).
2. Run the projects in order; each folder has a README with the idea, expected
   output, pitfalls, and extension ideas.

| Project | Technique | Question | Reproduced result |
|---|---|---|---|
| [`01_linear_probing`](projects/01_linear_probing/) | Linear probes | What is linearly readable at each layer? | Contextual POS disambiguation emerges after layer 0 (+ a live probing confound) |
| [`02_logit_lens`](projects/02_logit_lens/) | Logit lens | What does the model "believe" at each layer? | "Eiffel Tower → Paris" resolves in the last third of the network |
| [`03_induction_heads`](projects/03_induction_heads/) | Attention analysis | Which heads implement a known algorithm? | GPT-2 small's published induction heads (L5H1, L5H5, L6H9, L7H2, L7H10) |
| [`04_activation_patching`](projects/04_activation_patching/) | Activation patching | Which sites causally carry information? | ROME-style causal tracing: subject token → final position hand-off |
| [`05_sparse_autoencoder`](projects/05_sparse_autoencoder/) | Sparse autoencoders | What features compose an activation? | Trains a small SAE and surfaces (somewhat) interpretable features |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CPU torch is fine
cd projects/01_linear_probing && python probe_pos.py
```

First run downloads GPT-2 small (~500 MB) from HuggingFace.

## If you want the "real" version of this repo

These existing resources cover the same ground more deeply — worth checking
before customizing this one:

- **[ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0)** — the standard
  structured curriculum with exercises + solutions (mech interp is chapter 1).
- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)**
  — the library everyone uses for GPT-2-scale interp research;
  [intro exercises here](https://github.com/callummcdougall/TransformerLens-intro).
- **[nnsight](https://github.com/ndif-team/nnsight)** — interventions on
  bigger/remote models.
- **[SAELens](https://github.com/jbloomAus/SAELens)** +
  [Neuronpedia](https://www.neuronpedia.org/) — production SAE training and
  browsable pre-trained features.
- **[tuned-lens](https://github.com/AlignmentResearch/tuned-lens)** — the
  maintained logit-lens successor.
- **[Awesome mechanistic interpretability](https://github.com/apartresearch/mechanisticinterpretability)**
  — curated paper/resource list.

The deliberate difference here: no framework magic — every hook, cache, and
patch is visible in ~100-line scripts.
