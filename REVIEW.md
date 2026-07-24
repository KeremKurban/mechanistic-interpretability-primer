# Mechanistic Interpretability of Transformer Language Models: A Hands-On Primer

*A review-style primer with five runnable companion projects (this repository).*

> **Read the real reviews first.** This document is a working primer, not a
> substitute for the peer-reviewed surveys:
>
> | Reference | Scope |
> |---|---|
> | Rai et al., 2024 — [A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models](https://arxiv.org/abs/2407.02646) | The best newcomer-oriented technical review; organizes the field by object of study (features, circuits) and technique. |
> | Bereska & Gavves, 2024 — [Mechanistic Interpretability for AI Safety — A Review](https://arxiv.org/abs/2404.14082) | Same techniques, framed by their relevance to AI safety. |
> | Ferrando et al., 2024 — [A Primer on the Inner Workings of Transformer-Based Language Models](https://arxiv.org/abs/2405.00208) | Deep, math-forward primer on transformer internals and analysis methods. |
> | Belinkov, 2022 — [Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452) | Focused survey of the probing literature and its pitfalls. |
>
> **Similar hands-on repositories** (check these before customizing this repo — they are more complete):
> [ARENA 3.0 curriculum](https://github.com/callummcdougall/ARENA_3.0) ·
> [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) ·
> [TransformerLens intro exercises](https://github.com/callummcdougall/TransformerLens-intro) ·
> [nnsight](https://github.com/ndif-team/nnsight) ·
> [SAELens](https://github.com/jbloomAus/SAELens) ·
> [tuned-lens](https://github.com/AlignmentResearch/tuned-lens) ·
> [Awesome mech-interp list (Apart Research)](https://github.com/apartresearch/mechanisticinterpretability)

---

## Abstract

Mechanistic interpretability (MI) aims to reverse-engineer the computations
that trained neural networks actually perform, rather than merely correlating
inputs with outputs. This primer reviews the core technique families for
transformer language models — **linear probing**, **logit-lens decoding**,
**attention-pattern analysis**, **activation patching**, and **sparse
autoencoders** — ordered roughly from correlational to causal to
decompositional. For each technique we state the question it answers, its
assumptions, its known failure modes, and provide a minimal runnable
experiment on GPT-2 small (`projects/01`–`05`) that reproduces a known result
on a CPU in minutes. We close with cross-cutting limitations and open
problems. The intended reader is a researcher entering the field who wants
both a conceptual map and working code.

## 1. Introduction

Large language models perform tasks their designers never explicitly
programmed. MI takes the position that the learned parameters implement
*mechanisms* — algorithms distributed across attention heads, MLP neurons, and
the residual stream — and that these mechanisms can be discovered, named, and
causally verified. This differs from classical explainable-AI methods
(saliency maps, attributions), which explain *outputs*; MI explains
*computation*.

Why bother? Three standard motivations:

1. **Science** — models are the most capable artifacts we don't understand;
   understanding them is intrinsically interesting.
2. **Safety** — behavioral evaluation cannot distinguish "does the right
   thing" from "does the right thing *while being evaluated*"; auditing
   internals might ([Bereska & Gavves, 2024](https://arxiv.org/abs/2404.14082)).
3. **Engineering** — locating a mechanism enables targeted model editing
   ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)), steering, and
   debugging.

### 1.1 The residual-stream view (essential background)

Nearly every technique below relies on one architectural fact: a transformer
layer does not transform its input wholesale. Each attention head and MLP
*reads* from a shared **residual stream**, computes an update, and *adds* it
back. The stream is therefore a running sum of contributions, and the final
prediction is `unembed(layernorm(residual))`. Consequences:

- Intermediate states live (approximately) in one shared vector space, so a
  fixed decoder (the unembedding) can be applied at any depth (§3).
- Components can be studied additively — each head/MLP's contribution to the
  logits can be isolated ("direct logit attribution").
- Directions in the stream, not neurons, are the natural candidate unit of
  meaning — the **linear representation hypothesis**, supported by classic
  vector arithmetic ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781))
  and by interventions on learned world models
  ([Li et al., 2023 — Othello-GPT](https://arxiv.org/abs/2210.13382);
  [Nanda et al., 2023](https://arxiv.org/abs/2309.00941)).

This framing comes from
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
(Elhage et al., 2021), the field's foundational document, building on the
original circuits work in CNNs
([Olah et al., 2020](https://distill.pub/2020/circuits/zoom-in/)).

### 1.2 A taxonomy of techniques

| Technique | Question answered | Evidence type | Project |
|---|---|---|---|
| Linear probing | Is property X linearly present at layer L? | Correlational | `01` |
| Logit/tuned lens | What does the model "believe" at layer L? | Correlational | `02` |
| Attention analysis | What algorithm does head H implement? | Correlational | `03` |
| Activation patching | Which sites causally carry the information? | Causal | `04` |
| Sparse autoencoders | What features compose this activation? | Decompositional | `05` |

A useful discipline: **correlational tools generate hypotheses; causal tools
test them.** The projects are ordered accordingly.

## 2. Linear Probing

**Method.** Freeze the model; extract activations at a layer; train a linear
classifier to predict a property (syntax, semantics, world state) from them
([Alain & Bengio, 2016](https://arxiv.org/abs/1610.01644)). Layer-wise probe
accuracy maps where information becomes linearly accessible. Probing built an
entire subfield around "BERTology"
([Tenney et al., 2019](https://arxiv.org/abs/1905.05950) found the classical
NLP pipeline — POS → parsing → NER → coreference — laid out across BERT's
layers), and remains the workhorse for questions like "does the model
represent truth?" ([Marks & Tegmark, 2023](https://arxiv.org/abs/2310.06824)).

**Failure modes.** (i) A powerful probe can decode properties from noise —
control tasks and selectivity metrics are mandatory
([Hewitt & Liang, 2019](https://arxiv.org/abs/1909.03368)); (ii) presence ≠
use: the model may not read the direction the probe found
([Belinkov, 2022](https://arxiv.org/abs/2102.12452)); (iii) dataset
confounds — our own project demonstrates positional-embedding leakage at
layer 0.

**Project `01`** probes GPT-2's residual stream for noun-vs-verb on
*ambiguous* words ("run", "dance"), so token identity carries no signal:
embeddings probe at ~0.64 (positional leakage — a live demonstration of
failure mode iii), while every contextual layer probes at 1.00.

## 3. The Logit Lens Family

**Method.** Apply the model's own unembedding to the residual stream after
each layer, yielding a per-layer next-token distribution — the model's
*prediction trajectory*
([nostalgebraist, 2020](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)).
The **tuned lens** ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112))
learns per-layer affine translators before unembedding, correcting the
assumption that all layers share the final layer's coordinate system; it gives
lower-perplexity, less biased trajectories and transfers across model
families. Extensions decompose per-component contributions
("logit prisms", direct logit attribution) and apply the lens to modern LLMs
([LogitLens4LLMs](https://arxiv.org/abs/2503.11667)) and ViTs.

**Failure modes.** The raw logit lens can be badly miscalibrated on models
other than GPT-2 (it was famously near-useless on GPT-Neo); layernorm
interacts nonlinearly with the decoding; and a legible intermediate
"prediction" is still correlational evidence.

**Project `02`** shows GPT-2 resolving *"The Eiffel Tower is located in the
city of"*: generic syntax ("the") through layer ~5, the *city* category
appearing ~layer 8, candidates (London/Paris/Amsterdam) competing at 10, and
Paris winning from 11 — the semantic resolution happens in the last third of
the network.

## 4. Attention-Pattern Analysis and Induction Heads

**Method.** Attention weights are directly inspectable and sometimes reveal a
head's algorithm. The canonical success is the **induction head** — a
two-head circuit implementing `[A][B]…[A] → [B]` (a previous-token head writes
"what preceded me" into each position; the induction head queries it) —
identified in [Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)
and linked to the emergence of in-context learning via a training-dynamics
phase change in
[Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
Attention analysis generalizes into full **circuit discovery**: the indirect
object identification (IOI) circuit of 26 heads in 7 classes
([Wang et al., 2022](https://arxiv.org/abs/2211.00593)), automated variants
([ACDC, Conmy et al., 2023](https://arxiv.org/abs/2304.14997)), and
gradient-based approximations
([attribution patching, Nanda 2023](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)).

**Failure modes.** "Attention is not explanation" — weights alone ignore the
value vectors and everything downstream
([Jain & Wallace, 2019](https://arxiv.org/abs/1902.10186)); heads are often
polysemantic across contexts; behavioral and mechanistic evidence must be
paired.

**Project `03`** scores every GPT-2 head for attention at the induction offset
on repeated random tokens, recovering the published induction heads of GPT-2
small (L5H1, L5H5, L6H9, L7H2, L7H10) and pairing it with the behavioral
check (per-token loss collapses from 13.3 to 0.3 on the repeated half).

## 5. Activation Patching / Causal Tracing

**Method.** The core causal tool. Run a clean and a corrupted prompt; copy
("patch") a cached clean activation into the corrupted run at one site
(layer × position, or one component); measure restoration of clean behavior.
Introduced as *causal tracing* in ROME
([Meng et al., 2022](https://arxiv.org/abs/2202.05262)), where it showed
factual associations concentrated in mid-layer MLPs at the subject token —
motivating direct model editing. Methodological best practices (corruption
type, denoising vs. noising direction, logit-diff metrics) are analyzed in
[Zhang & Nanda, 2024](https://arxiv.org/abs/2309.16042) and
[Heimersheim & Nanda, 2024](https://arxiv.org/abs/2404.15255). The
generalization to arbitrary causal variables is **causal abstraction** /
interchange interventions
([Geiger et al., 2021](https://arxiv.org/abs/2106.02997)), with
distributed alignment search
([Geiger et al., 2023](https://arxiv.org/abs/2303.02536)) finding causal
directions rather than sites.

**Failure modes.** Results depend on the corruption distribution and metric;
single prompt pairs can mislead; backup/redundant circuits mask effects
(hydra effect); patching localizes *routes*, not *meanings*.

**Project `04`** patches the France→Italy capital pair and reproduces the
ROME-style picture: country identity sits at the subject token through
layer ~8, then moves to the final position (layers 9–11) where it drives the
prediction.

## 6. Sparse Autoencoders and Dictionary Learning

**Method.** Neurons are polysemantic, plausibly because models store more
features than dimensions — **superposition**
([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html)).
SAEs learn an overcomplete dictionary such that each activation is a sparse
non-negative combination of learned directions; sparsity is the proxy that
recovers (hopefully) the model's "true" features. Demonstrated on LMs by
[Cunningham et al., 2023](https://arxiv.org/abs/2309.08600) and
[Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html);
scaled to Claude 3 Sonnet
([Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html))
finding abstract multilingual/multimodal features (and the famous "Golden Gate
Bridge" feature); open suites exist for Gemma
([Gemma Scope](https://arxiv.org/abs/2408.05147)). SAE features compose into
**sparse feature circuits** ([Marks et al., 2024](https://arxiv.org/abs/2403.19647)),
and successors — transcoders
([Dunefsky et al., 2024](https://arxiv.org/abs/2406.11944);
[cross-layer transcoders](https://arxiv.org/abs/2501.18823)) — power
"attribution graph" biology-of-LLM analyses
([Lindsey et al., 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).

**Failure modes.** Reconstruction error is nonzero (the SAE explains *most*
of the activation); dead features and feature splitting; hyperparameter
sensitivity; and a live debate over whether SAE features carve the model at
its joints or impose our sparsity prior on it.

**Project `05`** trains a 768→3072 SAE on layer-6 GPT-2 activations over
tiny-shakespeare in minutes on CPU, then prints top-activating contexts per
feature — some clearly themed, some not, which is the honest state of small
SAEs.

## 7. Techniques Not Covered by a Project (but worth knowing)

- **Steering / activation addition** — add a direction to the residual stream
  to change behavior ([Turner et al., 2023](https://arxiv.org/abs/2308.10248);
  [representation engineering, Zou et al., 2023](https://arxiv.org/abs/2310.01405)).
- **Model editing** — surgically rewrite a fact's weights
  ([ROME](https://arxiv.org/abs/2202.05262), MEMIT).
- **Reverse-engineering toy models** — full algorithmic understanding of small
  transformers, e.g. modular addition via Fourier features and the "grokking"
  connection ([Nanda et al., 2023](https://arxiv.org/abs/2301.05217)).
- **Weight-based analysis** — e.g. singular vectors of OV/QK matrices; the
  "residual stream as memory" analyses.
- **Automated interpretability** — LLMs labeling neurons/features
  ([Bills et al., 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)).

## 8. Cross-Cutting Limitations and Open Problems

1. **Faithfulness vs. plausibility.** A legible story is not necessarily the
   model's story; interpretability illusions are documented for probes,
   attention, and subspace interventions.
2. **Scalability.** Full circuit analyses exist mainly for GPT-2-scale models
   and narrow tasks; scaling honest mechanism-finding to frontier models is
   the field's central open problem (attribution graphs are the current
   frontier).
3. **The unit-of-analysis question.** Neurons? Directions? SAE features?
   Circuits? Each choice bakes in assumptions the model need not respect.
4. **Evaluation.** No agreed benchmark says an "explanation" is correct;
   causal-intervention recovery and downstream task utility (editing,
   steering, unlearning) are the current proxies.
5. **Universality.** Do the same circuits recur across models and scales?
   Induction heads suggest yes; details often differ.

## 9. How to Use This Repository

Run the projects in order — each introduces one technique and reproduces one
known result (see `projects/*/README.md` for per-project details, expected
outputs, pitfalls, and extensions). Then port an experiment to
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) —
hook-based workflows are what the plain-PyTorch scripts here teach, and
TransformerLens makes them ergonomic. When ready for real exercises, do
[ARENA](https://github.com/callummcdougall/ARENA_3.0) chapter 1.

## References (deduplicated, roughly chronological)

- Mikolov et al., 2013. [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)
- Alain & Bengio, 2016. [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
- Jain & Wallace, 2019. [Attention is not Explanation](https://arxiv.org/abs/1902.10186)
- Tenney et al., 2019. [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950)
- Hewitt & Liang, 2019. [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368)
- Olah et al., 2020. [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)
- nostalgebraist, 2020. [Interpreting GPT: the logit lens](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- Elhage et al., 2021. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Geiger et al., 2021. [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)
- Belinkov, 2022. [Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452)
- Meng et al., 2022. [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262)
- Olsson et al., 2022. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Elhage et al., 2022. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- Wang et al., 2022. [Interpretability in the Wild (IOI)](https://arxiv.org/abs/2211.00593)
- Li et al., 2023. [Emergent World Representations (Othello-GPT)](https://arxiv.org/abs/2210.13382)
- Nanda et al., 2023. [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)
- Belrose et al., 2023. [Eliciting Latent Predictions with the Tuned Lens](https://arxiv.org/abs/2303.08112)
- Geiger et al., 2023. [Finding Alignments (DAS)](https://arxiv.org/abs/2303.02536)
- Conmy et al., 2023. [Automated Circuit Discovery (ACDC)](https://arxiv.org/abs/2304.14997)
- Turner et al., 2023. [Activation Addition](https://arxiv.org/abs/2308.10248)
- Cunningham et al., 2023. [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600)
- Zhang & Nanda, 2024. [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042)
- Bricken et al., 2023. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- Zou et al., 2023. [Representation Engineering](https://arxiv.org/abs/2310.01405)
- Marks & Tegmark, 2023. [The Geometry of Truth](https://arxiv.org/abs/2310.06824)
- Bereska & Gavves, 2024. [Mechanistic Interpretability for AI Safety — A Review](https://arxiv.org/abs/2404.14082)
- Heimersheim & Nanda, 2024. [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
- Ferrando et al., 2024. [A Primer on the Inner Workings of Transformer-Based LMs](https://arxiv.org/abs/2405.00208)
- Marks et al., 2024. [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647)
- Templeton et al., 2024. [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- Dunefsky et al., 2024. [Transcoders](https://arxiv.org/abs/2406.11944)
- Rai et al., 2024. [A Practical Review of Mechanistic Interpretability](https://arxiv.org/abs/2407.02646)
- Lieberum et al., 2024. [Gemma Scope](https://arxiv.org/abs/2408.05147)
- Lindsey et al., 2025. [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- Paulo et al., 2025. [Transcoders Beat Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2501.18823)
