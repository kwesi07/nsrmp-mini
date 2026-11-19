# nsrmp-mini

A small, from-scratch re-implementation of the core ideas of

> Namasivayam K., Singh H., Bindal V., Tuli A., Agrawal V., Jain R., Singla P., Paul R.
> **Learning Neuro-symbolic Programs for Language Guided Robot Manipulation.** ICRA 2023.
> [arXiv:2211.06652](https://arxiv.org/abs/2211.06652)

made for the COMP3314 bonus task (a small re-implementation study, not a reproduction).

## What this is, and what it is not

It **is** a simplified, runnable study of the paper's central claim: executing a *symbolic program*
over learned concepts (Filter / Relate / Unique + an Action Simulator, trained only from final
object boxes) handles relational instructions and generalises better than a purely neural
language-attention model. In our simplified world, with the parse held equal, we do **not** find that advantage
(see Discussion).

It is **not** a reproduction of the paper's numbers. The world is a 2D top-down blocks world with
no images, no robot and no object detector. All code is ours; no code from the official repository was
used. The two papers' ideas (NSRMP, and NS-CL for soft concept execution) are the inspiration.
Our numbers and the paper's numbers are measured on different data and **must not be compared as
like with like**.

## Paper module -> our file

| Paper | File |
|---|---|
| Sec. IV-A dataset (here: synthetic scenes, instructions, gold programs, gold final states) | `nsrmp/data.py` |
| Sec. III-A language parser (given, as in the paper's "program supervision" ablation) | `nsrmp/parser.py` |
| Sec. III-B Visual Reasoner: Scene / Filter / Relate / Unique, NS-CL style soft execution | `nsrmp/executor.py`, `Concepts` in `nsrmp/model.py` |
| Sec. III-C Action Simulator, predicted final world | `ActionSimulator`, `apply_move`, `NSModel` in `nsrmp/model.py` |
| Sec. III-D loss and curriculum training | `nsrmp/train.py` |
| Sec. IV-B NMN+ baseline (Fig. 3): parse-matched `ParsedNeuralBaseline` and end-to-end GRU `NeuralBaseline` | `nsrmp/baseline.py`, role word groups from `parse_roles` in `nsrmp/parser.py` |
| Sec. IV-B metrics (IOU, IOU-M, subject and placement accuracy) | `nsrmp/metrics.py` |
| Table II, Fig. 4a, Fig. 4b | `run_experiments.py` |

## Simplifications and why

- **No images, no detector.** "Visual features" are `P · [onehot(colour); onehot(shape)] + N(0, 0.1²)`
  with a fixed random 32x9 matrix `P`. We have no YOLO/ResNet, so only the paper's **gold-BB** setting is
  tested; the "extracted BB" rows of Table II have no analogue here. Concepts are trivially separable
  from these features, which makes the *concept* part easy (see discussion).
- **Given parser.** A deterministic rule-based parser produces the programs (the paper trains its own
  language encoder, with a given-program ablation in its repo). This removes the language-learning part.
- **Templated language.** ~15 words, one sentence template, relations only through "which is <relation> the <simple object>".
  Every referring expression selects exactly one object (the generator checks this).
- **Four actions** (`MoveLeft/Right/Front/Behind`) with a fixed gap; the paper has a richer action set.
- **Action Simulator = residual MLP** (output is added to the subject box) and the relation MLP also
  receives the scaled difference of the two boxes. Both only make optimisation easier; neither changes the idea.
- **Cosine learning-rate decay** (both models). Without it IOU-M of our model plateaus near 0.66 because
  the Action Simulator never gets precise.
- **Filter threshold/temperature** fixed at `gamma=0.2`, `tau=0.1` (NS-CL style).
- **Multi-step**: the splitter on "then" is given and shared by all models (the paper's LSTM splitter is learned).
- **Metrics**: IOU averages over all objects, IOU-M pools all moved objects. Subject accuracy is averaged over steps.
  Placement accuracy: the predicted box centre of a moved object lies inside its gold final box.
  The paper's exact aggregation may differ.

## Setup and run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest -q                              # 59 tests
.venv/bin/python run_experiments.py --phase all  # ~7 min on an M2 Pro CPU
.venv/bin/python diagnose_baseline.py            # ~2 min, explains the GRU baseline result
.venv/bin/python run_experiments.py --phase noise # ~4.5 min, optional feature-noise sweep
.venv/bin/python diagnose_parsed_baseline.py     # ~3 min, re-runs the parse-matched diagnostics table
```

Settings used for the results below: 4000 train / 1000 test examples (50/50 simple/complex, 3-5 objects),
**60 epochs**, Adam lr 3e-3 with cosine decay, batch 64, curriculum = first 20 epochs simple-only, 3 seeds (0, 1, 2),
500 test examples per object count (3..10) or step count (1..5). All models get identical data, seeds, epochs,
curriculum, Action Simulator architecture and loss (`sum_i |box_hat_i - box_i|_1 + 1 * (1 - IoU)`). The full run took
**8 min 6 s** (phase 1: 232 s, phase 2: 250 s, plus data generation). Mean ± std is over the 3 seeds, using the
**sample** standard deviation (`ddof=1`, appropriate for only 3 seeds); each seed changes the data, initialisation and shuffling.

## The two baselines, and why the parse-matched one is the fair comparison

The paper compares against NMN+, which **shares the language encoder and splitter** with the main model, so the comparison
isolates *symbolic program execution* vs *neural attention*. We report two neural baselines (`nsrmp/baseline.py`):

- **Parse-matched neural baseline (NMN+-like)**, `ParsedNeuralBaseline`. It gets *the same parse* as our model, as word
  groups per step (`parse_roles` in `nsrmp/parser.py`): the subject's concept words, its relation word and the concept
  words of the relation's anchor object, the same three for the reference, and the action word. These are exactly the
  names the program contains (no "the"; "object" = wildcard is dropped). Each group is a bag of word embeddings. It does
  **not** run Filter/Relate/Unique and has no concept embeddings: one neural attention module `Find` locates every noun
  phrase, `w_i = softmax_i(<A f_i, B head_words> + [has relation] * MLP([box_i; c; 10(box_i - c); relation_word]))`, where
  `c = sum_j a_j box_j` is a context box pooled with an attention `a` computed from the anchor's words. The action comes
  from the action-word embedding through a softmax head. Same Action Simulator, `apply_move`, loss and training as ours.
  **This is the fairer comparison**: the parse is held equal, so the gap measures symbolic execution vs neural attention.
- **End-to-end GRU baseline**, `NeuralBaseline` (unchanged). It reads the whole step through one GRU and must discover
  which words describe the subject, the reference and the action from the final-box loss alone. Our model gets all of
  that from the parser, so this comparison is **confounded** by the parse: it does not tell us whether symbolic
  execution helps.

### Making the parse-matched baseline learn (diagnostics, seed 0 unless stated)

The first version, with the per-role concat-MLP attention `MLP([f_i; box_i; phrase_vec])` from the spec, **collapsed exactly like
the GRU baseline** even with the parse. Every row below is re-run by `diagnose_parsed_baseline.py` (about 3 min; output in
`results/diagnose_parsed_baseline.{txt,json}`). These are diagnostics only, not reported results. Phase-1 data, same training as above;
IOU-M and subject accuracy are given as simple / complex.

| Variant | Epochs, seeds | Outcome |
|---|---|---|
| Concat-MLP attention, per role (spec design), all phrase words | 30, seed 0 | collapses: IOU-M 0.005 / 0.003, subject acc. 0.26 / 0.26 |
| ... one head shared by subject and reference; NMN-style product fusion (per role or shared); boxes fed only through the relation term | 30, seed 0 | all collapse: IOU-M <= 0.005, subject acc. <= 0.30 |
| ... subject excluded from the reference inside `apply_move` (tried on all three models) | 30, seed 0 | no change: this baseline and the GRU baseline still collapse; NSModel 0.779 / 0.761 |
| ... gold action given (diagnostic) | 30, seed 0 | still collapses |
| ... gold subject given (diagnostic) | 30, seed 0 | learns: IOU-M 0.756 / 0.669 |
| Bias-free dot-product word score, separate Find per role, "the" kept | 30, seed 0 | collapses: IOU-M 0.005 / 0.003 |
| ... concept words only | 60, seeds 0-2 | learns on 1 of 3 seeds (seed 2: 0.806 / 0.758); seeds 0, 1 collapse (IOU-M <= 0.007, subject acc. 0.44-0.75 simple) |
| ... one Find shared by subject and reference, separate anchor weights | 60, seeds 0-2 | learns on 1 of 3 seeds (seed 2: 0.790 / 0.738); seeds 0, 1 collapse |
| Bias-free dot-product, **one Find shared by subject, reference and anchor** (final) | 60, seeds 0-2 | learns on all 3 seeds: IOU-M 0.781-0.800 / 0.772-0.796 |

The two "1 of 3 seeds" variants are fragile. An earlier, unscripted run, which built its modules in a different order
and so started from a different initialisation, gave different seed outcomes. Only the final design learned in every
run we made, and three seeds are still a small sample.

What we found (the spec design, seed 0, from the same script):
1. **The collapse is fast and word-independent.** The entropy of the subject attention on the test set falls from 1.37
   to 0.22 after one epoch and to 0.02 by the end, while the weight on the gold subject stays at chance (0.26) throughout.
   In 91% of test examples the subject and reference attention pick the *same* object, and the Action Simulator learns
   to leave it in place: "move nothing".
2. **The gradient itself is not the problem.** At initialisation, `-dL/dw` favours the gold subject over the other
   objects in 94% of simple test examples. But a concat MLP can lower the loss faster by sharpening attention on cues that
   ignore the words (box position, features), and once a softmax is saturated its gradient vanishes.
3. **The soft `apply_move` blend makes this a strong local minimum.** Under the L1 loss, when the reference is
   unknown, the best Action Simulator output is "stay put". With a stay-put simulator, neither the subject nor the
   reference attention receives any signal. The oracle-subject run shows that everything else (reference, relation
   context, action, simulator) learns once the subject is right.
4. **What fixed it.** Both changes are structural choices, applied only to this baseline and documented here. There is
   no subject supervision, and no per-model learning rate or epoch count.
   (a) The word-object score is a **bias-free dot product** over **only the parsed concept words**, so with no words
   every object scores the same and attention can only sharpen by matching words.
   (b) **One `Find` module serves every noun phrase**, including the anchor, as in NMN and as our model shares Filter
   across roles. Grounding learned for one role then transfers to the others.

These fixes do not transfer to the GRU baseline: it has no word groups to restrict, dot-product attention was already
tried on it without success, and with its single sentence vector a shared Find would give subject = reference. So it is
left unchanged by design. We did **not** test whether the same
mechanism fully explains the GRU collapse; `results/diagnose_baseline.txt` remains the evidence for it (loss equal to
"move nothing", learns with an oracle subject or with extra subject supervision). The `Find` word score is a dot product
between projected object features and word embeddings, which is close in spirit to concept matching. So the remaining
difference to our model is the **symbolic composition**: Filter products, `Unique`, and `Relate` over a learned relation
classifier, versus softmax attention with a pooled-context relation term. It is not grounding as such.

## Our results (synthetic 2D world, gold boxes)

### Table II analogue, test set with 3-5 objects, single step

| Model | IOU | IOU-M | IOU-M simple | IOU-M complex | Subject acc. simple | Subject acc. complex | Placement acc. |
|---|---|---|---|---|---|---|---|
| Ours (neuro-symbolic) | 0.929 ± 0.002 | 0.776 ± 0.005 | 0.779 ± 0.006 | 0.774 ± 0.003 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.955 ± 0.005 |
| Parse-matched neural baseline (NMN+-like) | 0.946 ± 0.002 | 0.786 ± 0.011 | 0.788 ± 0.010 | 0.784 ± 0.012 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.940 ± 0.013 |
| End-to-end GRU baseline | 0.744 ± 0.002 | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.004 ± 0.001 | 0.281 ± 0.018 | 0.249 ± 0.009 | 0.007 ± 0.001 |
| Ours, no curriculum (ablation) | 0.935 ± 0.002 | 0.796 ± 0.005 | 0.797 ± 0.007 | 0.795 ± 0.008 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.969 ± 0.004 |

### Fig. 4a analogue: IOU-M vs number of objects (trained on 3-5)

![IOU-M vs objects](results/fig4a_objects.png)

| Model | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| Ours | 0.785 ± 0.016 | 0.781 ± 0.016 | 0.767 ± 0.005 | 0.762 ± 0.004 | 0.749 ± 0.007 | 0.733 ± 0.007 | 0.716 ± 0.016 | 0.707 ± 0.014 |
| Parse-matched baseline | 0.788 ± 0.009 | 0.787 ± 0.012 | 0.786 ± 0.010 | 0.792 ± 0.006 | 0.782 ± 0.012 | 0.782 ± 0.004 | 0.777 ± 0.004 | 0.771 ± 0.007 |
| End-to-end GRU baseline | 0.005 ± 0.004 | 0.004 ± 0.003 | 0.005 ± 0.000 | 0.005 ± 0.004 | 0.006 ± 0.002 | 0.007 ± 0.001 | 0.003 ± 0.002 | 0.007 ± 0.002 |
| Ours, no curriculum | 0.801 ± 0.012 | 0.798 ± 0.012 | 0.788 ± 0.006 | 0.778 ± 0.006 | 0.771 ± 0.008 | 0.756 ± 0.007 | 0.738 ± 0.011 | 0.727 ± 0.008 |

### Fig. 4b analogue (phase 2): trained on 1-2 steps, tested on 1-5 steps (3-5 objects)

On the 1-2 step test set (IOU / IOU-M / subject acc.): ours 0.905 ± 0.001 / 0.776 ± 0.002 / 0.999 ± 0.000;
parse-matched baseline 0.931 ± 0.005 / 0.799 ± 0.012 / 0.999 ± 0.000; GRU baseline 0.657 ± 0.002 / 0.007 ± 0.002 / 0.272 ± 0.003.

![IOU-M vs steps](results/fig4b_steps.png)

| Model | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Ours | 0.790 ± 0.001 | 0.774 ± 0.001 | 0.738 ± 0.009 | 0.720 ± 0.007 | 0.694 ± 0.003 |
| Parse-matched baseline | 0.807 ± 0.014 | 0.797 ± 0.012 | 0.761 ± 0.018 | 0.749 ± 0.012 | 0.729 ± 0.008 |
| End-to-end GRU baseline | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.007 ± 0.002 | 0.008 ± 0.002 |

### Feature-noise sweep (optional extra): phase-1 test set, train and test at the same noise σ

A cheap stand-in for the paper's gold-vs-detected-box gap: only the feature noise changes; scenes and instructions are identical
(`run_experiments.py --phase noise`, 4 min 29 s; same settings as above).

| Model | IOU-M σ=0.1 | IOU-M σ=0.3 | IOU-M σ=0.5 | Subject acc. σ=0.1 | Subject acc. σ=0.3 | Subject acc. σ=0.5 |
|---|---|---|---|---|---|---|
| Ours (neuro-symbolic) | 0.776 ± 0.005 | 0.786 ± 0.003 | 0.759 ± 0.005 | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 |
| Parse-matched baseline | 0.786 ± 0.011 | 0.806 ± 0.011 | 0.802 ± 0.006 | 0.999 ± 0.001 | 0.998 ± 0.001 | 0.997 ± 0.001 |

Subject accuracy stays ~1.0 for both models even at σ=0.5. Even this much noise leaves the 9 concepts separable in 32 dimensions,
so the sweep does **not** recreate the grounding difficulty of detected boxes and is a weak stand-in. IOU-M changes come from
Action Simulator precision, not grounding (we did not investigate why both models are slightly better at σ=0.3).

Raw per-seed numbers: `results/phase1.json`, `results/phase2.json`, `results/noise.json`; tables: `results/results_phase{1,2}.md`,
`results/results_noise.md`.

## Checks against the authors' code

Two questions about how our setup differs from the authors' released code. Both are opt-in; every number above is unchanged. Our results, not the paper's. Script: `compare_authors_variants.py` (phase-1 settings, 3 seeds, 311 s on the same CPU); full tables incl. simple/complex splits in `results/authors_variants.md`.

**1. Hard (argmax) evaluation of a move step.** We train and (by default) evaluate with a soft blend: every object moves toward the simulator's output by its subject weight. The authors blend softly in training but at inference take the argmax subject `o1` and argmax reference `o2`, feed `boxes[o1]`, `boxes[o2]` to the simulator and overwrite only `boxes[o1]`. `evaluate(model, batch, hard=True)` does the same (`apply_move(..., hard=True)`); the models are trained exactly as before, only the evaluation changes.

**2. A port of the authors' NMN+ object scoring.** `AuthorsNMNBaseline` (`nsrmp/baseline.py`) keeps our GRU encoder and multi-step loop and replaces the per-role scorer with the authors': per role (subject, reference) two attentions `softmax_i(MLP([Map(sentence); f_i]))`, multiplied and renormalised, with no box input; the action is `softmax(decoder(sentence))`. Hidden sizes are the authors' divided by 4 (Map 512-256 -> 128-64; scorer 256-256-128-1 -> 64-64-32-1; decoder 512-256-256-128 -> 128-64-64-32), since their feature and GRU widths come from configs we cannot see. Deviation: we keep our shared `ActionSimulator` and `apply_move` instead of their `pred` MLP so the comparison stays fair.

Phase-1 test set (3-5 objects, single step), overall; mean ± sample std over 3 seeds. Soft (the evaluation used everywhere above):

| Model | IOU | IOU-M | Subject acc. | Placement acc. |
|---|---|---|---|---|
| Ours (neuro-symbolic) | 0.929 ± 0.002 | 0.776 ± 0.005 | 1.000 ± 0.001 | 0.955 ± 0.005 |
| Parse-matched neural baseline (NMN+-like) | 0.946 ± 0.002 | 0.786 ± 0.011 | 0.999 ± 0.001 | 0.940 ± 0.013 |
| End-to-end GRU baseline | 0.744 ± 0.002 | 0.005 ± 0.001 | 0.265 ± 0.005 | 0.007 ± 0.001 |
| NMN+ (authors' scoring) | 0.750 ± 0.002 | 0.005 ± 0.001 | 0.247 ± 0.011 | 0.007 ± 0.001 |

Hard (argmax subject and reference, authors' inference):

| Model | IOU | IOU-M | Subject acc. | Placement acc. |
|---|---|---|---|---|
| Ours (neuro-symbolic) | 0.941 ± 0.002 | 0.764 ± 0.005 | 1.000 ± 0.001 | 0.951 ± 0.008 |
| Parse-matched neural baseline (NMN+-like) | 0.947 ± 0.003 | 0.789 ± 0.012 | 0.999 ± 0.001 | 0.942 ± 0.015 |
| End-to-end GRU baseline | 0.751 ± 0.003 | 0.005 ± 0.001 | 0.265 ± 0.005 | 0.007 ± 0.001 |
| NMN+ (authors' scoring) | 0.751 ± 0.003 | 0.005 ± 0.001 | 0.247 ± 0.011 | 0.007 ± 0.001 |

Reading: the soft numbers for our model and the two existing baselines reproduce the Table II analogue exactly (same data, seeds and training). Hard evaluation lowers our model's IOU-M by about 0.012 (0.776 to 0.764) but raises its IOU (0.929 to 0.941), because objects that were only slightly blended toward the moved box now stay exactly in place; the parse-matched baseline barely changes. The authors-style NMN+ scoring collapses exactly like the GRU baseline (IOU-M 0.005, subject accuracy about 0.25): per-seed IOU-M matches the GRU baseline to three decimals. Removing the box input and using product attention did not fix it, so in our setting the collapse is not explained by the scorer's form. This is a statement about our synthetic setting and our scaled-down port, not about the authors' system.

## From the paper (Table II / Fig. 4), gold bounding boxes

| Model | IOU | IOU-M overall | IOU-M simple | IOU-M complex |
|---|---|---|---|---|
| Ours (gold BB) | 0.87 | 0.72 | 0.73 | 0.64 |
| NMN+ (gold BB) | 0.77 | 0.55 | 0.71 | 0.31 |

Fig. 4 gives no numbers, only the trend: the paper's model degrades more gracefully than the baseline
as the number of objects (4a) and steps (4b) grows. The paper reports no standard deviations.

## Discussion

**The fair comparison does not show a neuro-symbolic advantage in our world.** With the parse held equal, the
parse-matched neural baseline is on par with our neuro-symbolic model on the 3-5 object test set (IOU-M 0.786 ± 0.011 vs
0.776 ± 0.005, a difference within about one standard deviation of the baseline, 3 seeds). It is equally good on simple and complex instructions (0.788 vs 0.784)
and degrades *less* with more objects (0.771 vs 0.707 at 10 objects) and more steps (0.729 vs 0.694 at 5 steps).
Our model's placement accuracy is slightly higher (0.955 vs 0.940). So in this synthetic world we do **not** observe
the paper's main result, that symbolic programs beat NMN+, especially on complex instructions (paper: NMN+ 0.31 vs 0.64).
The large gap to the end-to-end GRU baseline (0.776 vs 0.005) is therefore not caused by symbolic execution,
since a parse-matched neural model closes it. But the parse alone was not enough either: with the spec's concat-MLP
attention the parse-matched model collapsed too, so the attention design (word-only scores, one shared Find) mattered as well.

**Why our world probably cannot show the paper's effect.**
1. **Relations are easy here.** Every relation is a comparison of two box centres, and every noun phrase has at most
   one relation with a simple anchor. A pooled-context attention handles that as well as `Relate`. The paper's real
   scenes, detected boxes and richer language are harder.
2. **Concepts are trivially separable.** Features are noisy one-hots through a fixed projection, so a dot-product
   attention grounds colours and shapes as easily as concept embeddings do (subject accuracy 1.000 for both models).
3. **The baseline needed design care to learn at all** (above). The paper's NMN+ results presumably also depend on such
   choices; we cannot tell from the paper how its NMN+ attention was built.

**Other observations.**
- **The end-to-end GRU baseline collapses** to "move nothing" (IOU-M 0.005, subject acc. at chance), unlike the
  paper's NMN+ (0.71 simple / 0.31 complex). This is a real optimisation failure, not a measurement bug
  (`diagnose_baseline.py`, `results/diagnose_baseline.txt`): its final training loss (2.29) equals the loss of
  moving nothing (2.22 on the test set); with the gold subject given it learns to move boxes (IOU-M 0.52), and with an
  extra subject-supervision loss (diagnostic only, never used in reported results) it grounds subjects with 0.86
  accuracy. Wider heads, product / dot-product attention, learning rates 5e-4 to 1e-2, batch sizes 16-256, 4x data,
  100 epochs, sharper attention and removing the box input did not help. It should not be read as evidence for the paper.
- **No curriculum is slightly better than curriculum for our model** (IOU-M 0.796 vs 0.776, one ablation, 3 seeds).
  This does not contradict the paper. Its curriculum mainly serves the parts we do not have: its parser is learned with
  REINFORCE, and its Action Simulator is trained in stages, so simple instructions first make the early program and
  action learning tractable. With a given parser that benefit cannot appear here, and the curriculum only costs a third
  of the epochs on half the data.
- **Absolute numbers differ and are not comparable** (synthetic data, templated language, different metric
  aggregation). Our IOU-M ceiling (~0.78-0.80) comes from Action Simulator precision, not grounding: subject accuracy
  is ~1.0 and placement accuracy ~0.95, but boxes are only ~0.1 wide, so a 0.02 placement error already costs a lot of IoU.
- **Standard deviations are small** (mostly <= 0.015) because data and features are synthetic and the models are small;
  they reflect seed variance only, not variance from a different world or generator.
- Our model's generalisation degrades partly because softer `Unique` weights and relation errors accumulate with more
  objects / steps (predicted boxes of earlier steps feed later ones); we did not isolate which factor dominates.

## Our contributions vs reused

Everything in this repository was written by us from the paper text: the world and generator, parser, soft executor,
concept/relation/Action-Simulator networks, both baselines, metrics, training and experiment scripts, and tests (written test-first).
The *ideas* come from NSRMP (program-based visual reasoning + Action Simulator trained from final states, curriculum, NMN+
comparison) and from NS-CL (soft Filter/Relate/Unique execution with concept embeddings). We reused no code from either project;
the only third-party libraries are PyTorch, NumPy, Matplotlib and pytest.

## Honest limitations

Gold boxes only; given parser and splitter; synthetic templated language; 3 seeds; one fixed hyper-parameter setting shared by all
models (chosen on seed-0 data of our own model, not tuned per model); the GRU baseline failure above. The parse-matched baseline's
attention design was chosen after diagnostic runs evaluated on the same test-set generator (seeds 0-2), so it is
slightly favoured by selection; the choice was made on a structural principle (attention may only sharpen by matching
words; one shared Find module), not by sweeping hyper-parameters.
