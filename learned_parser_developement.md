# Learned Parser Development

The grammar-constrained learned parser trains from final-scene feedback with REINFORCE. The current refactor combines the experiment entry points and shares model assembly, execution and program training across both grounding backends.

## 1. Learned Parser Modules

| File | Responsibility |
|---|---|
| `_learned_parser.py` | `lex_step()` extracts keywords and context. `Grammar` masks invalid roles and `role_to_program()` builds a Move AST. |
| `batch.py` | `ParserInputs` and `LearnedBatch` separate parser inputs from final-scene targets. |
| `learned_lr.py` | `RolePolicy` uses an embedding, BiGRU and linear classifier to predict seven keyword roles. `sample_many()` encodes each batch once for multiple grammar-valid draws. |
| `combined_train.py` | `fit_programs()` runs the supplied phases. Scene loss trains the backend, and the parser uses detached negative scene loss as its REINFORCE reward. |
| `combined_metrics.py` | Reports scene and exact-program metrics, including IoU-M, for overall, simple and complex examples. |

### Training flow

```text
Instruction + Scene
  → batch.py / ParserInputs
  → BiGRU RolePolicy
  → Grammar-constrained roles
  → role_to_program / Move AST
  → Backend grounding + shared Action Simulator
  → Predicted final scene
  → Scene loss + REINFORCE and entropy regularization
  → Parameter updates
```

The original mini experiments use `train.train()`. Both program experiment paths use `combined_train.fit_programs()`.

## 2. Shared Execution and Model Abstraction

### `execution.py`: Shared execution

**`apply_move(sim, boxes, subj_w, ref_w, action_emb, *, move_mode="soft")`**

- Soft mode uses weighted object positions and updates the scene proportionally to the subject weights.
- Hard mode selects one subject and reference with argmax and uses straight-through gradients.
- Returns the updated object boxes.

**`execute_steps(scene, program, sim, ground_step, *, move_mode)`**

- Takes an initial scene, a batch of Move sequences and a backend-specific grounding callback.
- Grounds each Move against the current boxes, then calls the simulator and applies the update before the next step.
- Returns `ExecutionResult(final_boxes, grounded_program)`.

`NSModel` grounds through the symbolic Executor. `ParsedNeuralBaseline` converts Move ASTs to word groups and uses its neural grounding modules. Both expose `execute()` and pass their own `_ground_step()` to the shared loop.

### Model selection and training

**`model.build(*, parser_source, backend)`**

| Parameter | Options | Effect |
|---|---|---|
| `parser_source` | `"rule"` / `"learned"` | Fixed programs from `parser.parse()` / attached `RolePolicy` |
| `backend` | `"symbolic"` / `"neural"` | `NSModel` / `ParsedNeuralBaseline` |

The factory injects an Action Simulator and attaches the optional learned policy as `model.parser`. For a fixed backend and seed, initializing the policy in a separate RNG scope preserves paired simulator and grounding weights across parser settings. The factory replaces `ParseNSModel`, and shared execution supplies the final boxes and grounding trace previously handled by its separate path.

**`combined_train.fit_programs(model, data, settings, *, phases, seed, ...)`** uses one optimizer and scheduler per run. `run_experiments.py` supplies the phase names and epoch counts, and the trainer applies freezing and sample selection for each phase.

- **Schedule A.** Train jointly for 20 simple-instruction epochs and 40 mixed-instruction epochs.
- **Schedule B.** Use 20 joint warm-up epochs, 30 Action Simulator-only epochs, then 20 relation-grounding epochs that also train the learned parser when present.

## 3. Changes to Existing Mini Modules

| File | Changes |
|---|---|
| `batch.py` | Shares scene tensor construction between rule and learned collation behind `collate(mode="rule")`. |
| `baseline.py` | Extracts neural grounding from `ParsedNeuralBaseline.forward()` and adds AST-to-token conversion, simulator injection and shared execution. |
| `model.py` | Adds `build()`, simulator injection and shared execution to the symbolic path. |
| `data.py` | Adds keyword classification and supports sampling extra complex scenes for the matrix. |
| `run_experiments.py` | Keeps the original mini phases and folds learned-parser experiments, group/seed training and checkpoint replay into the same CLI. |

The mini training loop still lives in `train.py`. Both learned training paths use `combined_train.py`, and the separate `LearnedOutput` class was removed.

## 4. Two Experiment Paths

**Dedicated learned-parser experiments**

```text
run_experiments.py --learned-parser
  → build(parser_source="learned", backend="symbolic")
  → fit_programs() → evaluate_learned() on the final in-memory model
```

**Configurable curriculum experiments**

```text
run_experiments.py --experiment
  → build() and fit_programs() for each group and seed
  → save_checkpoint() after each phase → evaluate() on the final checkpoint
```

The matrix compares parser source, backend, loss, schedule and move mode. `--checkpoint` replays a saved run directly through `experiment_eval.evaluate()`. [Experiment instructions](learned_parser_experiment.md)
