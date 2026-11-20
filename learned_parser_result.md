# Learned parser results

## Round 3 on the formal test scenes

Moved-object IoU (IoU-M), mean ± sample standard deviation across seeds 0–9. Each independently trained model uses its own final checkpoint, parser and move mode on 1,000 test scenes per instruction length.

```text
Group  Parser   Backend       Loss      Schedule  Move   Single-step       Two-step
    1  learned  NS            original     A      soft   0.2397 ± 0.3064   0.2253 ± 0.2866
    2  learned  NS            squared      A      soft   0.5310 ± 0.0478   0.4843 ± 0.0551
    3  rule     NS            original     A      soft   0.7392 ± 0.0216   0.7046 ± 0.0223
    4  rule     NS            squared      A      soft   0.7741 ± 0.0170   0.7430 ± 0.0137
    5  learned  NS            squared      B      soft   0.5238 ± 0.0705   0.4807 ± 0.0644
    6  learned  NS            squared      B      hard   0.5606 ± 0.0562   0.5187 ± 0.0471
    7  rule     NS            squared      B      soft   0.6877 ± 0.0262   0.6483 ± 0.0208
    8  rule     NS            squared      B      hard   0.7061 ± 0.0221   0.6812 ± 0.0255
    9  learned  ParsedNeural  squared      B      soft   0.4946 ± 0.0868   0.4447 ± 0.0786
   10  learned  ParsedNeural  squared      B      hard   0.4798 ± 0.1077   0.4303 ± 0.1064
```

ParsedNeural denotes `ParsedNeuralBaseline`. [Per-seed scores and paired differences](results/parser-refactor-final/matrix/round3/curriculum/summary.json)

**Schedule A (groups 1–4)** trains jointly for 20 epochs on simple instructions and 40 on mixed instructions. Both phases use soft moves.

**Schedule B (groups 5–10)** trains jointly for 20 warm-up epochs, then trains only the Action Simulator for 30 epochs. The last 20 epochs use mixed instructions to train relation grounding and the learned parser when present. Hard-move groups switch at epoch 21.

- With the symbolic backend, the rule parser led on both instruction lengths for every seed in the four matched comparisons. Group 6 led the learned-parser groups with 0.5606 on one-step instructions and 0.5187 on two-step instructions.
- With learned parsers, the symbolic backend led ParsedNeural on average under both move modes in Round 3. With hard moves, the advantage was 0.0809 on one-step instructions and 0.0884 on two-step instructions. Each backend trained its own parser.
- With the symbolic backend and Schedule A, squared loss raised the learned parser’s mean one-step IoU-M from 0.2397 to 0.5310 and reduced its sample standard deviation from 0.3064 to 0.0478. The rule parser gained 0.0349 in mean one-step IoU-M.
- In Schedule B, hard moves raised mean two-step IoU-M by 0.0380 in the learned symbolic group and by 0.0328 in the rule symbolic group. The learned neural group scored 0.0143 higher with soft moves.

## Hyperparameters across the three matrices

Each round retrained the same ten groups on seeds 0–9 using the refactored code. Changes were cumulative.

```text
Matrix     Minimum LR   Complex scenes per seed   Samples per instruction*
Round 1    0.0006       2,000                     4
Round 2    0.0006       6,000                     4
Round 3    0.0006       6,000                     8
```

*Samples per instruction when training the learned parser. Adam starts at 0.003 in every round. The pairs below show one-step / two-step mean IoU-M.*

```text
Group     Round 1           Round 2           Round 3
    1     0.2257 / 0.2101   0.2464 / 0.2298   0.2397 / 0.2253
    2     0.4582 / 0.4152   0.5255 / 0.4819   0.5310 / 0.4843
    3     0.7358 / 0.7037   0.7392 / 0.7046   0.7392 / 0.7046
    4     0.7652 / 0.7344   0.7741 / 0.7430   0.7741 / 0.7430
    5     0.4759 / 0.4322   0.4756 / 0.4287   0.5238 / 0.4807
    6     0.5359 / 0.4962   0.5404 / 0.5040   0.5606 / 0.5187
    7     0.6877 / 0.6479   0.6877 / 0.6483   0.6877 / 0.6483
    8     0.7057 / 0.6810   0.7061 / 0.6812   0.7061 / 0.6812
    9     0.5264 / 0.4741   0.4684 / 0.4204   0.4946 / 0.4447
   10     0.4892 / 0.4382   0.5116 / 0.4619   0.4798 / 0.4303
```

- **Learning-rate floor.** Across the six learned-parser groups, Round 1 mean IoU-M was 0.4519 for one-step scenes and 0.4110 for two-step scenes. [Round 1](results/parser-refactor-final/matrix/round1/curriculum/summary.json)
- **Complex-scene diversity.** Round 2 raised those averages to 0.4613 and 0.4211. The soft neural group declined on both instruction lengths. [Round 2](results/parser-refactor-final/matrix/round2/curriculum/summary.json)
- **REINFORCE sampling.** Round 3 raised the learned-parser averages to 0.4716 and 0.4307. All 80 rule-parser endpoints matched Round 2 exactly. [Round 3](results/parser-refactor-final/matrix/round3/curriculum/summary.json)

The pilot and formal audits passed for each round. All 600 formal IoU-M endpoints matched the archived runs for the same group, seed and instruction length.
