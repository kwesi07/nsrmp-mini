# Reproduce the learned-parser experiments

Run the commands from the repository root. The reported matrices ran on CPU with Python 3.12.14 and PyTorch 2.15.0. The test suite can be run independently.

## Set up the environment

```sh
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run the tests

The suite covers the existing mini behavior, grammar constraints, policy gradients, phase scheduling and evaluation.

```sh
.venv/bin/python -m pytest -q --ignore=tests/test_learned_parser.py  # Expected: 70 passed
.venv/bin/python -m pytest -q tests/test_learned_parser.py  # Expected: 104 passed
.venv/bin/python -m pytest -q  # Expected: 174 passed
```

## Training datasets

All ten groups share the same training pool for a given seed. Each scene contains 3–5 objects and one instruction step.

- [Round 1](results/ten-group-followup/round1.json) generates 2,000 simple and 2,000 complex scenes per seed. The cosine learning rate has a floor of 20% of the initial rate.
- [Round 2](results/ten-group-followup/round2.json) adds 4,000 complex scenes generated with `20000 + seed`. A mixed epoch uses all 2,000 simple scenes and samples 2,000 of the 6,000 complex scenes without replacement.
- [Round 3](results/ten-group-followup/round3.json) uses the same pool and increases the learned parser's candidate count from four to eight.

## Reproduce the three ten-group matrices

Start in a fresh output directory. The reported run verified a reduced-data pilot before each formal matrix. Run the tracked Round 3 pilot and its checkpoint checks with the following commands.

```sh
out=$(mktemp -d)
cfg=results/ten-group-followup
py=.venv/bin/python
"$py" run_experiments.py --experiment 1 2 3 4 5 6 7 8 9 10 --experiment-config "$cfg/pilot3.json" --out "$out/pilot" &&
cp "$cfg/pilot3/verify.py" "$out/pilot/verify.py" &&
PYTHONPATH=. "$py" "$out/pilot/verify.py"
```

Launch the formal rounds in order after verification.

```sh
(
    set -e
    for round in 1 2 3; do
        "$py" run_experiments.py --experiment 1 2 3 4 5 6 7 8 9 10 \
            --experiment-config "$cfg/round${round}.json" --out "$out/round${round}"
    done
)
```

Each group trains independently on seeds 0–9 with up to ten single-threaded workers per round. Each model uses 1,000 test scenes per instruction length, with seed offsets of 1000 and 6000 for one-step and two-step scenes. The per-run `curriculum/group-<g>/seed-<s>/metrics.json` records IoU-M. Each `curriculum/summary.json` records all 200 endpoints, seed means, sample standard deviations and paired differences. [Conditions and results](learned_parser_result.md)
