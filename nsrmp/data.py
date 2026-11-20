"""The 2D blocks world and instruction generator.

Paper link: Sec. IV-A / Table I describe the real (image) dataset. Here we replace images by
random "visual features" of gold boxes (the paper's gold-BB setting), and generate the
instructions, gold programs (Sec. III-B DSL) and ground-truth final states ourselves.
"""
from dataclasses import dataclass

import numpy as np

COLOURS = ["red", "green", "blue", "yellow", "cyan", "white"]
SHAPES = ["cube", "dice", "lego"]
RELATIONS = ["left", "right", "front", "behind"]
ACTIONS = ["MoveLeft", "MoveRight", "MoveFront", "MoveBehind"]  # ACTIONS[i] <-> RELATIONS[i]
ANY = "any"  # wildcard concept: "the cube" has colour ANY, "the red object" has shape ANY
CONCEPTS = COLOURS + SHAPES + [ANY]

FEATURE_DIM = 32
FEATURE_NOISE = 0.1
GAP = 0.02  # distance between subject and reference after a move

REL_PHRASE = {"left": "to the left of", "right": "to the right of",
              "front": "in front of", "behind": "behind"}


def keyword_kind(keyword):
    if keyword in COLOURS:
        return "colour"
    if keyword in SHAPES or keyword == "object":
        return "noun"
    if keyword in RELATIONS:
        return "relation"
    raise ValueError(f"unknown keyword: {keyword!r}")


@dataclass(frozen=True)
class Obj:
    colour: str
    shape: str
    box: tuple  # (x1, y1, x2, y2) in [0, 1]^2; +y is "front"


def feature_matrix():
    """Fixed random projection P (32 x 9), shared by train and test sets."""
    return np.random.RandomState(12345).randn(FEATURE_DIM, len(COLOURS) + len(SHAPES))


def gen_scene(rng, n):
    """Sample n non-overlapping boxes with random colours and shapes."""
    for _ in range(1000):  # restart if the square is too crowded
        objs = []
        for _ in range(200):
            if len(objs) == n:
                break
            w, h = rng.uniform(0.08, 0.12, size=2)
            x1, y1 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            box = (x1, y1, x1 + w, y1 + h)
            if all(not _overlap(box, o.box) for o in objs):
                objs.append(Obj(COLOURS[rng.randint(len(COLOURS))],
                                SHAPES[rng.randint(len(SHAPES))], box))
        if len(objs) == n:
            return objs
    raise RuntimeError("could not place objects")


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def holds(rel, box_i, box_j):
    """True if object i is `rel` of object j (compares centres; +y is front)."""
    (xi, yi), (xj, yj) = centre(box_i), centre(box_j)
    return {"left": xi < xj, "right": xi > xj, "front": yi > yj, "behind": yi < yj}[rel]


def apply_action(action, subj_box, ref_box):
    """Ground-truth Action Simulator: place the subject GAP away from the reference."""
    w, h = subj_box[2] - subj_box[0], subj_box[3] - subj_box[1]
    cx, cy = centre(ref_box)
    if action == "MoveLeft":
        cx = ref_box[0] - GAP - w / 2
    elif action == "MoveRight":
        cx = ref_box[2] + GAP + w / 2
    elif action == "MoveFront":
        cy = ref_box[3] + GAP + h / 2
    elif action == "MoveBehind":
        cy = ref_box[1] - GAP - h / 2
    else:
        raise ValueError(action)
    x1 = min(max(cx - w / 2, 0.0), 1.0 - w)  # shift (not shrink) to stay inside
    y1 = min(max(cy - h / 2, 0.0), 1.0 - h)
    return (x1, y1, x1 + w, y1 + h)


# ---------------------------------------------------------------------------------------
# Programs and crisp (non-neural) execution. Operators follow the paper's DSL:
#   ("Scene",)  ("Filter", set, concept)  ("Relate", unique_obj, rel)  ("Unique", set)
#   ("Move", action, subject_prog, reference_prog)
# A missing colour/shape is the wildcard concept ANY, so every noun phrase has the same shape.
# ---------------------------------------------------------------------------------------
def noun_phrase_program(colour, shape, source=("Scene",)):
    return ("Unique", ("Filter", ("Filter", source, colour), shape))


def _crisp_set(prog, objs):
    op = prog[0]
    if op == "Scene":
        return np.ones(len(objs), dtype=bool)
    if op == "Unique":
        return _crisp_set(prog[1], objs)
    if op == "Filter":
        keep = _crisp_set(prog[1], objs)
        concept = prog[2]
        if concept == ANY:
            return keep
        return keep & np.array([concept in (o.colour, o.shape) for o in objs])
    if op == "Relate":
        anchor = _crisp_set(prog[1], objs)
        return np.array([any(holds(prog[2], objs[i].box, objs[j].box)
                             for j in np.flatnonzero(anchor) if j != i)
                         for i in range(len(objs))])
    raise ValueError(op)


def crisp_select(prog, objs):
    """Indices of the objects a noun-phrase program selects (should be exactly one)."""
    return np.flatnonzero(_crisp_set(prog, objs))


def _attribute_options(rng, colour, shape):
    options = [(colour, shape), (colour, ANY), (ANY, shape)]
    return [options[i] for i in rng.permutation(3)]


def _text_np(colour, shape):
    return " ".join(["the"] + ([colour] if colour != ANY else []) + [shape if shape != ANY else "object"])


def _simple_np(rng, objs, idx):
    """A colour/shape description that picks out objs[idx] and nothing else."""
    for colour, shape in _attribute_options(rng, objs[idx].colour, objs[idx].shape):
        prog = noun_phrase_program(colour, shape)
        if len(crisp_select(prog, objs)) == 1:
            return _text_np(colour, shape), prog
    return None


def _complex_np(rng, objs, idx):
    """'the lego which is behind the green dice': ambiguous alone, unique via the relation."""
    for a in rng.permutation(len(objs)):
        if a == idx:
            continue
        anchor = _simple_np(rng, objs, a)
        rels = [r for r in RELATIONS if holds(r, objs[idx].box, objs[a].box)]
        if anchor is None or not rels:
            continue
        rel = rels[rng.randint(len(rels))]
        for colour, shape in _attribute_options(rng, objs[idx].colour, objs[idx].shape):
            prog = noun_phrase_program(colour, shape, source=("Relate", anchor[1], rel))
            if (len(crisp_select(noun_phrase_program(colour, shape), objs)) >= 2
                    and crisp_select(prog, objs).tolist() == [idx]):
                return f"{_text_np(colour, shape)} which is {REL_PHRASE[rel]} {anchor[0]}", prog
    return None


def _gen_step(rng, objs, is_complex):
    s, r = rng.choice(len(objs), size=2, replace=False)
    mode = ["subject", "ref", "both"][rng.randint(3)] if is_complex else "none"
    subj = (_complex_np if mode in ("subject", "both") else _simple_np)(rng, objs, s)
    ref = (_complex_np if mode in ("ref", "both") else _simple_np)(rng, objs, r)
    if subj is None or ref is None:
        return None
    a = rng.randint(len(ACTIONS))
    text = f"put {subj[0]} {REL_PHRASE[RELATIONS[a]]} {ref[0]}"
    return text, ("Move", ACTIONS[a], subj[1], ref[1]), int(s), int(r)


@dataclass
class Example:
    objs: list
    feats: np.ndarray       # (n, 32) visual features
    text: str
    program: tuple          # gold program: one Move per step
    subjects: list          # gold subject index per step
    refs: list              # gold reference index per step
    final_boxes: np.ndarray  # (n, 4) boxes after the last step
    is_complex: bool

    @property
    def n_steps(self):
        return len(self.program)


def object_features(rng, objs, noise=FEATURE_NOISE):
    onehots = np.zeros((len(objs), len(COLOURS) + len(SHAPES)))
    for i, o in enumerate(objs):
        onehots[i, COLOURS.index(o.colour)] = 1
        onehots[i, len(COLOURS) + SHAPES.index(o.shape)] = 1
    return onehots @ feature_matrix().T + noise * rng.randn(len(objs), FEATURE_DIM)


def make_example(rng, n_objects, complex_flags, feature_noise=FEATURE_NOISE):
    """One scene + instruction with len(complex_flags) steps. Steps see the updated scene."""
    for _ in range(2000):
        objs = gen_scene(rng, n_objects)
        boxes = [o.box for o in objs]
        texts, program, subjects, refs = [], [], [], []
        for flag in complex_flags:
            now = [Obj(o.colour, o.shape, b) for o, b in zip(objs, boxes)]
            step = next(filter(None, (_gen_step(rng, now, flag) for _ in range(30))), None)
            if step is None:
                break
            text, move, s, r = step
            boxes[s] = apply_action(move[1], boxes[s], boxes[r])
            texts.append(text); program.append(move); subjects.append(s); refs.append(r)
        else:
            return Example(objs, object_features(rng, objs, feature_noise), ", then ".join(texts),
                           tuple(program), subjects, refs, np.array(boxes), any(complex_flags))
    raise RuntimeError("could not generate an example")


def make_dataset(count, seed, n_objects=(3, 5), n_steps=(1, 1), p_complex=0.5, feature_noise=FEATURE_NOISE):
    """`count` examples; ranges are inclusive. The first step is complex in exactly
    round(p_complex * count) examples; later steps are complex with probability p_complex.
    `feature_noise` (sigma of the feature noise) changes only the features, not the scenes."""
    rng = np.random.RandomState(seed)
    first = rng.permutation([True] * round(p_complex * count) + [False] * (count - round(p_complex * count)))
    examples = []
    for i in range(count):
        n = rng.randint(n_objects[0], n_objects[1] + 1)
        steps = rng.randint(n_steps[0], n_steps[1] + 1)
        flags = [bool(first[i])] + [bool(rng.rand() < p_complex) for _ in range(steps - 1)]
        examples.append(make_example(rng, n, flags, feature_noise))
    return examples


def training_examples(config, seed):
    examples = make_dataset(config["train_examples"], seed)
    extra = config.get("extra_complex_examples", 0)
    if extra:
        examples += make_dataset(extra, config["extra_train_seed"] + seed, p_complex=1.0)
    return examples
