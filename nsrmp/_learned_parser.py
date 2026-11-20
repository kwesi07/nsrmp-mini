"""Lexical inputs and deterministic DSL construction for the learned parser."""
from dataclasses import dataclass

from nsrmp.data import (ACTIONS, ANY, COLOURS, REL_PHRASE, RELATIONS, SHAPES,
                        keyword_kind, noun_phrase_program)

VOCAB = ["<pad>", "put", "the", "which", "is"] + COLOURS + SHAPES + ["object"] + RELATIONS
_WORD_ID = {w: i for i, w in enumerate(VOCAB)}
ROLES = ("subj_head", "subj_rel", "subj_anchor", "ref_head", "ref_rel", "ref_anchor", "action")


@dataclass
class LexedStep:
    context_tokens: tuple
    keywords: tuple
    keyword_positions: tuple


# Issue #1 (Keywords): collect colours, shapes, object and relation phrases in order,
# without parse_roles or role inference. keywords_only follows the issue's input;
# surface_context is our extension, retaining context tokens and keyword positions.
def lex_step(clause, lexer_mode="surface_context"):
    """clause: str (one step), lexer_mode: "surface_context" | "keywords_only"
    -> LexedStep(context_tokens: tuple[str, ...], keywords: tuple[str, ...],
                 keyword_positions: tuple[int, ...])
    context_tokens[keyword_positions[i]] == keywords[i]  # 0-based
    """
    if lexer_mode not in ("surface_context", "keywords_only"):
        raise ValueError(f"unknown lexer mode: {lexer_mode!r}")
    words = clause.lower().replace(",", " ").split()
    if not words:
        raise ValueError("empty instruction step")
    context, keywords, positions = [], [], []
    pos = 0
    while pos < len(words):
        for relation, phrase in REL_PHRASE.items():
            phrase_words = phrase.split()
            if words[pos:pos + len(phrase_words)] == phrase_words:
                word, pos = relation, pos + len(phrase_words)
                kind = keyword_kind(word)
                break
        else:
            word, pos = words[pos], pos + 1
            if word in ("put", "the", "which", "is"):
                kind = None
            else:
                try:
                    kind = keyword_kind(word)
                except ValueError:
                    raise ValueError(f"unknown word or incomplete relation phrase: {word!r}") from None
                if kind == "relation":
                    raise ValueError(f"unknown word or incomplete relation phrase: {word!r}")
        if kind is not None:
            keywords.append(word)
            positions.append(len(context))
        context.append(word)
    if lexer_mode == "keywords_only":
        context, positions = keywords, list(range(len(keywords)))
    return LexedStep(tuple(context), tuple(keywords), tuple(positions))


def _build_simple_np(words, source=("Scene",)):
    colours = [w for w in words if w in COLOURS]
    nouns = [w for w in words if w in SHAPES or w == "object"]
    if len(colours) > 1 or len(nouns) != 1:
        raise ValueError(f"expected one shape or 'object' and at most one colour: {' '.join(words)}")
    return noun_phrase_program(colours[0] if colours else ANY,
                               ANY if nouns[0] == "object" else nouns[0], source)


def _build_np(head, relation, anchor):
    if len(relation) > 1 or bool(relation) != bool(anchor):
        raise ValueError(f"expected one relation with an anchor, or neither: "
                         f"relation={relation!r}, anchor={anchor!r}")
    source = ("Scene",)
    if relation:
        source = ("Relate", _build_simple_np(anchor), relation[0])
    return _build_simple_np(head, source)


# Issue #1 (Build the program): turn sampled keyword roles into a Move using
# noun_phrase_program, preserving parser.parse's DSL shape. Check complete assignments
# here; masking invalid choices before sampling belongs to the policy in Stage 2.
def role_to_program(keywords, roles):
    """keywords: Sequence[str], roles: Sequence[str] (same length)
    -> ("Move", action: str, subject_program: tuple, reference_program: tuple)
    roles[i] in ROLES; action in ACTIONS
    """
    if len(keywords) != len(roles):
        raise ValueError("each keyword needs exactly one role")
    groups = {role: [] for role in ROLES}
    for word, role in zip(keywords, roles):
        if role not in groups:
            raise ValueError(f"unknown role: {role!r}")
        if word in COLOURS + SHAPES + ["object"]:
            if role not in ("subj_head", "subj_anchor", "ref_head", "ref_anchor"):
                raise ValueError("noun concepts need a noun role")
        elif word in RELATIONS:
            if role not in ("subj_rel", "ref_rel", "action"):
                raise ValueError("relations need a relation or action role")
        else:
            raise ValueError(f"unknown keyword: {word!r}")
        groups[role].append(word)
    if len(groups["action"]) != 1:
        raise ValueError("a Move needs exactly one action")
    subject = _build_np(groups["subj_head"], groups["subj_rel"], groups["subj_anchor"])
    reference = _build_np(groups["ref_head"], groups["ref_rel"], groups["ref_anchor"])
    return ("Move", ACTIONS[RELATIONS.index(groups["action"][0])], subject, reference)


_NOUN_ROLES = {"subj_head": 0, "subj_anchor": 1, "ref_head": 2, "ref_anchor": 3}
_RELATION_ROLES = {"subj_rel": 0, "ref_rel": 1}


class Grammar:
    def __init__(self, keywords):
        self._kinds = tuple(keyword_kind(keyword) for keyword in keywords)
        self._prefix = ()
        self._state = ((0, 0, 0, 0), (False, False), False)
        self._memo = {}

    @staticmethod
    def _advance_kind(state, kind, role):
        nouns, relations, action = state
        if kind in ("colour", "noun") and role in _NOUN_ROLES:
            slot, bit = _NOUN_ROLES[role], 1 if kind == "colour" else 2
            if not nouns[slot] & bit:
                updated = list(nouns)
                updated[slot] |= bit
                return tuple(updated), relations, action
        elif kind == "relation":
            if role == "action" and not action:
                return nouns, relations, True
            if role in _RELATION_ROLES:
                slot = _RELATION_ROLES[role]
                if not relations[slot]:
                    updated = list(relations)
                    updated[slot] = True
                    return nouns, tuple(updated), action
        return None

    # Recursively find whether the remaining keywords can complete a valid Move.
    def _can_complete(self, state, position):
        key = (state, position)
        if key in self._memo:
            return self._memo[key]
        nouns, relations, action = state
        occupied = sum(noun.bit_count() for noun in nouns) + sum(relations) + action
        if len(self._kinds) - position > 11 - occupied:
            possible = False
        elif position == len(self._kinds):
            possible = (action and bool(nouns[0] & 2) and bool(nouns[2] & 2)
                        and all(bool(nouns[slot] & 2) if relation else nouns[slot] == 0
                                for slot, relation in zip((1, 3), relations)))
        else:
            possible = False
            for role in ROLES:
                next_state = self._advance_kind(state, self._kinds[position], role)
                if next_state is not None and self._can_complete(next_state, position + 1):
                    possible = True
                    break
        self._memo[key] = possible
        return possible

    # Issue #1 (Build the program): mask roles that cannot finish the DSL.
    # _can_complete checks each candidate for a valid remaining role assignment.
    # A Grammar instance receives one serially growing sampled role prefix.
    def valid_roles(self, prefix_roles):
        """prefix_roles: serial sampled prefix -> bool[7] for the next keyword."""
        prefix = tuple(prefix_roles)
        if len(prefix) >= len(self._kinds):
            raise ValueError(f"expected a role prefix shorter than the keywords: {prefix!r}")
        if (len(prefix) < len(self._prefix) or len(prefix) > len(self._prefix) + 1
                or prefix[:len(self._prefix)] != self._prefix):
            raise ValueError(f"expected a sequential role prefix extending "
                             f"{self._prefix!r}: {prefix!r}")
        if len(prefix) > len(self._prefix):
            index = len(self._prefix)
            next_state = self._advance_kind(self._state, self._kinds[index], prefix[-1])
            if next_state is None:
                raise ValueError(f"expected an available role at keyword {index}: "
                                 f"{prefix[-1]!r} in {prefix!r}")
            self._state, self._prefix = next_state, prefix
        position = len(prefix)
        choices = []
        for role in ROLES:
            next_state = self._advance_kind(self._state, self._kinds[position], role)
            choices.append(next_state is not None and self._can_complete(next_state, position + 1))
        return tuple(choices)