"""Adapter for the `normalized_poker_gpt4o` bluff-judgement dataset.

This converts the repository's real poker CSV into the canonical strict-path
schema and recovers the nuisance metadata that the P2 confound controls need.
Everything it emits is *parsed from the existing text* or computed by a
deterministic rule; nothing is invented.

Read `docs/POKER_DATASET_NOTES.md` before using the output. Two properties of the
source data change what its results can support:

1. ``response`` is the label verbatim (``"Yes"``/``"No"``), so any response-token
   analysis scores perfectly and is meaningless. Only the pre-decision path is
   usable, which is what the strict extractor enforces.
2. Every prompt ends by asking "Is this a bluff?", and the judged action has
   *already been taken*. The subject model is a classifier of someone else's
   raise, not the actor. ``prompt_variant`` exists so that the size of that
   instruction confound can be measured rather than assumed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .paper import ResearchIntegrityError

SOURCE_COLUMNS = {"statement", "label", "scenario"}

RANKS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "jack": 11, "queen": 12, "king": 13, "ace": 14}
RANK_CHARS = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
              10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
SUITS = {"spade": "s", "heart": "h", "diamond": "d", "club": "c"}

MADE_HAND_NAMES = ("high_card", "pair", "two_pair", "trips", "straight", "flush",
                   "full_house", "quads", "straight_flush")

_DECISION_RE = re.compile(r"The player decided to:\s*([A-Za-z\- ]+?)\.")
_POSITION_RE = re.compile(r"your position is ([A-Z]+),")
_HOLDING_RE = re.compile(r"your holding is \[(.+?)\]")
_FLOP_RE = re.compile(r"The flop comes ([^,]+), ([^,]+), and ([^,\.]+)")
_TURN_RE = re.compile(r"The turn comes ([^,\.]+)")
_RIVER_RE = re.compile(r"The river comes ([^,\.]+)")
_CARD_RE = re.compile(r"([A-Za-z]+)\s+[Oo]f\s+([A-Za-z]+)")
# Amounts on the decision street, e.g. "HJ bet 3 chips" / "BB raise 10 chips".
_WAGER_RE = re.compile(r"\b(bet|raise|call)\s+([0-9]*\.?[0-9]+)\s+chips")

# The question the source prompt appends; stripping it gives the neutral variant.
_JUDGE_QUESTION_RE = re.compile(r"\s*Is this a bluff\?.*$", re.DOTALL)


def parse_card(text: str) -> str | None:
    """``"King of Diamond"`` -> ``"Kd"``. Returns None if unparseable."""
    match = _CARD_RE.search(str(text).strip())
    if not match:
        return None
    rank = RANKS.get(match.group(1).lower())
    suit = SUITS.get(match.group(2).lower().rstrip("s"))
    if rank is None or suit is None:
        return None
    return RANK_CHARS[rank] + suit


def _cards(values: Iterable[str]) -> list[str]:
    parsed = [parse_card(v) for v in values]
    return [c for c in parsed if c]


def evaluate_made_hand(cards: list[str]) -> int:
    """Best made-hand category for 5-7 cards, as an index into MADE_HAND_NAMES.

    This is *made-hand strength*, not equity: a strong draw scores as high card.
    It is named accordingly so it is not mistaken for an equity estimate.
    """
    if len(cards) < 5:
        raise ResearchIntegrityError(f"Made-hand evaluation needs at least 5 cards, got {len(cards)}")
    rank_of = {v: k for k, v in RANK_CHARS.items()}
    ranks = [rank_of[c[0]] for c in cards]
    suits = [c[1] for c in cards]

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    def best_straight_high(values: set[int]) -> int | None:
        # Ace plays low for the wheel.
        extended = set(values) | ({1} if 14 in values else set())
        best = None
        for high in range(14, 4, -1):
            if all(high - offset in extended for offset in range(5)):
                best = high
                break
        return best

    flush_suit = next((s for s, n in suit_counts.items() if n >= 5), None)
    if flush_suit is not None:
        flush_ranks = {rank_of[c[0]] for c in cards if c[1] == flush_suit}
        if best_straight_high(flush_ranks) is not None:
            return MADE_HAND_NAMES.index("straight_flush")

    counts = sorted(rank_counts.values(), reverse=True)
    if counts[0] >= 4:
        return MADE_HAND_NAMES.index("quads")
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return MADE_HAND_NAMES.index("full_house")
    if flush_suit is not None:
        return MADE_HAND_NAMES.index("flush")
    if best_straight_high(set(ranks)) is not None:
        return MADE_HAND_NAMES.index("straight")
    if counts[0] == 3:
        return MADE_HAND_NAMES.index("trips")
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return MADE_HAND_NAMES.index("two_pair")
    if counts[0] == 2:
        return MADE_HAND_NAMES.index("pair")
    return MADE_HAND_NAMES.index("high_card")


def board_texture(board: list[str]) -> dict[str, Any]:
    """Mechanical board descriptors: suitedness, pairing, and connectedness."""
    if not board:
        return {"board_texture": None, "board_paired": None, "board_connected": None}
    rank_of = {v: k for k, v in RANK_CHARS.items()}
    suits = [c[1] for c in board]
    ranks = sorted({rank_of[c[0]] for c in board})
    top_suit = max(suits.count(s) for s in set(suits))
    texture = "monotone" if top_suit >= 3 else ("two_tone" if top_suit == 2 else "rainbow")
    paired = len({rank_of[c[0]] for c in board}) < len(board)
    connected = any(b - a <= 2 for a, b in zip(ranks, ranks[1:])) if len(ranks) > 1 else False
    return {"board_texture": texture, "board_paired": bool(paired), "board_connected": bool(connected)}


@dataclass(frozen=True)
class ParsedHand:
    position: str | None
    street: str
    action: str | None
    hole_cards: str | None
    board: str | None
    bet_size: float | None
    made_hand_rank: int | None
    made_hand: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position, "street": self.street, "action": self.action,
            "hole_cards": self.hole_cards, "board": self.board, "bet_size": self.bet_size,
            "made_hand_rank": self.made_hand_rank, "made_hand": self.made_hand,
        }


def parse_statement(statement: str) -> ParsedHand:
    """Recover game state from one rendered prompt.

    Fields that cannot be parsed come back as ``None`` rather than a guess, so a
    downstream control reports ``not_run`` instead of matching on invented values.
    """
    text = str(statement)
    position_match = _POSITION_RE.search(text)
    action_match = _DECISION_RE.search(text)

    hole: list[str] = []
    holding_match = _HOLDING_RE.search(text)
    if holding_match:
        hole = _cards(re.split(r"\s+and\s+", holding_match.group(1)))

    board: list[str] = []
    flop = _FLOP_RE.search(text)
    if flop:
        board.extend(_cards(flop.groups()))
    turn = _TURN_RE.search(text)
    if turn:
        board.extend(_cards([turn.group(1)]))
    river = _RIVER_RE.search(text)
    if river:
        board.extend(_cards([river.group(1)]))

    street = "river" if river else ("turn" if turn else ("flop" if flop else "preflop"))

    # Amount faced is the last wager described on the decision street.
    street_anchor = {"river": river, "turn": turn, "flop": flop}.get(street)
    tail = text[street_anchor.start():] if street_anchor else text
    tail = _JUDGE_QUESTION_RE.sub("", tail)
    wagers = _WAGER_RE.findall(tail)
    bet_size = float(wagers[-1][1]) if wagers else 0.0

    made_rank: int | None = None
    if len(hole) == 2 and len(board) >= 3:
        made_rank = evaluate_made_hand(hole + board)

    return ParsedHand(
        position=position_match.group(1) if position_match else None,
        street=street,
        action=action_match.group(1).strip().lower() if action_match else None,
        hole_cards=" ".join(hole) if hole else None,
        board=" ".join(board) if board else None,
        bet_size=bet_size,
        made_hand_rank=made_rank,
        made_hand=MADE_HAND_NAMES[made_rank] if made_rank is not None else None,
    )


def game_state_core(statement: str) -> str:
    """The game summary with the decision and question removed.

    Rows sharing a core describe the same dealt hand, so this is what defines a
    split group: without it, two rows from one hand could straddle train and test.
    """
    return str(statement).split("The player decided to")[0].strip()


def neutral_state_prompt(statement: str) -> str:
    """The prompt with the 'Is this a bluff?' question stripped.

    Used by ``prompt_variant='neutral_state'`` so the instruction confound can be
    measured: the judged action is still stated, but the target concept is not named.
    """
    without_question = _JUDGE_QUESTION_RE.sub("", str(statement))
    return without_question.strip()


def adapt_normalized_poker_csv(
    source: str | pd.DataFrame,
    *,
    prompt_variant: str = "judge_question",
    scenario: str = "poker",
    label_source: str = "dataset_provided_bluff_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert the source CSV into the canonical schema plus parsed metadata.

    ``prompt_variant``:

    - ``judge_question`` — keep the prompt as shipped, ending at the explicit
      "Is this a bluff?" question. **Instruction-confounded**: the prompt names
      the target concept.
    - ``neutral_state`` — strip the question, leaving the game summary and the
      stated action. Less confounded, but the model is no longer performing a task.

    Run both and compare; the difference is the size of the instruction confound.
    """
    if prompt_variant not in ("judge_question", "neutral_state"):
        raise ResearchIntegrityError(
            f"prompt_variant must be judge_question or neutral_state, got {prompt_variant!r}")
    df = pd.read_csv(source) if isinstance(source, str) else source.copy()
    missing = SOURCE_COLUMNS - set(df.columns)
    if missing:
        raise ResearchIntegrityError(f"Source CSV missing columns: {sorted(missing)}")

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not set(labels.dropna().astype(int)) <= {0, 1}:
        raise ResearchIntegrityError("Source labels must be binary 0/1")

    parsed = [parse_statement(s) for s in df["statement"]]
    cores = [game_state_core(s) for s in df["statement"]]
    group_ids = [hashlib.sha256(c.encode("utf-8")).hexdigest()[:16] for c in cores]

    records: list[dict[str, Any]] = []
    for position, (row, hand, group) in enumerate(zip(df.itertuples(index=False), parsed, group_ids)):
        statement = str(row.statement)
        rendered = statement if prompt_variant == "judge_question" else neutral_state_prompt(statement)
        record = {
            "sample_id": f"pk{position:06d}",
            "base_item_id": group,
            "split_group_id": group,
            "statement": rendered,
            # Kept because the schema requires it; it is the label verbatim, so
            # the pre-decision path must never read it.
            "response": getattr(row, "response", "Yes" if int(row.label) == 1 else "No"),
            "label": int(row.label),
            "scenario": scenario,
            "dataset": "normalized_poker_gpt4o",
            "prompt_variant": prompt_variant,
            "prompt_template_id": f"poker_bluff_judgement_{prompt_variant}_v1",
            "label_source": label_source,
            **hand.to_dict(),
            **board_texture(hand.board.split() if hand.board else []),
        }
        records.append(record)

    adapted = pd.DataFrame.from_records(records)
    report = build_composition_report(adapted, prompt_variant=prompt_variant)
    return adapted, report


def build_composition_report(df: pd.DataFrame, *, prompt_variant: str) -> dict[str, Any]:
    """Dataset composition, for the dataset card Reviewer A and B both asked for."""
    def dist(column: str) -> dict[str, int] | None:
        if column not in df.columns:
            return None
        return {str(k): int(v) for k, v in df[column].value_counts(dropna=False).sort_index().items()}

    response_is_label = None
    if "response" in df.columns:
        mapped = df["response"].map({"No": 0, "Yes": 1})
        response_is_label = bool(mapped.notna().all() and (mapped == df["label"]).all())

    return {
        "prompt_variant": prompt_variant,
        "n_rows": int(len(df)),
        "n_groups": int(df["split_group_id"].nunique()),
        "label_counts": dist("label"),
        "positive_rate": float(df["label"].mean()),
        "by_street": dist("street"),
        "by_position": dist("position"),
        "by_action": dist("action"),
        "by_made_hand": dist("made_hand"),
        "by_board_texture": dist("board_texture"),
        "parse_coverage": {
            column: float(df[column].notna().mean())
            for column in ("position", "street", "action", "hole_cards", "board",
                           "bet_size", "made_hand_rank") if column in df.columns
        },
        # Integrity flags a reader needs before interpreting any result.
        "integrity_flags": {
            "response_is_label_verbatim": response_is_label,
            "single_action_category": bool(df["action"].nunique(dropna=True) <= 1)
                                      if "action" in df.columns else None,
            "prompt_names_target_concept": prompt_variant == "judge_question",
            "equity_available": False,
        },
    }


__all__ = [
    "MADE_HAND_NAMES", "ParsedHand", "adapt_normalized_poker_csv",
    "board_texture", "build_composition_report", "evaluate_made_hand",
    "game_state_core", "neutral_state_prompt", "parse_card", "parse_statement",
]
