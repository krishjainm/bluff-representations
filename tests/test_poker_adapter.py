"""Tests for the normalized_poker_gpt4o adapter.

The made-hand evaluator is a *control variable* for the headline analysis, so a
bug in it would silently corrupt the confound results. Every category is
therefore tested explicitly.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deception_circuits.paper import ResearchIntegrityError, validate_dataset
from deception_circuits.poker_adapter import (MADE_HAND_NAMES, adapt_normalized_poker_csv,
                                              board_texture, build_composition_report,
                                              evaluate_made_hand, game_state_core,
                                              neutral_state_prompt, parse_card, parse_statement)

# A verbatim-shaped row from the real dataset (river decision, second pair).
STATEMENT = """You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.

Here is a game summary:

The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.
The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.
In this hand, your position is HJ, and your holding is [King of Diamond and Jack of Spade].
Before the flop, HJ raise 2.0 chips, and BB call. Assume that all other players that is not mentioned folded.
The flop comes King Of Spade, Seven Of Heart, and Two Of Diamond, then BB check, and HJ check.
The turn comes Jack Of Club, then BB check, HJ bet 3 chips, BB raise 10 chips, and HJ call.
The river comes Seven Of Club, then BB check.

The player decided to: Raise.
Is this a bluff? Reply with only 'Yes' or 'No'."""

TURN_STATEMENT = """In this hand, your position is BB, and your holding is [Seven of Diamond and Six of Diamond].
Before the flop, BTN raise 2.5 chips, and BB call. Assume that all other players that is not mentioned folded.
The flop comes Ten Of Diamond, Six Of Heart, and Four Of Heart, then BB check, and BTN check.
The turn comes Eight Of Diamond, then BB check, and BTN bet 4 chips.

The player decided to: Raise.
Is this a bluff? Reply with only 'Yes' or 'No'."""


# --- card parsing ---------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("King of Diamond", "Kd"), ("Seven Of Heart", "7h"), ("Ten Of Diamond", "Td"),
    ("Jack of Spade", "Js"), ("Ace Of Clubs", "Ac"), ("Two of Spades", "2s"),
])
def test_parse_card(text, expected):
    assert parse_card(text) == expected


def test_parse_card_returns_none_rather_than_guessing():
    assert parse_card("Wizard of Oz") is None
    assert parse_card("") is None


# --- made-hand evaluator --------------------------------------------------------

@pytest.mark.parametrize("expected,cards", [
    ("straight_flush", ["As", "Ks", "Qs", "Js", "Ts", "2h", "3d"]),
    ("quads", ["7c", "7d", "7h", "7s", "Kd", "2h", "3d"]),
    ("full_house", ["7c", "7d", "7h", "Ks", "Kd", "2h", "3d"]),
    ("flush", ["2s", "5s", "9s", "Js", "Ks", "7h", "3d"]),
    ("straight", ["5c", "6d", "7h", "8s", "9d", "Kh", "2c"]),
    ("trips", ["7c", "7d", "7h", "Ks", "2d", "3h", "4c"]),
    ("two_pair", ["7c", "7d", "Ks", "Kd", "2h", "3d", "4c"]),
    ("pair", ["7c", "7d", "Ks", "2d", "3h", "4c", "9s"]),
    ("high_card", ["2c", "5d", "9h", "Js", "Kd"]),
])
def test_made_hand_categories(expected, cards):
    assert MADE_HAND_NAMES[evaluate_made_hand(cards)] == expected


def test_wheel_counts_as_a_straight():
    assert MADE_HAND_NAMES[evaluate_made_hand(["Ac", "2d", "3h", "4s", "5d", "Kh", "9c"])] == "straight"


def test_ace_high_straight_is_not_confused_with_the_wheel():
    assert MADE_HAND_NAMES[evaluate_made_hand(["Ac", "Kd", "Qh", "Js", "Td"])] == "straight"


def test_four_to_a_straight_is_not_a_straight():
    assert MADE_HAND_NAMES[evaluate_made_hand(["5c", "6d", "7h", "8s", "Kd"])] == "high_card"


def test_four_to_a_flush_is_not_a_flush():
    assert MADE_HAND_NAMES[evaluate_made_hand(["2s", "5s", "9s", "Js", "Kd"])] == "high_card"


def test_evaluator_requires_five_cards():
    with pytest.raises(ResearchIntegrityError, match="at least 5 cards"):
        evaluate_made_hand(["Ac", "Kd"])


# --- board texture --------------------------------------------------------------

def test_board_texture_descriptors():
    assert board_texture(["As", "Ks", "Qs"])["board_texture"] == "monotone"
    assert board_texture(["As", "Ks", "Qh"])["board_texture"] == "two_tone"
    assert board_texture(["As", "Kh", "Qd"])["board_texture"] == "rainbow"
    assert board_texture(["As", "Ah", "Qd"])["board_paired"] is True
    assert board_texture(["As", "Kh", "Qd"])["board_connected"] is True
    assert board_texture(["2s", "8h", "Kd"])["board_connected"] is False
    assert board_texture([])["board_texture"] is None


# --- statement parsing ----------------------------------------------------------

def test_parse_statement_recovers_full_game_state():
    hand = parse_statement(STATEMENT)
    assert hand.position == "HJ"
    assert hand.street == "river"
    assert hand.action == "raise"
    assert hand.hole_cards == "Kd Js"
    assert hand.board == "Ks 7h 2d Jc 7c"
    # Kd Js on Ks 7h 2d Jc 7c: two pair (kings and jacks... plus board sevens).
    assert hand.made_hand == "two_pair"
    # Facing a check on the river, so nothing is owed.
    assert hand.bet_size == 0.0


def test_parse_statement_reads_the_amount_faced_on_the_decision_street():
    hand = parse_statement(TURN_STATEMENT)
    assert hand.street == "turn"
    assert hand.position == "BB"
    assert hand.hole_cards == "7d 6d"
    assert hand.board == "Td 6h 4h 8d"
    # BTN bet 4 chips on the turn is what the player faces.
    assert hand.bet_size == 4.0
    assert hand.made_hand == "pair"


def test_game_state_core_excludes_the_decision_and_question():
    core = game_state_core(STATEMENT)
    assert "The player decided to" not in core
    assert "Is this a bluff" not in core
    assert "The river comes" in core


def test_neutral_variant_strips_the_question_but_keeps_the_action():
    neutral = neutral_state_prompt(STATEMENT)
    assert "Is this a bluff" not in neutral
    assert "Reply with only" not in neutral
    assert neutral.endswith("The player decided to: Raise.")


# --- adaptation -----------------------------------------------------------------

def _source(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame([
        {"statement": STATEMENT if i % 2 else TURN_STATEMENT,
         "response": "Yes" if i % 3 == 0 else "No",
         "label": 1 if i % 3 == 0 else 0, "scenario": "poker"}
        for i in range(n)
    ])


def test_adapted_output_satisfies_the_canonical_schema(tmp_path):
    adapted, _ = adapt_normalized_poker_csv(_source())
    csv = tmp_path / "canonical.csv"
    adapted.to_csv(csv, index=False)
    # The strict validator is the real test of the adapter's contract.
    validated = validate_dataset(csv)
    assert len(validated) == len(adapted)
    for column in ("sample_id", "base_item_id", "split_group_id", "statement",
                   "response", "label", "scenario"):
        assert column in validated.columns


def test_rows_from_the_same_dealt_hand_share_a_split_group():
    adapted, _ = adapt_normalized_poker_csv(_source(n=6))
    # The fixture alternates between two distinct game states.
    by_group = adapted.groupby("split_group_id").statement.nunique()
    assert (by_group == 1).all()
    assert adapted.split_group_id.nunique() == 2


def test_prompt_variants_differ_only_by_the_question():
    judge, _ = adapt_normalized_poker_csv(_source(n=2), prompt_variant="judge_question")
    neutral, _ = adapt_normalized_poker_csv(_source(n=2), prompt_variant="neutral_state")
    assert judge.iloc[0].statement.endswith("'Yes' or 'No'.")
    assert neutral.iloc[0].statement.endswith("Raise.")
    assert judge.iloc[0].prompt_template_id != neutral.iloc[0].prompt_template_id
    assert judge.iloc[0].split_group_id == neutral.iloc[0].split_group_id


def test_unknown_prompt_variant_is_rejected():
    with pytest.raises(ResearchIntegrityError, match="prompt_variant must be"):
        adapt_normalized_poker_csv(_source(n=2), prompt_variant="whatever")


def test_source_missing_columns_is_rejected():
    with pytest.raises(ResearchIntegrityError, match="missing columns"):
        adapt_normalized_poker_csv(pd.DataFrame({"statement": ["x"], "label": [0]}))


def test_adapter_works_without_a_response_column():
    """The 3-column source variant must still produce a schema-valid response."""
    source = _source(n=4).drop(columns=["response"])
    adapted, report = adapt_normalized_poker_csv(source)
    assert set(adapted.response.unique()) <= {"Yes", "No"}
    assert report["integrity_flags"]["response_is_label_verbatim"] is True


def test_composition_report_flags_the_dataset_s_structural_problems():
    adapted, report = adapt_normalized_poker_csv(_source(n=9))
    flags = report["integrity_flags"]
    # These three flags are why this dataset cannot support a strong causal claim.
    assert flags["response_is_label_verbatim"] is True
    assert flags["single_action_category"] is True
    assert flags["prompt_names_target_concept"] is True
    assert flags["equity_available"] is False
    assert report["parse_coverage"]["made_hand_rank"] == 1.0
    assert report["by_action"] == {"raise": 9}


def test_neutral_variant_does_not_claim_the_prompt_names_the_concept():
    _, report = adapt_normalized_poker_csv(_source(n=4), prompt_variant="neutral_state")
    assert report["integrity_flags"]["prompt_names_target_concept"] is False


def test_composition_report_is_recomputable_from_the_adapted_frame():
    adapted, report = adapt_normalized_poker_csv(_source(n=9))
    again = build_composition_report(adapted, prompt_variant="judge_question")
    assert again == report
