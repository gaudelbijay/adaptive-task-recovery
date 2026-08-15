"""D-079: H4's first real comparative test -- factorized versus monolithic
instruction representations on unseen goal compositions. See
`atr.language.compositional_generalization`'s module docstring for why this
tests the language axis (`atr.evaluation.splits`'s instruction-composition
split, D-044) rather than the intervention-mechanism axis: no env in this
project has two goals each independently threatened by a different
intervention, so there's no real goal-by-change cross product to hold a
combination out from at the simulator level, but there is one in language
(known objects recombined into instructions never literally seen together).

No mani_skill dependency -- this stays entirely at the parsed-instruction
level, same as `instruction_parser.py`/`goal_graph.py` themselves.
"""

from atr.language.compositional_generalization import (
    canonical_eval_cases,
    compositional_matrix_cases,
    compare_factorized_vs_monolithic,
    full_role_matrix_cases,
    monolithic_retrieval_predict,
    monolithic_predict,
    semantics,
    train_monolithic_lookup,
    train_monolithic_retriever,
)
from atr.language.goal_graph import canonical_example


class TestCanonicalEvalCases:
    def test_covers_all_three_splits(self):
        cases = canonical_eval_cases()
        splits = {case.split for case in cases}
        assert splits == {"train", "held_out_paraphrase", "held_out_composition"}

    def test_exactly_one_train_case(self):
        """The monolithic baseline's entire "training set" -- if this were
        more than one case, or the wrong one, the baseline comparison below
        wouldn't mean what it claims to."""
        cases = canonical_eval_cases()
        assert sum(case.split == "train" for case in cases) == 1


class TestMonolithicBaselineHasNoGeneralizationMechanism:
    def test_matches_the_real_parser_on_the_exact_trained_string(self):
        cases = [c for c in canonical_eval_cases() if c.split == "train"]
        lookup = train_monolithic_lookup(cases)
        train_case = cases[0]
        assert monolithic_predict(lookup, train_case.instruction_text) == train_case.ground_truth

    def test_returns_none_for_any_untrained_string(self):
        cases = [c for c in canonical_eval_cases() if c.split == "train"]
        lookup = train_monolithic_lookup(cases)
        assert monolithic_predict(lookup, "a sentence never in the training set") is None
        held_out = [c for c in canonical_eval_cases() if c.split != "train"]
        for case in held_out:
            assert monolithic_predict(lookup, case.instruction_text) is None


class TestFactorizedVersusMonolithicComparison:
    def test_factorized_parser_generalizes_across_every_split(self):
        result = compare_factorized_vs_monolithic()
        for split, total in result.total_by_split.items():
            assert result.factorized_correct_by_split[split] == total

    def test_monolithic_baseline_only_ever_matches_the_train_split(self):
        result = compare_factorized_vs_monolithic()
        assert result.monolithic_correct_by_split.get("train", 0) == result.total_by_split["train"]
        assert result.monolithic_correct_by_split.get("held_out_paraphrase", 0) == 0
        assert result.monolithic_correct_by_split.get("held_out_composition", 0) == 0

    def test_surface_retriever_handles_paraphrases_but_not_new_composition(self):
        """The stronger baseline removes D-079's main confound: unseen text
        alone is not enough to make it fail. It transfers the training graph
        across all same-semantics paraphrases, but cannot construct the novel
        held-out composition because its outputs are indivisible graphs."""
        result = compare_factorized_vs_monolithic()
        assert result.retrieval_correct_by_split["train"] == result.total_by_split["train"]
        assert (
            result.retrieval_correct_by_split["held_out_paraphrase"]
            == result.total_by_split["held_out_paraphrase"]
        )
        assert result.retrieval_correct_by_split["held_out_composition"] == 0

    def test_retriever_is_fitted_only_on_training_cases(self):
        train_cases = [c for c in canonical_eval_cases() if c.split == "train"]
        model = train_monolithic_retriever(train_cases)
        held_out = next(c for c in canonical_eval_cases() if c.split == "held_out_paraphrase")
        assert monolithic_retrieval_predict(model, held_out.instruction_text) == held_out.ground_truth

    def test_semantics_helper_matches_canonical_example_with_itself(self):
        """Sanity check on the comparison metric itself, not the models --
        the same graph must equal itself under `semantics()`."""
        assert semantics(canonical_example()) == semantics(canonical_example())


class TestLargerCompositionalMatrix:
    def test_has_multiple_disjoint_train_and_held_out_graphs(self):
        cases = compositional_matrix_cases()
        train = [case for case in cases if case.split == "train"]
        held_out = [case for case in cases if case.split == "held_out_composition"]
        assert len(train) == 4
        assert len(held_out) == 4
        assert {case.ground_truth for case in train}.isdisjoint(
            {case.ground_truth for case in held_out}
        )

    def test_factorized_parser_handles_every_held_out_recombination(self):
        result = compare_factorized_vs_monolithic(compositional_matrix_cases())
        assert result.factorized_correct_by_split["train"] == 4
        assert result.factorized_correct_by_split["held_out_composition"] == 4

    def test_whole_graph_retriever_cannot_construct_held_out_graphs(self):
        result = compare_factorized_vs_monolithic(compositional_matrix_cases())
        assert result.retrieval_correct_by_split["train"] == 4
        assert result.retrieval_correct_by_split["held_out_composition"] == 0


class TestFullRoleMatrix:
    """D-117: the exhaustive version of TestLargerCompositionalMatrix above --
    every possible goal-pair over the object pool (180 cases), not 4 hand-picked
    ones, with a checked guarantee that held-out goal-pairs never appeared as a
    goal-pair during training."""

    def test_no_goal_pair_shared_between_train_and_held_out(self):
        cases = full_role_matrix_cases()
        train = [case for case in cases if case.split == "train"]
        held_out = [case for case in cases if case.split == "held_out_composition"]
        assert len(train) == 96
        assert len(held_out) == 84

        def goal_pair(case):
            return frozenset(obj for _, obj in case.ground_truth[0])

        train_pairs = {goal_pair(case) for case in train}
        held_out_pairs = {goal_pair(case) for case in held_out}
        assert train_pairs.isdisjoint(held_out_pairs)

    def test_factorized_parser_handles_the_full_matrix(self):
        result = compare_factorized_vs_monolithic(full_role_matrix_cases())
        assert result.factorized_correct_by_split["train"] == 96
        assert result.factorized_correct_by_split["held_out_composition"] == 84

    def test_monolithic_baselines_still_fail_every_held_out_case(self):
        result = compare_factorized_vs_monolithic(full_role_matrix_cases())
        assert result.monolithic_correct_by_split["held_out_composition"] == 0
        assert result.retrieval_correct_by_split["held_out_composition"] == 0
