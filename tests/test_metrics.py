import unittest

from chemicheck119_speech.metrics import (
    normalize_text,
    paired_bootstrap_cer_delta,
    paired_bootstrap_error_delta,
    score_record,
    term_presence_counts,
)


class MetricsTest(unittest.TestCase):
    def test_normalizes_spacing_case_and_punctuation(self) -> None:
        self.assertEqual("lpg 가스 누출", normalize_text(" LPG,  가스 누출! "))

    def test_scores_korean_character_and_word_errors_separately(self) -> None:
        metric = score_record("가스 누출", "가스 유출")
        self.assertEqual(1, metric.character_edits)
        self.assertEqual(4, metric.reference_characters)
        self.assertEqual(1, metric.word_edits)
        self.assertEqual(2, metric.reference_words)

    def test_term_metric_counts_false_insertions(self) -> None:
        counts = term_presence_counts(
            ["가스 누출", "일반 화재"],
            ["가스 누출", "가스 화재"],
            ["가스"],
        )
        self.assertEqual(1, counts["true_positive"])
        self.assertEqual(1, counts["false_insertion"])
        self.assertEqual(0.5, counts["precision"])

    def test_paired_bootstrap_is_deterministic(self) -> None:
        baseline = [score_record("가스", "가자"), score_record("화재", "화재")]
        hinted = [score_record("가스", "가스"), score_record("화재", "화재")]
        first = paired_bootstrap_cer_delta(baseline, hinted, samples=100, seed=119)
        second = paired_bootstrap_cer_delta(baseline, hinted, samples=100, seed=119)
        self.assertEqual(first, second)
        self.assertLess(first["estimate"], 0)

    def test_bootstrap_estimate_matches_observed_aggregate_delta(self) -> None:
        baseline = [score_record("가", "나"), score_record("가" * 100, "가" * 50)]
        hinted = [score_record("가", "가"), score_record("가" * 100, "가" * 100)]
        result = paired_bootstrap_cer_delta(baseline, hinted, samples=100, seed=119)
        expected = (
            sum(item.character_edits for item in hinted)
            - sum(item.character_edits for item in baseline)
        ) / sum(item.reference_characters for item in baseline)
        self.assertEqual(expected, result["estimate"])

    def test_generic_bootstrap_supports_paired_wer(self) -> None:
        baseline = [score_record("가스 누출", "가스 유출")]
        candidate = [score_record("가스 누출", "가스 누출")]
        result = paired_bootstrap_error_delta(
            baseline, candidate, metric="wer", samples=20, seed=119
        )
        self.assertEqual("candidate_wer_minus_baseline_wer", result["metric"])
        self.assertEqual(-0.5, result["estimate"])

    def test_generic_bootstrap_rejects_different_reference_denominators(self) -> None:
        with self.assertRaisesRegex(ValueError, "denominator"):
            paired_bootstrap_error_delta(
                [score_record("가스", "가스")],
                [score_record("가스 누출", "가스 누출")],
                metric="cer",
            )


if __name__ == "__main__":
    unittest.main()
