from data.nli_examples import LANGUAGE_NAMES, build_examples


def test_dataset_is_balanced_across_labels():
    examples = build_examples()
    from collections import Counter

    counts = Counter(e.label for e in examples)
    assert set(counts) == {"entailment", "neutral", "contradiction"}
    assert len(set(counts.values())) == 1, f"Labels are not balanced: {counts}"


def test_dataset_covers_expected_languages():
    examples = build_examples()
    languages = {e.language for e in examples}
    assert languages == set(LANGUAGE_NAMES)


def test_every_example_has_nonempty_text():
    examples = build_examples()
    for e in examples:
        assert e.premise.strip()
        assert e.hypothesis.strip()
        assert e.premise != e.hypothesis


def test_dataset_size_is_substantial():
    """Not a huge dataset (this is a hand-curated offline demo, not the
    real ~12,000-pair Kaggle set), but should be large enough for the
    classifier to have more than a handful of examples per language.
    """
    examples = build_examples()
    assert len(examples) >= 150
