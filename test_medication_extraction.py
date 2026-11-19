from graph_app import extract_medications


def test_extract_two_medications():
    result = extract_medications(
        "Can I take aspirin with warfarin?"
    )

    assert result == ["aspirin", "warfarin"]


def test_extract_three_medications():
    result = extract_medications(
        "Can I take aspirin with warfarin and simvastatin?"
    )

    assert len(result) == 3
    assert set(result) == {
        "aspirin",
        "warfarin",
        "simvastatin",
    }


def test_extract_medication_aliases():
    result = extract_medications(
        "Can I take Tylenol with Coumadin?"
    )

    assert set(result) == {
        "acetaminophen",
        "warfarin",
    }