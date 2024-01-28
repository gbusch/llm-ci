from app import llm_chain


def test_france_quiz():
    app = llm_chain()
    question = "Tell me 3 facts about France."
    answer = app.invoke({"question": question})
    expected_subjects = ["Paris", "Euro", "French"]
    print(answer)
    assert any(
        subject.lower() in answer.lower() for subject in expected_subjects
    ), f"expected answer to include {expected_subjects}, but didn't"


def test_france_quiz_failonpurpose():
    app = llm_chain()
    question = "Tell me 3 facts about France."
    answer = app.invoke({"question": question})
    expected_subjects = ["xxx"]
    print(answer)
    assert any(
        subject.lower() in answer.lower() for subject in expected_subjects
    ), f"expected answer to include {expected_subjects}, but didn't"


def test_italy_quiz():
    app = llm_chain()
    question = "Tell me something about Italy."
    answer = app.invoke({"question": question})
    decline_answer = "no country, no facts"
    assert (
        decline_answer.lower() in answer.lower()
    ), f"expected app to decline with {decline_answer}, but didn't"


def test_food_quiz():
    app = llm_chain()
    question = "Give me a pizza recipe."
    answer = app.invoke({"question": question})
    decline_answer = "no country, no facts"
    assert (
        decline_answer.lower() in answer.lower()
    ), f"expected app to decline with {decline_answer}, but didn't"
