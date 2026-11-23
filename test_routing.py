from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage

import graph_app


def make_fake_llm(response=None, error=None):
    """Create a predictable fake LLM for agent_node tests."""

    fake_llm = Mock()

    if error is not None:
        fake_llm.invoke.side_effect = error
    else:
        fake_llm.invoke.return_value = response

    return fake_llm


def make_tool_call(name, args=None):
    """Create a valid model-generated tool call."""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {},
                "id": "test-tool-call-1",
                "type": "tool_call",
            }
        ],
    )


def test_agent_node_preserves_model_selected_drug_interaction_tool(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    question = "Does warfarin interact with aspirin?"

    fake_llm = make_fake_llm(
        response=make_tool_call(
            "check_drug_interactions",
            {
                "medications": ["warfarin", "aspirin"],
            },
        )
    )

    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {"messages": [HumanMessage(content=question)]}
    )

    tool_call = result["messages"][-1].tool_calls[0]

    assert tool_call["name"] == "check_drug_interactions"
    assert tool_call["args"]["medications"] == [
        "warfarin",
        "aspirin",
    ]
    fake_llm.invoke.assert_called_once()


def test_evidence_request_preserves_original_question(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    question = (
        "Find clinical evidence about iron deficiency anemia "
        "in adolescents."
    )

    fake_llm = make_fake_llm(
        response=make_tool_call(
            "search_medical_evidence",
            {"query": "iron deficiency anemia"},
        )
    )

    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {"messages": [HumanMessage(content=question)]}
    )

    tool_call = result["messages"][-1].tool_calls[0]
    args = tool_call["args"]

    assert tool_call["name"] == "search_medical_evidence"
    assert args["original_question"] == question
    assert args["query"] == "iron deficiency anemia"
    assert "patient_age" not in args


def test_evidence_request_adds_patient_context(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    question = (
        "Find medical evidence about anemia "
        "in a pregnant 16-year-old patient."
    )

    monkeypatch.setattr(
        graph_app,
        "_extract_patient_age",
        lambda question: 16,
    )

    monkeypatch.setattr(
        graph_app,
        "_extract_patient_pregnancy_status",
        lambda question: "pregnant",
    )

    fake_llm = make_fake_llm(
        response=make_tool_call(
            "search_medical_evidence",
            {"query": "anemia"},
        )
    )

    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {"messages": [HumanMessage(content=question)]}
    )

    args = result["messages"][-1].tool_calls[0]["args"]

    assert args["patient_age"] == 16
    assert args["patient_pregnancy_status"] == "pregnant"
    assert args["original_question"] == question


def test_evidence_request_generates_query_when_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    question = (
        "Find diagnostic accuracy studies about deep learning "
        "for diabetic retinopathy."
    )

    expected_query = (
        "diabetic retinopathy diagnostic accuracy"
    )

    monkeypatch.setattr(
        graph_app,
        "extract_clinical_research_topic",
        lambda question: expected_query,
    )

    fake_llm = make_fake_llm(
        response=make_tool_call(
            "search_medical_evidence",
            {},
        )
    )

    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {"messages": [HumanMessage(content=question)]}
    )

    args = result["messages"][-1].tool_calls[0]["args"]

    assert args["query"] == expected_query
    assert args["original_question"] == question


def test_agent_node_returns_direct_answer_without_tool_calls(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    expected_answer = "Warfarin is an anticoagulant."

    fake_llm = make_fake_llm(
        response=AIMessage(content=expected_answer)
    )

    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {
            "messages": [
                HumanMessage(content="What is warfarin?")
            ]
        }
    )

    response = result["messages"][-1]

    assert response.content == expected_answer
    assert not response.tool_calls
    fake_llm.invoke.assert_called_once()


def test_agent_node_handles_llm_failure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_llm = make_fake_llm(
        error=RuntimeError("Test model failure")
    )

    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {
            "messages": [
                HumanMessage(content="Explain this medical case.")
            ]
        }
    )

    response = result["messages"][-1]

    assert "could not generate a response" in response.content
    assert "RuntimeError" in response.content
    assert "Test model failure" in response.content


def test_agent_node_handles_empty_question(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_llm = make_fake_llm()
    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node({"messages": []})

    response = result["messages"][-1]

    assert response.content == "Please enter a question."
    fake_llm.invoke.assert_not_called()


def test_agent_node_handles_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    fake_llm = make_fake_llm()
    monkeypatch.setattr(graph_app, "llm_with_tools", fake_llm)

    result = graph_app.agent_node(
        {
            "messages": [
                HumanMessage(content="Find medical evidence about anemia.")
            ]
        }
    )

    response = result["messages"][-1]

    assert "language model is not configured" in response.content
    fake_llm.invoke.assert_not_called()
