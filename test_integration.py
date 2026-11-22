
import json
from unittest.mock import Mock, patch

import graph_app
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from graph_app import graph


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def make_response(json_data=None, content=b"", status_code=200):
    response = Mock()
    response.status_code = status_code
    response.content = content
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


def get_final_message(result):
    return result["messages"][-1]


def make_tool_calling_llm(tool_name, tool_args):
    """
    Fake LLM that:
    1. Requests execution of a specific tool.
    2. Returns the actual tool result as its final response.

    The real LangGraph workflow and ToolNode remain active.
    """

    class FakeLLMWithTools:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "id": "integration-tool-call",
                            "type": "tool_call",
                        }
                    ],
                )

            tool_messages = [
                message
                for message in messages
                if isinstance(message, ToolMessage)
            ]

            assert tool_messages, (
                "Expected the executed tool result."
            )

            return AIMessage(
                content=tool_messages[-1].content
            )

    return FakeLLMWithTools()


# --------------------------------------------------
# 1. Full workflow: medication interaction
# --------------------------------------------------

def test_graph_end_to_end_drug_interaction(monkeypatch):

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    monkeypatch.setattr(
        graph_app,
        "llm_with_tools",
        make_tool_calling_llm(
            "check_drug_interactions",
            {
                "question": "Does Zocor interact with Biaxin?"
            },
        ),
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Does Zocor interact with Biaxin?"
                )
            ]
        }
    )

    final_message = get_final_message(result)
    output = json.loads(final_message.content)

    assert output["status"] == "POTENTIAL_INTERACTION"

    assert set(output["medications"]) == {
        "simvastatin",
        "clarithromycin",
    }

    assert output["interaction_found"] is True
    assert isinstance(output["evidence"], list)
    assert isinstance(output["limitations"], list)


# --------------------------------------------------
# 2. Full workflow: PubMed evidence search
# --------------------------------------------------

@patch("graph_app.requests.get")
def test_graph_end_to_end_pubmed_search(
    mock_get,
    monkeypatch,
):

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    monkeypatch.setattr(
        graph_app,
        "llm_with_tools",
        make_tool_calling_llm(
            "search_medical_evidence",
            {
                "query": "aspirin bleeding risk"
            },
        ),
    )

    esearch_response = make_response(
        json_data={
            "esearchresult": {
                "idlist": ["12345"]
            }
        }
    )

    xml_content = b"""
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>
                        Aspirin and bleeding risk study
                    </ArticleTitle>
                    <Journal>
                        <Title>Test Medical Journal</Title>
                        <JournalIssue>
                            <PubDate>
                                <Year>2025</Year>
                            </PubDate>
                        </JournalIssue>
                    </Journal>
                    <Abstract>
                        <AbstractText>
                            This is a test abstract.
                        </AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    efetch_response = make_response(
        content=xml_content
    )

    mock_get.side_effect = [
        esearch_response,
        efetch_response,
    ]

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Find medical evidence about "
                        "aspirin and bleeding risk."
                    )
                )
            ]
        }
    )

    final_message = get_final_message(result)
    output = json.loads(final_message.content)

    assert output["status"] == "SUCCESS"

    assert output["source"] == "PubMed / NCBI"

    assert output["result_count"] == 1

    assert output["results"][0]["pmid"] == "12345"

    assert (
        "Aspirin and bleeding risk study"
        in output["results"][0]["title"]
    )

    assert mock_get.call_count == 2


# --------------------------------------------------
# 3. General question through the LLM
# --------------------------------------------------

def test_graph_general_question(monkeypatch):

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    mock_llm = Mock()

    mock_llm.invoke.return_value = AIMessage(
        content=(
            "Machine Learning is a field of AI "
            "that learns patterns from data."
        )
    )

    monkeypatch.setattr(
        graph_app,
        "llm_with_tools",
        mock_llm,
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is Machine Learning?"
                )
            ]
        }
    )

    final_message = get_final_message(result)

    assert isinstance(final_message, AIMessage)

    assert "Machine Learning" in final_message.content

    mock_llm.invoke.assert_called_once()


# --------------------------------------------------
# 4. LLM failure handling
# --------------------------------------------------

def test_graph_handles_llm_failure(monkeypatch):

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    mock_llm = Mock()

    mock_llm.invoke.side_effect = RuntimeError(
        "LLM service is unavailable"
    )

    monkeypatch.setattr(
        graph_app,
        "llm_with_tools",
        mock_llm,
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is Machine Learning?"
                )
            ]
        }
    )

    final_message = get_final_message(result)

    assert isinstance(final_message, AIMessage)

    assert (
        "could not generate a response"
        in final_message.content.lower()
    )

    assert "RuntimeError" in final_message.content

    mock_llm.invoke.assert_called_once()
