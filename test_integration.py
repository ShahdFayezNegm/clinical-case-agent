
import json
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from graph_app import graph, llm


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


# --------------------------------------------------
# 1. Full workflow: medication interaction
# --------------------------------------------------

def test_graph_end_to_end_drug_interaction():

    result = graph.invoke({
        "messages": [
            HumanMessage(
                content="Does Zocor interact with Biaxin?"
            )
        ]
    })

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
def test_graph_end_to_end_pubmed_search(mock_get):

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

    result = graph.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Find medical evidence about "
                    "aspirin and bleeding risk."
                )
            )
        ]
    })

    final_message = get_final_message(result)
    output = json.loads(final_message.content)

    assert output["status"] == "SUCCESS"
    assert output["source"] == "PubMed / NCBI"
    assert output["result_count"] == 1

    assert output["results"][0]["pmid"] == "12345"

    assert output["results"][0]["title"] == (
        "Aspirin and bleeding risk study"
    )

    assert mock_get.call_count == 2


# --------------------------------------------------
# 3. Full workflow: general question through Ollama
# --------------------------------------------------

def test_graph_end_to_end_general_question():

    with patch.object(
        type(llm),
        "invoke",
        autospec=True
    ) as mock_invoke:

        mock_invoke.return_value = AIMessage(
            content=(
                "Machine Learning is a field of AI that "
                "learns patterns from data."
            )
        )

        result = graph.invoke({
            "messages": [
                HumanMessage(
                    content="What is Machine Learning?"
                )
            ]
        })

        final_message = get_final_message(result)

        assert isinstance(final_message, AIMessage)

        assert "Machine Learning" in (
            final_message.content
        )

        mock_invoke.assert_called_once()


# --------------------------------------------------
# 4. Full workflow: Ollama failure handling
# --------------------------------------------------

def test_graph_handles_ollama_failure():

    with patch.object(
        type(llm),
        "invoke",
        autospec=True
    ) as mock_invoke:

        mock_invoke.side_effect = RuntimeError(
            "Ollama is unavailable"
        )

        result = graph.invoke({
            "messages": [
                HumanMessage(
                    content="Explain artificial intelligence."
                )
            ]
        })

        final_message = get_final_message(result)

        assert isinstance(final_message, AIMessage)

        assert "could not generate a response" in (
            final_message.content.lower()
        )

        assert "RuntimeError" in final_message.content
