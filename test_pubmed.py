
import json
from unittest.mock import Mock, patch
from datetime import date
import requests

from graph_app import (
    extract_pubmed_topic,
    _extract_pubmed_date_range,
    _topic_concepts,
    _relevance_score,
    search_medical_evidence,
)

def test_extract_pubmed_date_range_treats_recent_as_last_five_years():
    result = _extract_pubmed_date_range(
        "Find recent clinical studies about iron deficiency"
    )

    today = date.today()
    expected_start = today.replace(
        year=today.year - 5
    ).strftime("%Y/%m/%d")

    assert result == {
        "mindate": expected_start,
        "maxdate": today.strftime("%Y/%m/%d"),
        "datetype": "pdat",
    }


def test_extract_pubmed_date_range_returns_none_without_recency_request():
    result = _extract_pubmed_date_range(
        "Find clinical studies about iron deficiency"
    )

    assert result is None
# =========================================================
# Helpers
# =========================================================

def make_response(json_data=None, text="", status_code=200):
    """Create a mocked HTTP response compatible with JSON and XML requests."""
    response = Mock()
    response.status_code = status_code
    response.raise_for_status.return_value = None
    response.json.return_value = json_data or {}

    if isinstance(text, str):
        response.text = text
        response.content = text.encode("utf-8")
    else:
        response.text = text.decode("utf-8")
        response.content = text

    return response


def make_esearch_response(ids):
    """Create a PubMed ESearch JSON response."""
    return {
        "esearchresult": {
            "idlist": ids
        }
    }


def make_pubmed_article(pmid, title, abstract):
    """Create one PubMed article XML element."""
    return f"""
    <PubmedArticle>
        <MedlineCitation>
            <PMID>{pmid}</PMID>
            <Article>
                <ArticleTitle>{title}</ArticleTitle>
                <Journal>
                    <Title>Test Journal</Title>
                    <JournalIssue>
                        <PubDate>
                            <Year>2025</Year>
                        </PubDate>
                    </JournalIssue>
                </Journal>
                <Abstract>
                    <AbstractText>{abstract}</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """


def make_pubmed_xml(*articles):
    """Wrap article XML elements in a PubMedArticleSet."""
    return (
        "<PubmedArticleSet>"
        + "".join(articles)
        + "</PubmedArticleSet>"
    )


def invoke_search(query):
    """Invoke the LangChain PubMed tool and decode its JSON response."""
    return json.loads(
        search_medical_evidence.invoke({"query": query})
    )


# =========================================================
# Topic extraction tests
# =========================================================

def test_extract_pubmed_topic_removes_search_prefix():
    result = extract_pubmed_topic(
        "Search PubMed for aspirin bleeding risk"
    )

    assert isinstance(result, str)
    assert result
    assert "aspirin" in result.lower()


def test_extract_pubmed_topic_removes_find_prefix():
    result = extract_pubmed_topic(
        "Find medical evidence about warfarin"
    )

    assert isinstance(result, str)
    assert "warfarin" in result.lower()


def test_extract_pubmed_topic_preserves_main_topic():
    result = extract_pubmed_topic(
        "Search for diabetes treatment"
    )

    assert "diabetes" in result.lower()
    assert "treatment" in result.lower()


def test_extract_pubmed_topic_handles_simple_query():
    result = extract_pubmed_topic("aspirin and bleeding")

    assert result == "aspirin and bleeding"


def test_extract_pubmed_topic_returns_string():
    result = extract_pubmed_topic("Find studies about diabetes")

    assert isinstance(result, str)


# =========================================================
# Topic concepts tests
# =========================================================

def test_topic_concepts_detects_known_concepts():
    concepts = _topic_concepts(
        "aspirin and bleeding risk"
    )

    assert isinstance(concepts, list)
    assert "aspirin" in concepts
    assert "bleeding" in concepts


def test_topic_concepts_handles_unrecognized_terms():
    concepts = _topic_concepts(
        "uncommon biomarker condition"
    )

    assert isinstance(concepts, list)
    assert "biomarker" in concepts
    assert "condition" in concepts


# =========================================================
# Relevance scoring tests
# =========================================================

def test_relevance_score_rewards_title_matches():
    title_score, _ = _relevance_score(
        "aspirin bleeding",
        "Aspirin and bleeding risk",
        "A medical study.",
    )

    abstract_score, _ = _relevance_score(
        "aspirin bleeding",
        "A medical study",
        "A study about aspirin and bleeding.",
    )

    assert title_score > abstract_score


def test_relevance_score_rewards_relevant_articles():
    relevant_score, relevant_terms = _relevance_score(
        "aspirin bleeding risk",
        "Aspirin and bleeding risk",
        "This study investigates aspirin and bleeding.",
    )

    irrelevant_score, irrelevant_terms = _relevance_score(
        "aspirin bleeding risk",
        "Exercise and sleep quality",
        "This study investigates physical activity and sleep.",
    )

    assert relevant_score > irrelevant_score
    assert len(relevant_terms) > len(irrelevant_terms)


def test_relevance_score_returns_score_and_concepts():
    result = _relevance_score(
        "diabetes",
        "Diabetes treatment",
        "A study about diabetes.",
    )

    assert isinstance(result, tuple)
    assert len(result) == 2

    score, matched_concepts = result

    assert isinstance(score, (int, float))
    assert isinstance(matched_concepts, list)
    assert "diabetes" in matched_concepts


# =========================================================
# Input validation tests
# =========================================================

def test_search_medical_evidence_rejects_empty_query():
    result = invoke_search("")

    assert result["status"] == "INVALID_QUERY"
    assert result["results"] == []


def test_search_medical_evidence_rejects_long_query():
    query = "a" * 301

    result = invoke_search(query)

    assert result["status"] == "INVALID_QUERY"
    assert result["results"] == []


# =========================================================
# PubMed error handling tests
# =========================================================

@patch("graph_app.requests.get")
def test_search_medical_evidence_handles_connection_error(mock_get):
    mock_get.side_effect = requests.ConnectionError(
        "Connection failed"
    )

    result = invoke_search("aspirin bleeding")

    assert result["status"] == "SEARCH_ERROR"
    assert result["results"] == []


@patch("graph_app.requests.get")
def test_search_medical_evidence_handles_no_results(mock_get):
    mock_get.return_value = make_response(
        make_esearch_response([])
    )

    result = invoke_search("unknown medical topic")

    assert result["status"] == "NO_RESULTS"
    assert result["results"] == []
    assert result["result_count"] == 0


@patch("graph_app.requests.get")
def test_search_medical_evidence_handles_http_error(mock_get):
    response = make_response()
    response.raise_for_status.side_effect = (
        requests.HTTPError("HTTP error")
    )

    mock_get.return_value = response

    result = invoke_search("aspirin bleeding")

    assert result["status"] == "SEARCH_ERROR"
    assert result["results"] == []


@patch("graph_app.requests.get")
def test_search_medical_evidence_handles_invalid_esearch_json(mock_get):
    response = make_response()
    response.json.side_effect = ValueError("Invalid JSON")

    mock_get.return_value = response

    result = invoke_search("aspirin bleeding")

    assert result["status"] == "SEARCH_ERROR"
    assert result["results"] == []


# =========================================================
# Successful PubMed extraction and ranking
# =========================================================

@patch("graph_app.requests.get")
def test_search_medical_evidence_extracts_and_ranks_articles(
    mock_get,
):
    article_1 = make_pubmed_article(
        "101",
        "Aspirin and bleeding risk",
        (
            "This study investigates the relationship between "
            "aspirin use and bleeding risk."
        ),
    )

    article_2 = make_pubmed_article(
        "102",
        "Exercise and sleep quality",
        (
            "This study investigates physical activity and sleep."
        ),
    )

    esearch_response = make_response(
        make_esearch_response(["101", "102"])
    )

    efetch_response = make_response(
        text=make_pubmed_xml(article_1, article_2)
    )

    mock_get.side_effect = [
        esearch_response,
        efetch_response,
    ]

    result = invoke_search("aspirin bleeding risk")

    assert result["status"] == "SUCCESS"
    assert len(result["results"]) == 2

    first_article = result["results"][0]

    assert first_article["pmid"] == "101"
    assert "Aspirin" in first_article["title"]
    assert "pubmed.ncbi.nlm.nih.gov" in first_article["source_url"]

    assert (
        result["results"][0]["relevance_score"]
        >= result["results"][1]["relevance_score"]
    )
@patch("graph_app.requests.get")
def test_search_medical_evidence_returns_all_expanded_search_queries(
    mock_get,
):
    article = make_pubmed_article(
        "150",
        "Iron deficiency anemia and ferritin",
        (
            "This study investigates iron deficiency anemia "
            "and measures serum ferritin and transferrin saturation."
        ),
    )

    def mock_pubmed_request(url, *args, **kwargs):
        if "esearch.fcgi" in url:
            return make_response(
                make_esearch_response(["150"])
            )

        if "efetch.fcgi" in url:
            return make_response(
                text=make_pubmed_xml(article)
            )

        raise AssertionError(f"Unexpected PubMed URL: {url}")

    mock_get.side_effect = mock_pubmed_request

    result = invoke_search("iron deficiency anemia and ferritin")

    assert result["status"] == "SUCCESS"
    assert "search_queries" in result
    assert isinstance(result["search_queries"], list)
    assert len(result["search_queries"]) > 1
    assert all(
        isinstance(query, str) and query.strip()
        for query in result["search_queries"]
    )

@patch("graph_app.requests.get")
def test_search_medical_evidence_uses_topic_specific_marker_coverage(
    mock_get,
):
    article = make_pubmed_article(
        "201",
        "Iron deficiency anemia biomarkers",
        (
            "Serum ferritin, transferrin saturation, "
            "methylmalonic acid, homocysteine and "
            "mean corpuscular volume were measured."
        ),
    )

    def mock_pubmed_request(url, *args, **kwargs):
        if "esearch.fcgi" in url:
            return make_response(
                make_esearch_response(["201"])
            )

        if "efetch.fcgi" in url:
            return make_response(
                text=make_pubmed_xml(article)
            )

        raise AssertionError(f"Unexpected PubMed URL: {url}")

    mock_get.side_effect = mock_pubmed_request

    result = json.loads(
        search_medical_evidence.invoke({
            "query": "iron deficiency anemia",
        })
    )

    assert result["status"] == "SUCCESS"
    assert len(result["results"]) == 1

    returned_article = result["results"][0]

    assert returned_article["marker_coverage_score"] == 9
    assert set(returned_article["matched_markers"]) == {
        "ferritin",
        "transferrin_saturation",
        "mcv",
    }


@patch("graph_app.requests.get")
def test_search_medical_evidence_preserves_original_question_constraints(
    mock_get,
):
    article = make_pubmed_article(
        "301",
        "Iron deficiency anemia in adolescents",
        (
            "A cohort study of adolescents aged 12 to 17 years "
            "investigated iron deficiency anemia and measured "
            "serum ferritin and transferrin saturation."
        ),
    )

    esearch_params = []

    def mock_pubmed_request(url, *args, **kwargs):
        if "esearch.fcgi" in url:
            esearch_params.append(kwargs["params"])
            return make_response(
                make_esearch_response(["301"])
            )

        if "efetch.fcgi" in url:
            return make_response(
                text=make_pubmed_xml(article)
            )

        raise AssertionError(f"Unexpected PubMed URL: {url}")

    mock_get.side_effect = mock_pubmed_request

    original_question = (
        "Find medical evidence about iron deficiency anemia "
        "in adolescents published in the last 5 years."
    )

    result = json.loads(
        search_medical_evidence.invoke({
            "query": "iron deficiency anemia",
            "original_question": original_question,
        })
    )

    today = date.today()
    expected_start = today.replace(
        year=today.year - 5
    ).strftime("%Y/%m/%d")

    expected_end = today.strftime("%Y/%m/%d")

    assert result["status"] == "SUCCESS"
    assert esearch_params

    for params in esearch_params:
        assert params["mindate"] == expected_start
        assert params["maxdate"] == expected_end
        assert params["datetype"] == "pdat"

    assert len(result["results"]) == 1

    returned_article = result["results"][0]

    assert (
        returned_article["study_population_relevance"]
        == "Adolescent"
    )
    assert returned_article["age_relevance_score"] == 3

# =========================================================
# Invalid XML handling
# =========================================================

@patch("graph_app.requests.get")
def test_search_medical_evidence_handles_invalid_xml(mock_get):
    esearch_response = make_response(
        make_esearch_response(["101"])
    )

    efetch_response = make_response(
        text="<PubmedArticleSet><Invalid XML"
    )

    def mock_pubmed_request(url, *args, **kwargs):
        if "esearch.fcgi" in url:
            return make_response(
                make_esearch_response(["201"])
            )

        if "efetch.fcgi" in url:
            return make_response(
                text=make_pubmed_xml(article)
            )

        raise AssertionError(f"Unexpected PubMed URL: {url}")

    mock_get.side_effect = mock_pubmed_request
# =========================================================
# Maximum five results
# =========================================================

@patch("graph_app.requests.get")
def test_search_medical_evidence_returns_maximum_five_results(
    mock_get,
):
    ids = [str(number) for number in range(101, 111)]

    articles = [
        make_pubmed_article(
            str(number),
            f"Aspirin research study {number}",
            "This study investigates aspirin and its effects.",
        )
        for number in range(101, 111)
    ]

    esearch_response = make_response(
        make_esearch_response(ids)
    )

    efetch_response = make_response(
        text=make_pubmed_xml(*articles)
    )

    mock_get.side_effect = [
        esearch_response,
        efetch_response,
    ]

    result = invoke_search("aspirin research")

    assert result["status"] == "SUCCESS"
    assert len(result["results"]) <= 5


# =========================================================
# Missing article fields
# =========================================================

@patch("graph_app.requests.get")
def test_pubmed_handles_articles_with_missing_fields(mock_get):
    article_with_title = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>201</PMID>
            <Article>
                <ArticleTitle>Aspirin research study</ArticleTitle>
                <Journal>
                    <Title>Test Journal</Title>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """

    article_without_title = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>202</PMID>
            <Article>
                <Journal>
                    <Title>Test Journal</Title>
                </Journal>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """

    esearch_response = make_response(
        make_esearch_response(["201", "202"])
    )

    efetch_response = make_response(
        text=make_pubmed_xml(
            article_with_title,
            article_without_title,
        )
    )

    mock_get.side_effect = [
        esearch_response,
        efetch_response,
    ]

    result = invoke_search("aspirin research")

    assert result["status"] == "SUCCESS"
    assert len(result["results"]) == 2

    articles_by_id = {
        article["pmid"]: article
        for article in result["results"]
    }

    assert (
        articles_by_id["201"]["title"]
        == "Aspirin research study"
    )

    assert (
        articles_by_id["202"]["title"]
        == "Title unavailable"
    )

    assert (
        articles_by_id["202"]["abstract"]
        == "Abstract unavailable"
    )
