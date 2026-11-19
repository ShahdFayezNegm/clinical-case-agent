from datetime import date, timedelta
import json
import xml.etree.ElementTree as ET
from graph_app import _build_pubmed_queries
from graph_app import validate_pubmed_result, _select_diverse_articles
from graph_app import (
    _marker_coverage_score,
    _age_relevance_score,
    _study_population_relevance,
    _extract_target_population,
    _extract_age_ranges,
    _extract_patient_age,
    _extract_age_groups,
)
from graph_app import _extract_patient_age
from graph_app import _is_pregnancy_related
from graph_app import (
    extract_medications,
    extract_unrecognized_medications,
    is_medication_question,
    check_drug_interactions,
    validate_result,
    _extract_article_types,
)


def run_interaction_tool(question: str) -> dict:
    """Run the LangChain tool directly and parse its JSON output."""
    raw_result = check_drug_interactions.invoke(
        {"question": question}
    )
    return json.loads(raw_result)

# ============================================================
# Target population extraction tests
# ============================================================

def test_extract_target_population_adults():
    assert _extract_target_population(
        "Find studies about vitamin D in adults"
    ) == "adult"


def test_extract_target_population_adolescents():
    assert _extract_target_population(
        "Find studies about vitamin D in adolescents"
    ) == "adolescent"


def test_extract_target_population_children():
    assert _extract_target_population(
        "Find studies about vitamin D in children"
    ) == "child"


def test_extract_target_population_age_range():
    assert _extract_target_population(
        "Find studies about vitamin D in adolescents aged 12 to 18 years"
    ) == "adolescent"


def test_extract_target_population_without_age_group():
    assert _extract_target_population(
        "Find studies about vitamin D deficiency"
    ) is None
# ============================================================
# Age group extraction tests
# ============================================================

def test_extract_age_groups_returns_meSH_age_descriptors():
    article = ET.fromstring("""
    <PubmedArticle>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Adult</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Middle Aged</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Adolescent</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </PubmedArticle>
    """)

    assert _extract_age_groups(article) == [
        "Adult",
        "Middle Aged",
        "Adolescent",
    ]


def test_extract_age_groups_ignores_non_age_descriptors():
    article = ET.fromstring("""
    <PubmedArticle>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Adult</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Iron Deficiency</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Female</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </PubmedArticle>
    """)

    assert _extract_age_groups(article) == ["Adult"]


def test_extract_age_groups_removes_duplicates():
    article = ET.fromstring("""
    <PubmedArticle>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Adult</DescriptorName>
            </MeshHeading>
            <MeshHeading>
                <DescriptorName>Adult</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </PubmedArticle>
    """)

    assert _extract_age_groups(article) == ["Adult"]


def test_extract_age_groups_returns_empty_list_without_mesh_headings():
    article = ET.fromstring("<PubmedArticle />")

    assert _extract_age_groups(article) == []
# ============================================================
# Patient age extraction tests
# ============================================================
def test_extract_patient_age_prefers_explicit_patient_age():
    assert _extract_patient_age(
        "A 35-year-old patient needs evidence from studies "
        "in adolescents aged 12 to 18 years"
    ) == 35
def test_extract_patient_age_year_old():
    assert _extract_patient_age(
        "A 25-year-old patient presents with fatigue"
    ) == 25


def test_extract_patient_age_aged():
    assert _extract_patient_age(
        "The patient is aged 40"
    ) == 40


def test_extract_patient_age_age_label():
    assert _extract_patient_age(
        "Patient age: 65"
    ) == 65


def test_extract_patient_age_years_old_abbreviation():
    assert _extract_patient_age(
        "A 32 y/o patient presents with anemia"
    ) == 32


def test_extract_patient_age_adolescent_study_range():
    assert _extract_patient_age(
        "Find studies in adolescents aged 12 to 18 years"
    ) is None


def test_extract_patient_age_not_specified():
    assert _extract_patient_age(
        "Find medical evidence about iron deficiency anemia"
    ) is None


def test_extract_patient_age_out_of_range():
    assert _extract_patient_age(
        "A 150-year-old patient"
    ) is None
# ============================================================
# Medication alias tests
# ============================================================

def test_brand_names_are_normalized():
    medications = extract_medications(
        "Does Zocor interact with Biaxin?"
    )

    assert set(medications) == {
        "simvastatin",
        "clarithromycin",
    }


def test_paracetamol_is_normalized():
    medications = extract_medications(
        "Could paracetamol and loratadine interact?"
    )

    assert "acetaminophen" in medications
    assert "loratadine" in medications


def test_unknown_medication_candidate_is_detected():
    unknown = extract_unrecognized_medications(
        "Does warfarin interact with novamed?"
    )

    assert "novamed" in unknown


# ============================================================
# Routing tests
# ============================================================

def test_interaction_question_uses_tool():
    assert is_medication_question(
        "Could warfarin and aspirin cause problems together?"
    )


def test_two_known_medications_use_tool():
    assert is_medication_question(
        "Can I take warfarin with amoxicillin?"
    )


def test_general_aspirin_question_uses_llm():
    assert not is_medication_question(
        "What is aspirin used for?"
    )


def test_daily_aspirin_question_uses_llm():
    assert not is_medication_question(
        "Should I take aspirin every day?"
    )


def test_machine_learning_question_uses_llm():
    assert not is_medication_question(
        "What is Machine Learning?"
    )


# ============================================================
# Interaction result tests
# ============================================================

def test_warfarin_and_aspirin_interaction():
    result = run_interaction_tool(
        "Could warfarin and aspirin cause problems together?"
    )

    assert result["status"] == "POTENTIAL_INTERACTION"
    assert result["interaction_found"] is True
    assert set(result["medications"]) == {
        "warfarin",
        "aspirin",
    }


def test_zocor_and_biaxin_interaction():
    result = run_interaction_tool(
        "Does Zocor interact with Biaxin?"
    )

    assert result["status"] == "POTENTIAL_INTERACTION"
    assert result["interaction_found"] is True
    assert set(result["medications"]) == {
        "simvastatin",
        "clarithromycin",
    }


def test_one_medication_returns_insufficient_information():
    result = run_interaction_tool(
        "Can you check whether warfarin interacts with other medications?"
    )

    assert result["status"] == "INSUFFICIENT_INFORMATION"
    assert result["interaction_found"] is None


def test_unlisted_pair_does_not_claim_safety():
    result = run_interaction_tool(
        "Can you check whether warfarin interacts with amoxicillin?"
    )

    assert result["status"] == "NO_MATCH_FOUND"
    assert result["interaction_found"] is None

    assert any(
        "does not mean the combination is safe"
        in limitation.lower()
        for limitation in result["limitations"]
    )


def test_unknown_medication_returns_unknown_status():
    result = run_interaction_tool(
        "Does warfarin interact with novamed?"
    )

    assert result["status"] == "UNKNOWN_MEDICATION"


# ============================================================
# JSON schema tests
# ============================================================

def test_interaction_result_matches_schema():
    result = run_interaction_tool(
        "Does warfarin interact with aspirin?"
    )

    assert validate_result(result)


def test_result_is_valid_json():
    raw_result = check_drug_interactions.invoke(
        {
            "question": (
                "Does warfarin interact with aspirin?"
            )
        }
    )

    parsed = json.loads(raw_result)

    assert isinstance(parsed, dict)
    assert validate_result(parsed)
from graph_app import (
    validate_pubmed_result,
    is_valid_pubmed_topic,
    _relevance_score,
    _marker_coverage_score,
)

def test_validate_result_rejects_non_string_status():
    result = {
        "status": None,
        "medications": [],
        "interaction_found": None,
        "evidence": [],
        "limitations": [],
        "next_step": "Review the available information.",
    }

    assert not validate_result(result)


def test_validate_result_rejects_blank_medication():
    result = {
        "status": "NO_MATCH_FOUND",
        "medications": ["   "],
        "interaction_found": False,
        "evidence": ["No interaction found."],
        "limitations": ["Limited evidence."],
        "next_step": "Review the available information.",
    }

    assert not validate_result(result)


def test_validate_result_rejects_blank_next_step():
    result = {
        "status": "NO_MATCH_FOUND",
        "medications": [],
        "interaction_found": False,
        "evidence": ["No interaction found."],
        "limitations": ["Limited evidence."],
        "next_step": "   ",
    }

    assert not validate_result(result)
# ============================================================
# PubMed validation tests
# ============================================================


def test_valid_pubmed_result_matches_schema():
    result = {
        "status": "SUCCESS",
        "query": "aspirin and bleeding risk",
        "search_query": "aspirin AND bleeding risk",
        "source": "PubMed / NCBI",
        "result_count": 1,
        "candidate_count": 1,
        "results": [
            {
                "pmid": "12345678",
                "title": "Aspirin and bleeding risk",
                "journal": "Example Medical Journal",
                "publication_date": "2024",
                "abstract": "A study of aspirin and bleeding risk.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 5.0,
                                "marker_coverage_score": 3,
                "age_relevance_score": 2,
                "study_population_relevance": "relevant",
                "article_types": ["Clinical Trial"],
                "study_design": "Randomized Controlled Trial",
                "pregnancy_related": False,
                "matched_markers": ["aspirin", "bleeding"],
                "matched_terms": ["aspirin", "bleeding"],
                "matched_queries": ["aspirin AND bleeding risk"],
            }
        ],
        "limitations": [],
    }

    assert validate_pubmed_result(result)

def test_invalid_pubmed_status_is_rejected():
    result = {
        "status": "INVALID_STATUS",
        "query": "aspirin",
        "search_query": "aspirin",
        "source": "PubMed / NCBI",
        "result_count": 0,
        "candidate_count": 0,
        "results": [],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)


def test_pubmed_result_rejects_success_without_articles():
    result = {
        "status": "SUCCESS",
        "query": "aspirin",
        "results": [],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)


def test_pubmed_result_rejects_article_missing_required_field():
    result = {
        "status": "SUCCESS",
        "query": "aspirin",
        "results": [
            {
                "pmid": "12345678",
                "title": "Aspirin study",
                "journal": "Example Journal",
                "publication_date": "2024",
                "abstract": "Example abstract",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 5.0,
                "matched_terms": ["aspirin"],
                # matched_queries is missing
            }
        ],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)


def test_pubmed_result_rejects_boolean_relevance_score():
    result = {
        "status": "SUCCESS",
        "query": "aspirin",
        "results": [
            {
                "pmid": "12345678",
                "title": "Aspirin study",
                "journal": "Example Journal",
                "publication_date": "2024",
                "abstract": "Example abstract",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": True,
                "matched_terms": ["aspirin"],
                "matched_queries": ["aspirin"],
            }
        ],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)

def test_pubmed_result_rejects_boolean_marker_coverage_score():
    result = {
        "status": "SUCCESS",
        "query": "aspirin",
        "results": [
            {
                "pmid": "12345678",
                "title": "Aspirin study",
                "journal": "Example Journal",
                "publication_date": "2024",
                "abstract": "Example abstract",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 5.0,
                "matched_terms": ["aspirin"],
                "matched_queries": ["aspirin"],
                "marker_coverage_score": True,
            }
        ],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)


def test_pubmed_result_rejects_invalid_matched_markers():
    result = {
        "status": "SUCCESS",
        "query": "aspirin",
        "results": [
            {
                "pmid": "12345678",
                "title": "Aspirin study",
                "journal": "Example Journal",
                "publication_date": "2024",
                "abstract": "Example abstract",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 5.0,
                "matched_terms": ["aspirin"],
                "matched_queries": ["aspirin"],
                "marker_coverage_score": 3,
                "matched_markers": "aspirin",
            }
        ],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)


def test_pubmed_result_rejects_non_boolean_pregnancy_related():
    result = {
        "status": "SUCCESS",
        "query": "aspirin",
        "results": [
            {
                "pmid": "12345678",
                "title": "Aspirin study",
                "journal": "Example Journal",
                "publication_date": "2024",
                "abstract": "Example abstract",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "relevance_score": 5.0,
                "matched_terms": ["aspirin"],
                "matched_queries": ["aspirin"],
                "marker_coverage_score": 3,
                "pregnancy_related": "False",
            }
        ],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)
def test_pubmed_result_rejects_inconsistent_result_count():
    result = {
        "status": "NO_RESULTS",
        "query": "aspirin",
        "result_count": 1,
        "candidate_count": 1,
        "results": [],
        "limitations": [],
    }

    assert not validate_pubmed_result(result)

# ============================================================
# PubMed query tests
# ============================================================

def test_invalid_pubmed_query_is_rejected():
    assert not is_valid_pubmed_topic("!!!")


def test_valid_pubmed_query_is_accepted():
    assert is_valid_pubmed_topic(
        "aspirin and bleeding risk"
    )


# ============================================================
# PubMed relevance tests
# ============================================================

def test_relevant_title_scores_higher_than_unrelated_title():
    topic = "wearable devices arrhythmia detection"

    relevant_score = _relevance_score(
        topic,
        "Wearable devices for arrhythmia detection",
        "",
    )

    unrelated_score = _relevance_score(
        topic,
        "Agricultural soil management",
        "",
    )

    assert relevant_score > unrelated_score


def test_topic_concepts_preserves_combined_medical_conditions():
    from graph_app import _topic_concepts

    concepts = _topic_concepts(
        "iron deficiency anemia and vitamin B12 deficiency"
    )

    assert concepts == [
        "iron deficiency anemia",
        "vitamin b12 deficiency",
    ]
def test_age_relevance_rewards_matching_age_group():
    score = _age_relevance_score(
        45,
        ["Middle Aged"],
    )

    assert score == 3


def test_age_relevance_penalizes_mismatched_age_group():
    score = _age_relevance_score(
        45,
        ["Child", "Adolescent"],
    )

    assert score == -3
def test_age_relevance_rewards_matching_target_adolescent():
    score = _age_relevance_score(
        None,
        ["Adolescent"],
        "adolescent",
    )

    assert score == 1
def test_age_relevance_does_not_assume_young_people_are_adolescents():
    score = _age_relevance_score(
        None,
        [],
        "adolescent",
        "Vitamin D deficiency in young people",
        "The study included young people with first-episode psychosis.",
    )

    assert score == 0

def test_age_relevance_scores_mixed_group_with_adolescents():
    score = _age_relevance_score(
        None,
        ["Child", "Adolescent", "Adult"],
        "adolescent",
    )

    assert score == 1
def test_age_relevance_scores_mixed_age_range_for_adolescent_target():
    score = _age_relevance_score(
        None,
        ["Adult", "Young Adult"],
        "adolescent",
        "Vitamin D supplementation",
        "Participants were aged 17 to 74 years.",
    )

    assert score == 1
def test_age_relevance_rewards_adolescent_specific_age_range():
    score = _age_relevance_score(
        None,
        ["Adolescent"],
        "adolescent",
        "Vitamin D deficiency in adolescents",
        "Participants were aged 12 to 18 years.",
    )

    assert score == 3

def test_age_relevance_penalizes_adult_only_for_adolescent_target():
    score = _age_relevance_score(
        None,
        ["Middle Aged", "Aged"],
        "adolescent",
    )

    assert score == -3
def test_age_relevance_penalizes_adult_age_range_for_adolescent_target():
    score = _age_relevance_score(
        None,
        ["Adult", "Young Adult"],
        "adolescent",
        "Vitamin D deficiency among adults",
        "Participants were aged 18 to 44 years.",
    )
    assert score == -3

def test_age_relevance_does_not_score_missing_age_groups():
    score = _age_relevance_score(
        None,
        [],
        "adolescent",
    )

    assert score == 0

def test_age_relevance_does_not_penalize_missing_age_group():
    score = _age_relevance_score(
        45,
        [],
    )

    assert score == 0

def test_study_population_relevance_identifies_adults():
    result = _study_population_relevance(
        ["Adult", "Middle Aged"],
        "adult",
    )

    assert result == "Adult"
def test_study_population_relevance_without_target_population():
    result = _study_population_relevance(
        [],
        "",
        article_title="Vitamin D deficiency in adolescents",
        abstract=(
            "We evaluated vitamin D deficiency "
            "in children and adolescents."
        ),
    )

    assert result == "Child"
def test_study_population_relevance_identifies_adolescents():
    result = _study_population_relevance(
        [],
        "",
        article_title="Vitamin D deficiency in adolescents",
        abstract=(
            "This study evaluates vitamin D deficiency "
            "among adolescents aged 12 to 18 years."
        ),
    )

    assert result == "Adolescent"

def test_study_population_range_identifies_children():
    result = _study_population_relevance(
        [],
        "",
        article_title="Vitamin D status in school children",
        abstract="Participants were aged 6 to 8 years.",
    )

    assert result == "Child"


def test_study_population_range_identifies_adults():
    result = _study_population_relevance(
        [],
        "",
        article_title="Vitamin D status in adults",
        abstract="Participants were aged 18 to 65 years.",
    )

    assert result == "Adult"


def test_study_population_range_identifies_mixed_population():
    result = _study_population_relevance(
        [],
        "",
        article_title="Vitamin D status across age groups",
        abstract="Participants were aged 12 to 65 years.",
    )

    assert result == "Mixed"


def test_study_population_range_spanning_children_and_adolescents():
    result = _study_population_relevance(
        [],
        "",
        article_title="Vitamin D status in young people",
        abstract="Participants were aged 6 to 15 years.",
    )

    assert result == "Child"

def test_general_review_with_incidental_pregnancy_mention_is_not_pregnancy_related():
    result = _study_population_relevance(
        [],
        "",
        article_title=(
            "Vitamin D deficiency in Bangladesh: "
            "A review of prevalence, causes and recommendations"
        ),
        abstract=(
            "Vitamin D deficiency affects infants, children, "
            "adolescents, adult men and women. "
            "Pregnant women are also among the affected groups."
        ),
        article_types=["Review"],
    )

    assert result == "Mixed"
def test_marker_coverage_detects_all_five_markers():
    title = "Iron deficiency and vitamin B12 deficiency"

    abstract = (
        "Serum ferritin, transferrin saturation, "
        "methylmalonic acid, homocysteine and "
        "mean corpuscular volume were measured."
    )

    score, matched_markers = _marker_coverage_score(
        title,
        abstract,
    )

    assert score == 15
    assert len(matched_markers) == 5


def test_more_marker_coverage_produces_higher_score():
    limited_abstract = "Ferritin and MCV were measured."

    comprehensive_abstract = (
        "Ferritin, transferrin saturation, "
        "methylmalonic acid, homocysteine and MCV "
        "were measured."
    )

    limited_score, _ = _marker_coverage_score(
        "",
        limited_abstract,
    )

    comprehensive_score, _ = _marker_coverage_score(
        "",
        comprehensive_abstract,
    )

    assert comprehensive_score > limited_score
def test_marker_coverage_does_not_match_mma_inside_other_words():
    title = "Anemia study"
    abstract = "Patients were assessed using standard laboratory methods."

    score, matched = _marker_coverage_score(title, abstract)

    assert "methylmalonic_acid" not in matched
    assert score == 0
def test_iron_topic_only_scores_iron_related_markers():
    abstract = (
        "Ferritin, transferrin saturation, methylmalonic acid, "
        "homocysteine and MCV were measured."
    )

    score, matched = _marker_coverage_score(
        "",
        abstract,
        "iron deficiency anemia",
    )

    assert score == 9
    assert set(matched) == {
        "ferritin",
        "transferrin_saturation",
        "mcv",
    }


def test_b12_topic_only_scores_b12_related_markers():
    abstract = (
        "Ferritin, transferrin saturation, methylmalonic acid, "
        "homocysteine and MCV were measured."
    )

    score, matched = _marker_coverage_score(
        "",
        abstract,
        "vitamin B12 deficiency",
    )

    assert score == 9
    assert set(matched) == {
        "methylmalonic_acid",
        "homocysteine",
        "mcv",
    }


def test_unrelated_topic_does_not_score_anemia_markers():
    abstract = (
        "Ferritin, transferrin saturation, methylmalonic acid, "
        "homocysteine and MCV were measured."
    )

    score, matched = _marker_coverage_score(
        "",
        abstract,
        "asthma",
    )

    assert score == 0
    assert matched == []


def test_combined_iron_and_b12_topic_scores_all_markers():
    abstract = (
        "Ferritin, transferrin saturation, methylmalonic acid, "
        "homocysteine and MCV were measured."
    )

    score, matched = _marker_coverage_score(
        "",
        abstract,
        "iron deficiency and vitamin B12 deficiency",
    )

    assert score == 15
    assert len(matched) == 5
def test_pregnancy_mesh_descriptor_alone_is_not_enough():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <MeshHeadingList>
            <MeshHeading>
                <DescriptorName>Pregnancy</DescriptorName>
            </MeshHeading>
        </MeshHeadingList>
    </PubmedArticle>
    """)

    assert _is_pregnancy_related(article) is False

def test_pregnancy_detection_from_title():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <ArticleTitle>
            Iron deficiency during pregnancy
        </ArticleTitle>
    </PubmedArticle>
    """)

    assert _is_pregnancy_related(article) is True


def test_pregnancy_detection_returns_false_for_general_study():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <ArticleTitle>
            Iron deficiency anemia in adult patients
        </ArticleTitle>
        <Abstract>
            <AbstractText>
                This study evaluated iron deficiency in adults.
            </AbstractText>
        </Abstract>
    </PubmedArticle>
    """)

    assert _is_pregnancy_related(article) is False
def test_pregnancy_detection_returns_false_for_incidental_mention():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <ArticleTitle>
            Iron deficiency anemia in women
        </ArticleTitle>
        <Abstract>
            <AbstractText>
                More than 42% of pregnant women have anemia.
                This review discusses iron deficiency in women
                across different life stages.
            </AbstractText>
        </Abstract>
    </PubmedArticle>
    """)

    assert _is_pregnancy_related(article) is False
def test_pregnancy_detection_returns_false_for_general_iron_review():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <ArticleTitle>
            Revisiting Iron Metabolism and Iron Deficiency Anemia
        </ArticleTitle>
        <Abstract>
            <AbstractText>
                Iron requirements increase in pregnancy and lactating
                mothers. This review discusses iron homeostasis,
                absorption, and laboratory diagnosis of iron deficiency.
            </AbstractText>
        </Abstract>
    </PubmedArticle>
    """)

    assert _is_pregnancy_related(article) is False
def test_select_diverse_articles_falls_back_when_no_queries_match():
    results = [
        {
            "title": "No query match",
            "relevance_score": 40,
            "marker_coverage_score": 5,
            "matched_queries": [],
        },
        {
            "title": "Query one match",
            "relevance_score": 30,
            "marker_coverage_score": 5,
            "matched_queries": ["query_1"],
        },
        {
            "title": "Query two match",
            "relevance_score": 25,
            "marker_coverage_score": 5,
            "matched_queries": ["query_2"],
        },
    ]

    selected = _select_diverse_articles(results, limit=3)

    assert len(selected) == 3
    assert {
        query
        for article in selected
        for query in article["matched_queries"]
    } == {"query_1", "query_2"}
def test_select_diverse_articles_handles_all_empty_matched_queries():
    results = [
        {
            "title": "Lower relevance",
            "relevance_score": 20,
            "marker_coverage_score": 3,
            "matched_queries": [],
        },
        {
            "title": "Higher relevance",
            "relevance_score": 40,
            "marker_coverage_score": 5,
            "matched_queries": [],
        },
        {
            "title": "Medium relevance",
            "relevance_score": 30,
            "marker_coverage_score": 4,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(results, limit=3)

    assert len(selected) == 3
    assert [item["title"] for item in selected] == [
        "Higher relevance",
        "Medium relevance",
        "Lower relevance",
    ]
def test_select_diverse_articles_covers_multiple_queries():
    results = [
        {
            "title": "Article A",
            "relevance_score": 30,
            "marker_coverage_score": 3,
            "matched_queries": ["query_1"],
        },
        {
            "title": "Article B",
            "relevance_score": 28,
            "marker_coverage_score": 3,
            "matched_queries": ["query_1"],
        },
        {
            "title": "Article C",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "matched_queries": ["query_2"],
        },
        {
            "title": "Article D",
            "relevance_score": 22,
            "marker_coverage_score": 3,
            "matched_queries": ["query_3"],
        },
    ]

    selected = _select_diverse_articles(results, limit=3)

    covered_queries = {
        query
        for article in selected
        for query in article["matched_queries"]
    }

    assert len(selected) == 3
    assert covered_queries == {"query_1", "query_2", "query_3"}
def test_select_diverse_articles_does_not_overprioritize_query_diversity():
    results = [
        {
            "title": "Highly relevant article",
            "relevance_score": 50,
            "marker_coverage_score": 10,
            "matched_queries": ["query_1"],
        },
        {
            "title": "Relevant article",
            "relevance_score": 45,
            "marker_coverage_score": 8,
            "matched_queries": ["query_1"],
        },
        {
            "title": "Weakly relevant article",
            "relevance_score": 5,
            "marker_coverage_score": 0,
            "matched_queries": ["query_2"],
        },
    ]

    selected = _select_diverse_articles(results, limit=2)

    assert selected[0]["title"] == "Highly relevant article"
    assert selected[1]["title"] == "Relevant article"

def test_select_diverse_articles_prefers_relevance_and_marker_coverage():
    results = [
        {
            "title": "High relevance",
            "relevance_score": 30,
            "marker_coverage_score": 0,
            "matched_queries": ["query_1"],
        },
        {
            "title": "High marker coverage",
            "relevance_score": 25,
            "marker_coverage_score": 9,
            "matched_queries": ["query_2"],
        },
        {
            "title": "Lower combined score",
            "relevance_score": 20,
            "marker_coverage_score": 3,
            "matched_queries": ["query_3"],
        },
    ]

    selected = _select_diverse_articles(results, limit=3)

    assert selected[0]["title"] == "High marker coverage"
    assert selected[1]["title"] == "High relevance"
def test_select_diverse_articles_uses_age_relevance_score():
    results = [
        {
            "title": "Child study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": -3,
            "matched_queries": [],
        },
        {
            "title": "Middle-aged study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 3,
            "matched_queries": [],
        },
        {
            "title": "Unknown age study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(results, limit=3)

    assert selected[0]["title"] == "Middle-aged study"
    assert selected[1]["title"] == "Unknown age study"
    assert selected[2]["title"] == "Child study"

def test_select_diverse_articles_prioritizes_higher_age_relevance_score():
    results = [
        {
            "title": "Unknown population study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "matched_queries": [],
        },
        {
            "title": "Adult population study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 3,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(
        results,
        limit=2,
    )

    assert selected[0]["title"] == "Adult population study"
def test_select_diverse_articles_uses_adult_target_population():
    results = [
        {
            "title": "Child study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "study_population_relevance": "Child",
            "matched_queries": [],
        },
        {
            "title": "Adult study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "study_population_relevance": "Adult",
            "matched_queries": [],
        },
        {
            "title": "Mixed population study",
            "relevance_score": 25,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "study_population_relevance": "Mixed",
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(
        results,
        limit=3,
        target_population="adult",
    )

    assert selected[0]["title"] == "Adult study"
    assert selected[1]["title"] == "Mixed population study"
    assert selected[2]["title"] == "Child study"
    assert selected[0]["title"] == "Adult study"
    assert selected[1]["title"] == "Mixed population study"
    assert selected[2]["title"] == "Child study"


def test_target_population_does_not_override_large_relevance_gap():
    results = [
        {
            "title": "Highly relevant child study",
            "relevance_score": 60,
            "marker_coverage_score": 9,
            "age_relevance_score": -3,
            "matched_queries": [],
        },
        {
            "title": "Weakly relevant adult study",
            "relevance_score": 10,
            "marker_coverage_score": 0,
            "age_relevance_score": 3,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(
        results,
        limit=2,
        target_population="adult",
    )

    assert selected[0]["title"] == "Highly relevant child study"
def test_select_diverse_articles_prioritizes_age_specific_studies():
    results = [
        {
            "title": "General population study",
            "relevance_score": 25,
            "marker_coverage_score": 0,
            "age_relevance_score": 1,
            "matched_queries": [],
        },
        {
            "title": "Adolescent-specific study",
            "relevance_score": 15,
            "marker_coverage_score": 0,
            "age_relevance_score": 3,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(results, limit=2)

    assert selected[0]["title"] == "Adolescent-specific study"
def test_age_relevance_does_not_override_large_relevance_gap():
    results = [
        {
            "title": "Highly relevant but age mismatched",
            "relevance_score": 60,
            "marker_coverage_score": 9,
            "age_relevance_score": -3,
            "matched_queries": [],
        },
        {
            "title": "Age matched but weakly relevant",
            "relevance_score": 10,
            "marker_coverage_score": 0,
            "age_relevance_score": 3,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(results, limit=2)

    assert selected[0]["title"] == (
        "Highly relevant but age mismatched"
    )
def test_age_relevance_rewards_matching_aged_group():
    score = _age_relevance_score(
        75,
        ["Aged"],
    )
    assert score == 3

def test_adult_target_rewards_adult_only_age_range():
    score = _age_relevance_score(
        None,
        ["Adult", "Middle Aged"],
        "adult",
        "Vitamin D deficiency in adults",
        "Participants were aged 18 to 65 years.",
    )

    assert score == 3


def test_adult_target_gives_partial_score_to_mixed_age_groups():
    score = _age_relevance_score(
        None,
        ["Child", "Adult"],
        "adult",
        "Vitamin D status across age groups",
        "Participants included children and adults aged 6 to 65 years.",
    )

    assert score == 1


def test_adult_target_penalizes_child_only_studies():
    score = _age_relevance_score(
        None,
        ["Child", "Adolescent"],
        "adult",
        "Vitamin D deficiency in children",
        "Participants were aged 6 to 17 years.",
    )

    assert score == -3

def test_general_population_article_wins_when_scores_are_equal():
    results = [
        {
            "title": "Pregnancy study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": True,
            "matched_queries": [],
        },
        {
            "title": "General population study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": False,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(results, limit=2)

    assert selected[0]["title"] == "General population study"


def test_higher_scoring_pregnancy_article_is_not_demoted():
    results = [
        {
            "title": "Pregnancy study",
            "relevance_score": 30,
            "marker_coverage_score": 9,
            "age_relevance_score": 0,
            "pregnancy_related": True,
            "matched_queries": [],
        },
        {
            "title": "General population study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": False,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(results, limit=2)

    assert selected[0]["title"] == "Pregnancy study"
def test_pregnant_patient_prefers_pregnancy_studies():
    results = [
        {
            "title": "General study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": False,
            "matched_queries": [],
        },
        {
            "title": "Pregnancy study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": True,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(
        results,
        limit=2,
        patient_pregnancy_status="pregnant",
    )

    assert selected[0]["title"] == "Pregnancy study"


def test_not_pregnant_patient_prefers_general_studies():
    results = [
        {
            "title": "Pregnancy study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": True,
            "matched_queries": [],
        },
        {
            "title": "General study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": False,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(
        results,
        limit=2,
        patient_pregnancy_status="not_pregnant",
    )

    assert selected[0]["title"] == "General study"


def test_unknown_pregnancy_status_prefers_general_studies():
    results = [
        {
            "title": "Pregnancy study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": True,
            "matched_queries": [],
        },
        {
            "title": "General study",
            "relevance_score": 26,
            "marker_coverage_score": 6,
            "age_relevance_score": 0,
            "pregnancy_related": False,
            "matched_queries": [],
        },
    ]

    selected = _select_diverse_articles(
        results,
        limit=2,
        patient_pregnancy_status="unknown",
    )

    assert selected[0]["title"] == "General study"
def test_extract_patient_pregnancy_status():
    from graph_app import _extract_patient_pregnancy_status

    assert _extract_patient_pregnancy_status(
        "Patient is pregnant"
    ) == "pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient is not pregnant"
    ) == "not_pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient pregnancy status is unknown"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "45-year-old patient with anemia"
    ) == "unknown"
    assert _extract_patient_pregnancy_status(
        "No evidence of pregnancy"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Not sure if pregnant"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Unsure if she is pregnant"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Pregnancy test was not performed"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Pregnancy has not been confirmed"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Possible pregnancy"
    ) == "unknown"
    assert _extract_patient_pregnancy_status(
        "Patient was pregnant last year"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "History of pregnancy complications"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Patient had a previous pregnancy"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Patient is currently pregnant"
    ) == "pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient is currently not pregnant"
    ) == "not_pregnant"
    assert _extract_patient_pregnancy_status(
        "Patient was not pregnant last year"
    ) == "unknown"

    assert _extract_patient_pregnancy_status(
        "Patient was pregnant last year but is currently pregnant"
    ) == "pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient was pregnant last year and is currently not pregnant"
    ) == "not_pregnant"
    assert _extract_patient_pregnancy_status(
        "Patient is non-pregnant"
    ) == "not_pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient denies pregnancy"
    ) == "not_pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient has a negative pregnancy test"
    ) == "not_pregnant"

    assert _extract_patient_pregnancy_status(
        "Patient is not currently pregnant"
    ) == "not_pregnant"
def test_agent_node_passes_patient_pregnancy_status():
    from graph_app import agent_node
    from langchain_core.messages import HumanMessage

    cases = [
        ("Patient is pregnant. Research clinical studies.", "pregnant"),
        ("Patient is not pregnant. Research clinical studies.", "not_pregnant"),
        ("Patient pregnancy status is unknown. Research clinical studies.", "unknown"),
    ]

    for question, expected_status in cases:
        result = agent_node({
            "messages": [HumanMessage(content=question)]
        })

        tool_call = result["messages"][0].tool_calls[0]

        assert tool_call["name"] == "search_medical_evidence"
        assert tool_call["args"]["patient_pregnancy_status"] == expected_status
def test_pregnancy_detection_returns_false_for_incidental_gestational_diabetes():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <ArticleTitle>
            Health disparities in endocrine disorders
        </ArticleTitle>
        <Abstract>
            <AbstractText>
                This review discusses endocrine disorders,
                including gestational diabetes and vitamin D deficiency.
            </AbstractText>
        </Abstract>
    </PubmedArticle>
    """)

    assert _is_pregnancy_related(article) is False
# ============================================================
# PubMed query builder tests
# ============================================================

def test_vitamin_d_queries_are_generated():
    queries = _build_pubmed_queries(
        "vitamin D deficiency in adults"
    )

    assert queries
    assert any("vitamin d" in q.lower() for q in queries)

def test_vitamin_d_queries_cover_research_aspects():
    queries = _build_pubmed_queries(
        "vitamin D deficiency in adults"
    )

    combined_queries = " ".join(queries).lower()

    assert "prevalence" in combined_queries
    assert "risk factors" in combined_queries
    assert "clinical outcomes" in combined_queries
def test_iron_queries_are_preserved():
    queries = _build_pubmed_queries(
        "iron deficiency anemia and ferritin"
    )

    assert any("iron deficiency anemia AND ferritin" == q for q in queries)
    assert any("transferrin saturation" in q for q in queries)


def test_b12_queries_are_preserved():
    queries = _build_pubmed_queries(
        "vitamin B12 deficiency and methylmalonic acid"
    )

    assert any("methylmalonic acid" in q for q in queries)
    assert any("homocysteine" in q for q in queries)


def test_pregnancy_queries_are_preserved():
    queries = _build_pubmed_queries(
        "iron deficiency anemia",
        patient_pregnancy_status="pregnant",
    )

    assert any("pregnancy" in q for q in queries)
    assert any("pregnant women" in q for q in queries)
def test_study_population_relevance_identifies_mixed_population():
    result = _study_population_relevance(
        ["Adult", "Child"],
        "adult",
    )

    assert result == "Mixed"

def test_review_focused_on_children_is_not_mixed_by_incidental_adult_mention():
    result = _study_population_relevance(
        ["Child", "Adolescent"],
        "adult",
        article_title="Vitamin D deficiency in childhood-onset inflammatory bowel disease",
        abstract=(
            "Most previous research concerns adult-onset disease. "
            "This review focuses on pediatric inflammatory bowel disease."
        ),
        article_types=["Review"],
    )

    assert result == "Child"


def test_review_covering_adults_and_children_is_mixed():
    result = _study_population_relevance(
        ["Adult", "Child"],
        "adult",
        article_title="Vitamin D deficiency across age groups",
        abstract=(
            "This review examines vitamin D deficiency in both "
            "adult and pediatric populations."
        ),
        article_types=["Review"],
    )

    assert result == "Mixed"
def test_study_population_relevance_handles_missing_age_groups():
    result = _study_population_relevance(
        [],
        "adult",
    )

    assert result == "Not specified"
def test_extract_article_types_returns_review():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <PublicationTypeList>
            <PublicationType>Review</PublicationType>
            <PublicationType>Systematic Review</PublicationType>
        </PublicationTypeList>
    </PubmedArticle>
    """)

    assert _extract_article_types(article) == [
        "Review",
        "Systematic Review",
    ]


def test_extract_article_types_returns_journal_article():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <PublicationTypeList>
            <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
    </PubmedArticle>
    """)

    assert _extract_article_types(article) == [
        "Journal Article"
    ]


def test_extract_article_types_handles_missing_types():
    import xml.etree.ElementTree as ET

    article = ET.fromstring("""
    <PubmedArticle>
        <ArticleTitle>Example article</ArticleTitle>
    </PubmedArticle>
    """)

    assert _extract_article_types(article) == []
def test_extract_study_design_identifies_meta_analysis():
    from graph_app import _extract_study_design

    result = _extract_study_design(
        "Vitamin D prevalence",
        "We performed a meta-analysis of previous studies.",
        ["Journal Article", "Meta-Analysis"],
    )

    assert result == "Meta-Analysis"


def test_extract_study_design_identifies_randomized_trial():
    from graph_app import _extract_study_design

    result = _extract_study_design(
        "Vitamin D supplementation",
        "This randomized controlled trial included 200 adults.",
    )

    assert result == "Randomized Controlled Trial"


def test_extract_study_design_identifies_cross_sectional():
    from graph_app import _extract_study_design

    result = _extract_study_design(
        "Vitamin D deficiency in adults",
        "We conducted a cross-sectional study.",
    )

    assert result == "Cross-Sectional Study"


def test_extract_study_design_returns_not_specified():
    from graph_app import _extract_study_design

    result = _extract_study_design(
        "Vitamin D deficiency",
        "The article discusses vitamin D deficiency.",
    )

    assert result == "Not specified"
def test_extract_study_design_does_not_misclassify_review_as_rct():
    from graph_app import _extract_study_design

    result = _extract_study_design(
        "Vitamin D and health outcomes: an overview",
        (
            "This review discusses findings from several "
            "randomized controlled trials."
        ),
        ["Journal Article", "Review"],
    )

    assert result == "Review"