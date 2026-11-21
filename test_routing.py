
from langchain_core.messages import AIMessage, HumanMessage

from graph_app import agent_node


def get_tool_name(question):
    result = agent_node({
        "messages": [
            HumanMessage(content=question)
        ]
    })

    message = result["messages"][-1]

    assert isinstance(message, AIMessage)
    assert message.tool_calls

    return message.tool_calls[0]["name"]


# --------------------------------------------------
# 1. Evidence request with two known medications
# --------------------------------------------------

def test_evidence_request_with_two_medications_uses_pubmed():

    tool_name = get_tool_name(
        "Find medical evidence about warfarin and aspirin."
    )

    assert tool_name == "search_medical_evidence"


# --------------------------------------------------
# 2. Clinical studies request with two medications
# --------------------------------------------------

def test_clinical_studies_request_uses_pubmed():

    tool_name = get_tool_name(
        "Find clinical studies about warfarin and aspirin."
    )

    assert tool_name == "search_medical_evidence"


# --------------------------------------------------
# 3. Explicit drug interaction request
# --------------------------------------------------

def test_interaction_question_uses_interaction_tool():

    tool_name = get_tool_name(
        "Does warfarin interact with aspirin?"
    )

    assert tool_name == "check_drug_interactions"


# --------------------------------------------------
# 4. Medication combination question
# --------------------------------------------------

def test_can_i_take_two_medications_uses_interaction_tool():

    tool_name = get_tool_name(
        "Can I take warfarin with aspirin?"
    )

    assert tool_name == "check_drug_interactions"
def test_evidence_request_passes_original_question_to_pubmed():
    question = (
        "Find medical evidence about iron deficiency anemia "
        "in adolescents published in the last 5 years."
    )
    result = agent_node({
        "messages": [
            HumanMessage(content=question)
        ]
    })

    message = result["messages"][-1]

    assert isinstance(message, AIMessage)
    assert message.tool_calls

    tool_call = message.tool_calls[0]

    assert tool_call["name"] == "search_medical_evidence"

    args = tool_call["args"]

    assert args["original_question"] == question
    assert args["patient_age"] is None
    assert "query" in args


def test_diagnostic_accuracy_research_question_uses_pubmed():
    question = (
        "Evaluate the diagnostic accuracy of deep learning models "
        "for detecting diabetic retinopathy from retinal fundus images. "
        "Compare sensitivity, specificity, AUC, and accuracy across "
        "different datasets, and identify systematic reviews, "
        "meta-analyses, validation studies, and prospective "
        "observational studies."
    )

    tool_name = get_tool_name(question)

    assert tool_name == "search_medical_evidence"

def test_evidence_available_question_uses_pubmed():
    question = (
        "What evidence is available regarding gastrointestinal "
        "bleeding and malignancy?"
    )

    assert get_tool_name(question) == "search_medical_evidence"


def test_medical_literature_question_uses_pubmed():
    question = (
        "What does the medical literature say about celiac disease "
        "and iron deficiency anemia?"
    )

    assert get_tool_name(question) == "search_medical_evidence"
