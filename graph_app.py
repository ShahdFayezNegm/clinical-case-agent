import json
import os
import re
import uuid
import time
import requests
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any
from itertools import combinations
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "meta-llama/llama-3.2-3b-instruct"

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or "not-configured",
    temperature=0,
)


# ============================================================
# 2. MEDICATION ALIASES
# ============================================================

MEDICATION_ALIASES = {
    "acetaminophen": "acetaminophen",
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",

    "aspirin": "aspirin",
    "bayer": "aspirin",

    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "jantoven": "warfarin",

    "simvastatin": "simvastatin",
    "zocor": "simvastatin",

    "clarithromycin": "clarithromycin",
    "biaxin": "clarithromycin",

    "loratadine": "loratadine",
    "claritin": "loratadine",

    "amoxicillin": "amoxicillin",
    "amoxil": "amoxicillin",
}


KNOWN_MEDICATION_NAMES = set(MEDICATION_ALIASES.keys())


# ============================================================
# 3. LOCAL DEMONSTRATION INTERACTION DATABASE
# ============================================================

# This is a small educational demonstration database.
# It is NOT a complete or clinically validated drug database.

INTERACTION_DATABASE = {
    frozenset({"warfarin", "aspirin"}): (
        "Using aspirin with warfarin may increase bleeding risk."
    ),

    frozenset({"simvastatin", "clarithromycin"}): (
        "Clarithromycin may increase simvastatin exposure "
        "and the risk of muscle-related adverse effects."
    ),
}


# ============================================================
# 4. RESULT SCHEMA
# ============================================================

REQUIRED_FIELDS = {
    "status",
    "medications",
    "interaction_found",
    "evidence",
    "limitations",
    "next_step",
}

ALLOWED_STATUSES = {
    "UNKNOWN_MEDICATION",
    "INSUFFICIENT_INFORMATION",
    "POTENTIAL_INTERACTION",
    "NO_MATCH_FOUND",
}


def make_result(
    status: str,
    medications: list[str],
    interaction_found: bool | None,
    evidence: list[str],
    limitations: list[str],
    next_step: str,
) -> dict[str, Any]:
    """Create a result following the expected JSON schema."""

    return {
        "status": status,
        "medications": medications,
        "interaction_found": interaction_found,
        "evidence": evidence,
        "limitations": limitations,
        "next_step": next_step,
    }


def validate_result(result: Any) -> bool:
    """Check that a result follows the expected schema."""

    if not isinstance(result, dict):
        return False

    if not REQUIRED_FIELDS.issubset(result.keys()):
        return False

    if (
        not isinstance(result["status"], str)
        or result["status"] not in ALLOWED_STATUSES
    ):
        return False

    if not isinstance(result["medications"], list):
        return False

    if not all(
        isinstance(item, str) and item.strip()
        for item in result["medications"]
    ):
        return False

    if result["interaction_found"] not in (True, False, None):
        return False

    if not isinstance(result["evidence"], list):
        return False

    if not isinstance(result["limitations"], list):
        return False

    if not all(
        isinstance(item, str)
        for item in result["evidence"]
    ):
        return False

    if not all(
        isinstance(item, str)
        for item in result["limitations"]
    ):
        return False

    if (
        not isinstance(result["next_step"], str)
        or not result["next_step"].strip()
    ):
        return False

    return True
def validate_pubmed_result(result: Any) -> bool:
    """Validate the structure of a PubMed search result."""

    if not isinstance(result, dict):
        return False

    required_fields = {
        "status",
        "query",
        "results",
        "limitations",
    }

    if not required_fields.issubset(result.keys()):
        return False

    allowed_statuses = {
        "SUCCESS",
        "NO_RESULTS",
        "INVALID_QUERY",
        "SEARCH_ERROR",
    }

    if (
        not isinstance(result["status"], str)
        or result["status"] not in allowed_statuses
    ):
        return False

    if not isinstance(result["query"], str):
        return False

    if not isinstance(result["results"], list):
        return False

    if not isinstance(result["limitations"], list):
        return False

    if not all(
        isinstance(item, str) and item.strip()
        for item in result["limitations"]
    ):
        return False

    # Validate optional metadata.
    for field in ("search_query", "source"):
     if field in result and not isinstance(result[field], str):
        return False

    if "search_queries" in result:
     if (
        not isinstance(result["search_queries"], list)
        or not all(
            isinstance(item, str) and item.strip()
            for item in result["search_queries"]
        )
    ):
        return False

    # Validate result counts.
    for field in ("result_count", "candidate_count"):
        if field in result:
            value = result[field]

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return False

    # Validate individual PubMed articles.
        article_fields = {
        "pmid",
        "title",
        "journal",
        "publication_date",
        "abstract",
        "source_url",
        "relevance_score",
        "matched_terms",
        "matched_queries",
        "marker_coverage_score",
    }
    string_fields = {
        "pmid",
        "title",
        "journal",
        "publication_date",
        "abstract",
        "source_url",
    }

    for article in result["results"]:

        if not isinstance(article, dict):
            return False

        if not article_fields.issubset(article.keys()):
            return False

        if not all(
            isinstance(article[field], str)
            for field in string_fields
        ):
            return False

        score = article["relevance_score"]

        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
        ):
            return False

        for field in (
            "matched_terms",
            "matched_queries",
            "matched_markers",
            "article_types",
        ):
            if field in article:
                if not isinstance(article[field], list):
                    return False

                if not all(
                    isinstance(item, str)
                    for item in article[field]
                ):
                    return False

        for field in (
            "relevance_score",
            "marker_coverage_score",
            "age_relevance_score",
        ):
            if field in article:
                value = article[field]

                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                ):
                    return False

        for field in (
            "study_population_relevance",
            "study_design",
        ):
            if field in article:
                if not isinstance(article[field], str):
                    return False

        if (
            "pregnancy_related" in article
            and not isinstance(article["pregnancy_related"], bool)
        ):
            return False

    # Validate status and result consistency.
    if result["status"] == "SUCCESS":
        if not result["results"]:
            return False

    elif result["results"]:
        return False

    if "result_count" in result:
        if result["result_count"] != len(result["results"]):
            return False

    if (
        "candidate_count" in result
        and "result_count" in result
        and result["candidate_count"] < result["result_count"]
    ):
        return False

    return True
def serialize_result(result: dict[str, Any]) -> str:
    """Validate and serialize a result as JSON."""

    if not validate_result(result):
        raise ValueError("The result does not follow the expected schema.")

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 5. MEDICATION EXTRACTION
# ============================================================

def extract_medications(question: str) -> list[str]:
    """
    Extract recognized medication names and normalize aliases.

    Preserve the order in which medications appear in the question.

    Example:
        'Does Zocor interact with Biaxin?'
        -> ['simvastatin', 'clarithromycin']
    """

    question_lower = question.lower()
    found_medications = []

    # Match all aliases and preserve their positions.
    matches = []

    for alias, canonical_name in MEDICATION_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"

        for match in re.finditer(pattern, question_lower):
            matches.append(
                (match.start(), canonical_name)
            )

    # Sort matches according to their position in the question.
    matches.sort(key=lambda item: item[0])

    # Normalize aliases and remove duplicate medications.
    for _, canonical_name in matches:
        if canonical_name not in found_medications:
            found_medications.append(canonical_name)

    return found_medications


# Words that should not be treated as medication names.
NON_MEDICATION_WORDS = {
    "a",
    "an",
    "the",
    "other",
    "another",
    "medication",
    "medications",
    "medicine",
    "medicines",
    "drug",
    "drugs",
    "treatment",
    "treatments",
    "daily",
    "together",
    "combination",
    "interaction",
    "interactions",
    "safe",
    "safely",
    "risk",
    "risks",
    "problem",
    "problems",
    "side",
    "effects",
    "effect",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "did",
    "is",
    "are",
    "be",
    "take",
    "taking",
    "use",
    "using",
    "with",
    "and",
    "plus",
    "mix",
    "combine",
    "combining",
    "between",
    "for",
    "of",
    "to",
    "my",
    "your",
    "this",
    "that",
    "what",
    "which",
    "check",
    "whether",
    "explain",
    "tell",
    "me",
    "about",
    "interact",
    "interacts",
    "interacting",
    "other",
    "please",
    "food",
    "water",
    "meal",
    "meals",
    "drink",
    "drinks",
    "food",
    "water",
    "meal",
    "meals",
    "drink",
    "drinks",
    "unknown",
    "unrecognized",
}

def extract_unrecognized_medications(
    question: str,
) -> list[str]:
    """
    Heuristically detect unknown medication names near
    medication-pair connectors.

    Explicit unknown medication placeholders are also detected.
    """

    question_lower = question.lower()
    candidates = []

    # Detect explicit unknown medication placeholders.
    unknown_patterns = [
        r"\bunknown\s+(?:drug|medication|medicine)\b",
        r"\bunrecognized\s+(?:drug|medication|medicine)\b",
    ]

    for pattern in unknown_patterns:
        if re.search(pattern, question_lower):
            candidates.append("unknown medication")

    # Detect possible medication names around pair connectors.
    pair_pattern = (
    r"\b([a-z][a-z0-9-]*)\s+"
    r"(?:and|with|plus)\s+"
    r"(?:(?:a|an|the)\s+)?"
    r"\b([a-z][a-z0-9-]*)\b"
)

    for match in re.finditer(pair_pattern, question_lower):
        for word in match.groups():

            if word in NON_MEDICATION_WORDS:
                continue

            if word in KNOWN_MEDICATION_NAMES:
                continue

            if word not in candidates:
                candidates.append(word)

    return candidates

# ============================================================
# 6. ROUTING LOGIC
# ============================================================

INTERACTION_INTENT_PATTERN = re.compile(
    r"\b("
    r"interact(?:ion|ions|s|ing)?|"
    r"combine|combining|combined|"
    r"mix|mixing|mixed|"
    r"together|"
    r"combination|"
    r"drug interaction|"
    r"check.*(?:drug|medication)|"
    r"between.*(?:drug|medication)"
    r")\b",
    re.IGNORECASE,
)


def is_medication_question(question: str) -> bool:
    """
    Route explicit interaction questions to the interaction tool.

    Questions mentioning two recognized medications are also
    routed to the tool.

    General questions such as 'What is aspirin used for?'
    are sent to the language model instead.
    """

    if INTERACTION_INTENT_PATTERN.search(question):
        return True

    medications = extract_medications(question)

    if len(medications) >= 2:
        return True

    return False


# ============================================================
# 7. DRUG INTERACTION TOOL
# ============================================================
@tool
def check_drug_interactions(question: str) -> str:
    """
    Check a medication question against a small local
    demonstration database.

    Checks every possible pair of recognized medications.
    Returns structured JSON.
    """

    medications = extract_medications(question)

    unknown_medications = extract_unrecognized_medications(
        question
    )

    limitations = [
        "This is a limited educational demonstration database.",
        "The tool does not access a live clinical database.",
        "The result is not a diagnosis or personalized medical advice.",
    ]

    # Case 1: A likely unknown medication was found.
    if unknown_medications:
        result = make_result(
            status="UNKNOWN_MEDICATION",
            medications=medications + unknown_medications,
            interaction_found=None,
            evidence=[
                "One or more possible medication names were not "
                "recognized by the local alias dictionary."
            ],
            limitations=limitations + [
                "Name detection is heuristic and may mistake "
                "ordinary words for medication names."
            ],
            next_step=(
                "Verify the exact medication names and consult "
                "a pharmacist or qualified healthcare professional."
            ),
        )

        return serialize_result(result)

    # Case 2: No recognized medication.
    if not medications:
        result = make_result(
            status="UNKNOWN_MEDICATION",
            medications=[],
            interaction_found=None,
            evidence=[
                "No medication recognized by the local "
                "alias dictionary was found."
            ],
            limitations=limitations,
            next_step=(
                "Provide the exact generic or brand names "
                "of the medications."
            ),
        )

        return serialize_result(result)

    # Case 3: Only one recognized medication.
    if len(medications) < 2:
        result = make_result(
            status="INSUFFICIENT_INFORMATION",
            medications=medications,
            interaction_found=None,
            evidence=[
                "A medication interaction assessment generally "
                "requires the names of at least two medications."
            ],
            limitations=limitations,
            next_step=(
                "Provide the name of the other medication "
                "you want to check."
            ),
        )

        return serialize_result(result)

    # Case 4: Check ALL possible medication pairs.
    medication_pairs = list(combinations(medications, 2))

    found_interactions = []

    for medication_a, medication_b in medication_pairs:

        medication_pair = frozenset({
            medication_a,
            medication_b
        })

        interaction_message = INTERACTION_DATABASE.get(
            medication_pair
        )

        if interaction_message:
            found_interactions.append({
                "medications": [
                    medication_a,
                    medication_b
                ],
                "message": interaction_message
            })

    # Case 5: One or more interactions were found.
    if found_interactions:

        evidence = []

        for interaction in found_interactions:

            medication_a, medication_b = (
                interaction["medications"]
            )

            evidence.append(
                f"Potential interaction between "
                f"{medication_a} and {medication_b}: "
                f"{interaction['message']}"
            )

        evidence.append(
            f"Checked {len(medication_pairs)} medication pairs "
            f"against the local demonstration database."
        )

        result = make_result(
            status="POTENTIAL_INTERACTION",
            medications=medications,
            interaction_found=True,
            evidence=evidence,
            limitations=limitations + [
                "Only interactions listed in the local demonstration "
                "database can be detected.",
                "The database may omit other interactions and does not "
                "assess an individual's full medical history, dosage, "
                "or clinical circumstances."
            ],
            next_step=(
                "Ask a pharmacist or qualified healthcare "
                "professional to assess all medications. "
                "Do not start, stop, or change prescribed "
                "medications based only on this prototype."
            ),
        )

        return serialize_result(result)

    # Case 6: No matching interaction was found.
    result = make_result(
        status="NO_MATCH_FOUND",
        medications=medications,
        interaction_found=None,
        evidence=[
            f"No matching interaction was found among "
            f"{len(medication_pairs)} checked medication pairs "
            f"in the local demonstration database."
        ],
        limitations=limitations + [
            "NO_MATCH_FOUND does not mean the combination is safe "
            "or has no known interaction.",
            "Only interactions listed in the local demonstration "
            "database can be detected."
        ],
        next_step=(
            "Verify the combination with a pharmacist or "
            "qualified healthcare professional, especially "
            "before starting a new medication."
        ),
    )

    return serialize_result(result)

# ============================================================
# 7B. PUBMED MEDICAL EVIDENCE SEARCH TOOL
# ============================================================

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _xml_text(element: ET.Element, path: str) -> str:
    """Extract normalized text from an XML element."""

    found = element.find(path)

    if found is None:
        return ""

    return " ".join("".join(found.itertext()).split())


def _get_abstract(article: ET.Element) -> str:
    """Extract an article abstract, preserving section labels when present."""

    abstract_parts = []

    for part in article.findall(".//Abstract/AbstractText"):
        text = " ".join("".join(part.itertext()).split())

        if not text:
            continue

        label = part.attrib.get("Label")

        if label:
            text = f"{label}: {text}"

        abstract_parts.append(text)

    return " ".join(abstract_parts)


# Leading request phrases are removed so PubMed receives the research topic,
# not instructions such as "Find medical evidence about ...".

_EVIDENCE_PREFIXES = (
    r"^\s*(?:please\s+)?(?:find|search for|search|look up|retrieve)\s+",
    r"^\s*pubmed\s+(?:about|for|on)\s+",
    r"^\s*(?:medical|scientific|clinical)\s+evidence\s+(?:about|for|on|regarding)\s+",
    r"^\s*(?:clinical\s+)?(?:studies|research|literature|evidence)\s+(?:about|for|on|regarding)\s+",
    r"^\s*(?:can you|could you|please)\s+(?:find|search for|look up)\s+",
)

# Small, transparent synonym groups support ranking without requiring exact
# wording. They are deliberately used for ranking only, never as a hard filter.

_RELEVANCE_CONCEPTS = {
        "iron deficiency anemia": (
        "iron deficiency anemia",
        "iron-deficiency anemia",
        "iron deficiency anaemia",
        "iron-deficiency anaemia",
    ),

    "vitamin b12 deficiency": (
        "vitamin b12 deficiency",
        "vitamin b-12 deficiency",
        "b12 deficiency",
        "cobalamin deficiency",
    ),
    "aspirin": (
        "aspirin",
        "acetylsalicylic acid",
        "acetylsalicylic",
    ),

    "bleeding": (
        "bleeding",
        "hemorrhage",
        "haemorrhage",
        "hemorrhagic",
        "haemorrhagic",
        "blood loss",
    ),

    "gastrointestinal bleeding": (
        "gastrointestinal bleeding",
        "gastrointestinal hemorrhage",
        "gastrointestinal haemorrhage",
        "upper gastrointestinal bleeding",
        "lower gastrointestinal bleeding",
        "gi bleeding",
    ),

    "gastrointestinal malignancy": (
        "gastrointestinal malignancy",
        "malignancy-related gastrointestinal bleeding",
        "gastrointestinal cancer",
        "gastrointestinal neoplasm",
        "colorectal cancer",
        "gastric cancer",
        "malignancy",
    ),

    "risk/safety": (
        "risk",
        "safety",
        "adverse event",
        "adverse effect",
        "harm",
        "complication",
    ),

    "wearable technology": (
        "wearable",
        "wearables",
        "smartwatch",
        "smart watch",
        "fitbit",
        "wrist-worn",
    ),

    "arrhythmia": (
        "arrhythmia",
        "arrhythmias",
        "atrial fibrillation",
        "afib",
        "cardiac rhythm",
        "heart rhythm",
        "bradyarrhythmia",
        "tachyarrhythmia",
    ),

    "detection/monitoring": (
        "detection",
        "detecting",
        "screening",
        "diagnosis",
        "identification",
        "monitoring",
        "detect",
    ),

    "warfarin": (
        "warfarin",
        "coumadin",
        "jantoven",
    ),

    "drug interaction": (
        "drug interaction",
        "drug-drug interaction",
        "interacting",
        "coadministration",
        "concomitant use",
    ),

    "clinical study": (
        "clinical trial",
        "clinical study",
        "prospective study",
        "randomized",
        "randomised",
    ),

    "machine learning": (
        "machine learning",
        "deep learning",
        "neural network",
        "artificial intelligence",
        " ai ",
    ),
        "diabetic retinopathy": (
        "diabetic retinopathy",
        "diabetic retinal disease",
    ),

    "retinal fundus imaging": (
        "retinal fundus",
        "fundus photography",
        "fundus photograph",
        "retinal imaging",
        "fundus image",
    ),

    "diagnostic accuracy": (
        "diagnostic accuracy",
        "sensitivity",
        "specificity",
        "area under the curve",
        "auc",
        "receiver operating characteristic",
        "roc",
    ),

    "systematic review": (
        "systematic review",
        "systematic reviews",
        "meta-analysis",
        "meta analysis",
    ),
}


_GENERIC_TERMS = {
    "find",
    "medical",
    "scientific",
    "evidence",
    "about",
    "for",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "with",
    "a",
    "an",
    "is",
    "are",
    "be",
    "can",
    "could",
    "would",
    "should",
    "what",
    "how",
    "clinical",
    "studies",
    "study",
    "research",
    "pubmed",
    "please",
    "devices",
    "device",
    "using",
    "used",
    "use",
    "detecting",
    "detect",
    "risk",
    "risks",
    "related",
    "effect",
    "effects",
}

def extract_pubmed_topic(question: str) -> str:
    """Extract the core medical research topic from a user request."""

    topic = " ".join((question or "").strip().split())
    topic = re.sub(r"[?!.]+$", "", topic).strip()

    # Remove common request prefixes
    prefixes = (
        r"^\s*what\s+evidence\s+(?:is|are)\s+available\s+(?:regarding|about|for|on)\s+",
        r"^\s*(?:please\s+)?(?:find|search for|search|look up|retrieve)\s+",
        r"^\s*(?:published\s+)?(?:clinical\s+)?(?:studies|systematic\s+reviews)\s+(?:and\s+systematic\s+reviews\s+)?(?:reporting|that\s+report|about|on|regarding)\s+",
        r"^\s*pubmed\s+(?:about|for|on)\s+",
        r"^\s*(?:medical|scientific|clinical)\s+evidence\s+(?:about|for|on|regarding)\s+",
        r"^\s*(?:clinical\s+)?(?:studies|research|literature|evidence)\s+(?:about|for|on|regarding)\s+",
        r"^\s*(?:can you|could you|please)\s+(?:find|search for|look up)\s+",
    )

    for pattern in prefixes:
        topic = re.sub(pattern, "", topic, flags=re.IGNORECASE).strip()
        # Remove descriptive study-type prefixes
    topic = re.sub(
        r"^(?:published\s+)?clinical\s+studies\s+and\s+systematic\s+reviews\s+reporting\s+",
        "",
        topic,
        flags=re.IGNORECASE,
    ).strip()
    # Handle PubMed evidence requests
    topic = re.sub(
        r"^pubmed\s+evidence\s+(?:comparing|about|for|on|regarding)\s+",
        "",
        topic,
        flags=re.IGNORECASE,
    ).strip()

    # Remove instructions after the research topic
    topic = re.split(
        r"\.\s*(?:summarize|provide|explain|compare|include|list)\b",
        topic,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    topic = re.sub(r"^about\s+", "", topic, flags=re.IGNORECASE).strip()

    return topic or (question or "").strip()

def _extract_pubmed_date_range(question: str) -> dict | None:
    """Extract an explicitly requested publication date range."""

    text = (question or "").lower()

    match = re.search(
        r"\b(?:last|past)\s+(\d{1,2})\s+years?\b",
        text,
    )

    if match:
        years = int(match.group(1))

    elif re.search(r"\b(?:last|past)\s+year\b", text):
        years = 1

    elif re.search(
        r"\b(recent|recently|latest|newest)\b",
        text,
    ):
        years = 5

    else:
        return None

    if not 1 <= years <= 50:
        return None

    today = date.today()
    start_date = today.replace(year=today.year - years)

    return {
        "mindate": start_date.strftime("%Y/%m/%d"),
        "maxdate": today.strftime("%Y/%m/%d"),
        "datetype": "pdat",
    }
def _extract_target_population(question: str) -> str | None:
    """Extract an explicitly requested study population."""

    text = (question or "").lower()

    if re.search(
        r"\b(adolescents?|teenagers?|teenage)\b",
        text,
    ):
        return "adolescent"

    if re.search(
        r"\b(children|child|childhood|pediatric|paediatric|infants?)\b",
        text,
    ):
        return "child"

    if re.search(
        r"\b(adults?|adult population)\b",
        text,
    ):
        return "adult"

    return None
def _extract_patient_age(question: str) -> int | None:
    """Extract an explicitly stated patient age from a clinical case."""

    patterns = (
        r"\b(\d{1,3})\s*[- ]year[- ]old\b",
        r"\baged\s+(\d{1,3})\b(?!\s*(?:to|-)\s*\d{1,3}\b)",
        r"\bage\s*[:=]?\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*y\s*/\s*o\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            question or "",
            flags=re.IGNORECASE,
        )

        if match:
            age = int(match.group(1))

            if 0 <= age <= 120:
                return age

    return None


def _extract_patient_pregnancy_status(question: str) -> str:
    """Extract explicitly stated pregnancy status from a clinical case."""

    text = (question or "").lower()
    unknown_patterns = (
        r"\bpregnancy status is unknown\b",
        r"\bunknown pregnancy status\b",
        r"\bpregnancy status unknown\b",
        r"\buncertain pregnancy status\b",
        r"\bno evidence of pregnancy\b",
        r"\bnot sure if (?:she is )?pregnant\b",
        r"\bunsure if (?:she is )?pregnant\b",
        r"\bpregnancy test (?:was )?not performed\b",
        r"\bpregnancy test (?:was )?not done\b",
        r"\bpregnancy has not been confirmed\b",
        r"\bpossible pregnancy\b",
    )

    if any(
        re.search(pattern, text)
        for pattern in unknown_patterns
    ):
        return "unknown"
    current_not_pregnant_patterns = (
        r"\bis not currently pregnant\b",
        r"\bcurrently not pregnant\b",
        r"\bis not pregnant\b",
        r"\bisn't pregnant\b",
    )

    if any(
        re.search(pattern, text)
        for pattern in current_not_pregnant_patterns
    ):
        return "not_pregnant"
    current_pregnant_patterns = (
        r"\bis currently pregnant\b",
        r"\bis pregnant\b",
        r"\bcurrently pregnant\b",
        r"\bpregnant now\b",
        r"\bis expecting a baby\b",
    )

    if any(
        re.search(pattern, text)
        for pattern in current_pregnant_patterns
    ):
        return "pregnant"
    historical_pregnancy_patterns = (
        r"\bwas not pregnant\b",
        r"\bnot pregnant last year\b",
        r"\bwas pregnant\b",
        r"\bwere pregnant\b",
        r"\bpregnant last year\b",
        r"\bprevious pregnancy\b",
        r"\bpast pregnancy\b",
        r"\bhistory of pregnancy\b",
        r"\bhistory of pregnant\b",
        r"\bprior pregnancy\b",
        r"\bpregnancy history\b",
        r"\bhistory of pregnancy complications\b",
    )

    if any(
        re.search(pattern, text)
        for pattern in historical_pregnancy_patterns
    ):
        return "unknown"
    not_pregnant_patterns = (
     r"\bnot pregnant\b",
     r"\bnon[- ]pregnant\b",
     r"\bnot currently pregnant\b",
     r"\bdenies pregnancy\b",
     r"\bnegative pregnancy test\b",
    )
    if any(
     re.search(pattern, text)
     for pattern in not_pregnant_patterns
    ):
     return "not_pregnant"
    
    pregnant_patterns = (
        r"\bpregnant\b",
        r"\bpregnancy\b",
        r"\bexpecting a baby\b",
    )

    if any(
        re.search(pattern, text)
        for pattern in pregnant_patterns
    ):
        return "pregnant"

    return "unknown"

def _normalise_for_matching(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {re.sub(r'\s+', ' ', text).strip()} "

def is_valid_pubmed_topic(topic: str) -> bool:
    """Check whether the extracted topic contains meaningful terms."""

    normalized_topic = _normalise_for_matching(topic)

    terms = normalized_topic.strip().split()

    meaningful_terms = [
        term
        for term in terms
        if term not in _GENERIC_TERMS
    ]

    return len(meaningful_terms) >= 1


def _topic_concepts(topic: str) -> list[str]:
    """Map query wording to known concepts; retain distinctive unmatched terms."""

    normalised = _normalise_for_matching(topic)
    concepts = []

    for concept, variants in _RELEVANCE_CONCEPTS.items():
        if any(
            f" {variant.strip().lower()} " in normalised
            for variant in variants
        ):
            concepts.append(concept)

    # Prefer the specific gastrointestinal bleeding concept over generic bleeding.
    if (
        "gastrointestinal bleeding" in concepts
        and "bleeding" in concepts
    ):
        concepts.remove("bleeding")

    # Keep meaningful exact query terms too, but do not let generic request
    # words dominate ranking. These are matched literally against title/abstract.

    words = re.findall(
        r"[a-z][a-z0-9-]*",
        (topic or "").lower(),
    )

    all_known_variants = {
        token
        for variants in _RELEVANCE_CONCEPTS.values()
        for variant in variants
        for token in re.findall(
            r"[a-z][a-z0-9-]*",
            variant.lower(),
        )
    }

    for word in words:
        if (
            word not in _GENERIC_TERMS
            and len(word) > 2
            and word not in all_known_variants
            and word not in concepts
        ):
            concepts.append(word)

    return list(dict.fromkeys(concepts))


def _relevance_score(
    topic: str,
    title: str,
    abstract: str,
    original_question: str | None = None,
) -> tuple[int, list[str]]:
    """Score topic relevance using weighted concepts and title/abstract evidence.

    This is a transparent heuristic, not a validated search-ranking model or a
    measure of study quality. Specific clinical concepts count more than
    generic terms; title matches count more than abstract-only matches.
    """

    title_norm = _normalise_for_matching(title)
    abstract_norm = _normalise_for_matching(abstract)
    question_norm = _normalise_for_matching(
        original_question or ""
    )

    requests_causes = any(
        term in question_norm
        for term in (
            "underlying causes",
            "possible causes",
            "causes of",
            "etiology",
            "aetiology",
            "risk factors",
        )
    )

    requests_investigations = any(
        term in question_norm
        for term in (
            "further investigations",
            "additional investigations",
            "what investigations",
            "diagnostic evaluation",
            "workup",
            "work up",
            "further evaluation",
            "identify the underlying cause",
        )
    )
    concepts = _topic_concepts(topic)

    concept_weights = {
        "iron deficiency anemia": 8,
        "vitamin b12 deficiency": 8,
        "aspirin": 8,
        "bleeding": 8,
        "gastrointestinal bleeding": 12,
        "gastrointestinal malignancy": 12,
        "risk/safety": 2,
        "wearable technology": 7,
        "arrhythmia": 10,
        "detection/monitoring": 4,
        "warfarin": 8,
        "drug interaction": 7,
        "clinical study": 2,
        "machine learning": 7,
        "diabetic retinopathy": 10,
        "retinal fundus imaging": 7,
        "diagnostic accuracy": 9,
        "systematic review": 5,
    }

    score = 0
    matched = []

    for concept in concepts:
        variants = _RELEVANCE_CONCEPTS.get(
            concept,
            (concept,),
        )

        title_hit = any(
            f" {variant.strip().lower()} " in title_norm
            for variant in variants
        )

        abstract_hit = any(
            f" {variant.strip().lower()} " in abstract_norm
            for variant in variants
        )

        weight = concept_weights.get(concept, 3)

        if title_hit:
            score += weight * 2
            matched.append(concept)

        elif abstract_hit:
            score += weight
            matched.append(concept)
    if requests_causes:
        cause_groups = (
            (
                "cause: gastrointestinal blood loss",
                (
                    "gastrointestinal bleeding",
                    "gastrointestinal blood loss",
                    "occult gastrointestinal bleeding",
                    "gi bleeding",
                ),
            ),
            (
                "cause: menstrual blood loss",
                (
                    "menstrual blood loss",
                    "heavy menstrual bleeding",
                    "menorrhagia",
                ),
            ),
            (
                "cause: celiac disease or malabsorption",
                (
                    "celiac disease",
                    "coeliac disease",
                    "malabsorption",
                ),
            ),
            (
                "cause: ulcers or h pylori",
                (
                    "h pylori",
                    "helicobacter pylori",
                    "peptic ulcer",
                ),
            ),
            (
                "cause: gastrointestinal malignancy",
                (
                    "gastrointestinal malignancy",
                    "colorectal cancer",
                    "gastric cancer",
                ),
            ),
        )

        for label, terms in cause_groups:
            title_hit = any(
                _normalise_for_matching(term) in title_norm
                for term in terms
            )
            abstract_hit = any(
                _normalise_for_matching(term) in abstract_norm
                for term in terms
            )

            if title_hit:
                score += 8
                matched.append(label)
            elif abstract_hit:
                score += 4
                matched.append(label)

    if requests_investigations:
        investigation_groups = (
            (
                "investigation: gastrointestinal endoscopy",
                (
                    "gastrointestinal endoscopy",
                    "upper endoscopy",
                    "upper gastrointestinal endoscopy",
                    "colonoscopy",
                    "esophagogastroduodenoscopy",
                    "oesophagogastroduodenoscopy",
                ),
            ),
            (
                "investigation: celiac serology",
                (
                    "celiac serology",
                    "coeliac serology",
                    "tissue transglutaminase",
                    "anti t t g",
                ),
            ),
            (
                "investigation: biopsy",
                (
                    "duodenal biopsy",
                    "small bowel biopsy",
                ),
            ),
            (
                "investigation: urine testing",
                (
                    "urinalysis",
                    "urine testing",
                ),
            ),
        )

        for label, terms in investigation_groups:
            title_hit = any(
                _normalise_for_matching(term) in title_norm
                for term in terms
            )
            abstract_hit = any(
                _normalise_for_matching(term) in abstract_norm
                for term in terms
            )

            if title_hit:
                score += 8
                matched.append(label)
            elif abstract_hit:
                score += 4
                matched.append(label)
    # Give a modest bonus to the exact distinctive topic phrase.
    meaningful_topic_words = [
        word
        for word in re.findall(
            r"[a-z][a-z0-9-]*",
            (topic or "").lower(),
        )
        if word not in _GENERIC_TERMS and len(word) > 2
    ]

    if len(meaningful_topic_words) >= 2:
        phrase = " ".join(meaningful_topic_words)

        if f" {phrase} " in title_norm:
            score += 8
            matched.append(phrase)

        elif f" {phrase} " in abstract_norm:
            score += 3
            matched.append(phrase)

    # Reward coverage of multiple distinct topic concepts.
    matched_concepts = len(
        set(matched).intersection(concepts)
    )

    score += min(matched_concepts, 4)

    return score, list(dict.fromkeys(matched))


def _get_applicable_marker_groups(
    topic: str | None,
) -> dict[str, list[str]]:
    """Return laboratory markers relevant to the research topic."""

    all_markers = {
        "ferritin": [r"\bferritin\b"],
        "transferrin_saturation": [
            r"\btransferrin saturation\b"
        ],
        "methylmalonic_acid": [
            r"\bmethylmalonic acid\b",
            r"\bmma\b",
        ],
        "homocysteine": [r"\bhomocysteine\b"],
        "mcv": [
            r"\bmean corpuscular volume\b",
            r"\bmcv\b",
        ],
        "hba1c": [
            r"\bhba1c\b",
            r"\bhemoglobin a1c\b",
            r"\bglycated hemoglobin\b",
        ],
        "fasting_glucose": [
            r"\bfasting blood glucose\b",
            r"\bfasting plasma glucose\b",
            r"\bfpg\b",
        ],
        "oral_glucose_tolerance": [
            r"\boral glucose tolerance\b",
            r"\bogtt\b",
        ],
        "creatinine": [r"\bcreatinine\b"],
        "egfr": [
            r"\begfr\b",
            r"\bestimated glomerular filtration rate\b",
        ],
        "blood_urea_nitrogen": [
            r"\bblood urea nitrogen\b",
            r"\bbun\b",
        ],
        "cystatin_c": [r"\bcystatin c\b"],
        "alt": [
            r"\balanine aminotransferase\b",
            r"\balt\b",
        ],
        "ast": [
            r"\baspartate aminotransferase\b",
            r"\bast\b",
        ],
        "alp": [
            r"\balkaline phosphatase\b",
            r"\balp\b",
        ],
        "ggt": [
            r"\bgamma.?glutamyl transferase\b",
            r"\bggt\b",
        ],
        "bilirubin": [r"\bbilirubin\b"],
        "albumin": [r"\balbumin\b"],
        "tsh": [r"\bthyroid stimulating hormone\b", r"\btsh\b"],
        "free_t4": [
            r"\bfree thyroxine\b",
            r"\bfree t4\b",
            r"\bft4\b",
        ],
        "free_t3": [
            r"\bfree triiodothyronine\b",
            r"\bfree t3\b",
            r"\bft3\b",
        ],
        "thyroid_antibodies": [
            r"\bthyroid peroxidase antibodies\b",
            r"\btpo antibodies\b",
            r"\bthyroglobulin antibodies\b",
        ],
        "crp": [r"\bc reactive protein\b", r"\bcrp\b"],
        "esr": [
            r"\berythrocyte sedimentation rate\b",
            r"\besr\b",
        ],
        "procalcitonin": [r"\bprocalcitonin\b"],
        "troponin": [r"\btroponin\b"],
        "bnp": [
            r"\bb-type natriuretic peptide\b",
            r"\bbnp\b",
        ],
        "nt_probnp": [r"\bnt.?pro.?bnp\b"],
        "ck_mb": [
            r"\bck.?mb\b",
            r"\bcreatine kinase.?mb\b",
        ],
    }

    if topic is None:
        return all_markers

    normalized_topic = topic.lower()
    markers: dict[str, list[str]] = {}

    topic_groups = {
        "iron": (
            "iron deficiency",
            "ferritin",
            "transferrin",
        ),
        "b12": (
            "vitamin b12",
            "b12 deficiency",
            "cobalamin",
        ),
        "diabetes": (
            "diabetes",
            "diabetic",
            "hyperglycemia",
            "hypoglycemia",
            "blood glucose",
            "blood sugar",
        ),
        "kidney": (
            "kidney function",
            "kidney disease",
            "renal function",
            "renal disease",
            "chronic kidney",
        ),
        "liver": (
            "liver function",
            "liver disease",
            "hepatic function",
            "hepatic disease",
        ),
        "thyroid": (
            "thyroid",
            "hypothyroidism",
            "hyperthyroidism",
        ),
        "inflammation": (
            "inflammation",
            "inflammatory markers",
            "systemic inflammatory response",
        ),
        "cardiac": (
            "cardiac markers",
            "cardiac biomarkers",
            "myocardial infarction",
            "heart attack",
            "heart failure",
            "cardiac injury",
        ),
    }

    group_markers = {
        "iron": (
            "ferritin",
            "transferrin_saturation",
            "mcv",
        ),
        "b12": (
            "methylmalonic_acid",
            "homocysteine",
            "mcv",
        ),
        "diabetes": (
            "hba1c",
            "fasting_glucose",
            "oral_glucose_tolerance",
        ),
        "kidney": (
            "creatinine",
            "egfr",
            "blood_urea_nitrogen",
            "cystatin_c",
        ),
        "liver": (
            "alt",
            "ast",
            "alp",
            "ggt",
            "bilirubin",
            "albumin",
        ),
        "thyroid": (
            "tsh",
            "free_t4",
            "free_t3",
            "thyroid_antibodies",
        ),
        "inflammation": (
            "crp",
            "esr",
            "procalcitonin",
        ),
        "cardiac": (
            "troponin",
            "bnp",
            "nt_probnp",
            "ck_mb",
        ),
    }

    for group, terms in topic_groups.items():
        if any(term in normalized_topic for term in terms):
            for marker in group_markers[group]:
                markers[marker] = all_markers[marker]

    return markers


def _marker_coverage_score(
    title: str,
    abstract: str,
    topic: str | None = None,
) -> tuple[int, list[str]]:
    """Score topic-specific coverage of relevant laboratory markers."""

    text = f"{title} {abstract}".lower()
    markers = _get_applicable_marker_groups(topic)

    matched_markers = [
        marker
        for marker, patterns in markers.items()
        if any(re.search(pattern, text) for pattern in patterns)
    ]

    coverage_score = len(matched_markers) * 3

    return coverage_score, matched_markers

def _extract_age_groups(article: ET.Element) -> list[str]:
    """Extract age-group MeSH descriptors from a PubMed article."""

    age_groups = {
        "Child",
        "Infant",
        "Adolescent",
        "Adult",
        "Young Adult",
        "Middle Aged",
        "Aged",
        "Aged, 80 and over",
    }

    descriptors = article.findall(
        ".//MeshHeading/DescriptorName"
    )

    matched_groups = [
        descriptor.text
        for descriptor in descriptors
        if descriptor.text in age_groups
    ]

    return list(dict.fromkeys(matched_groups))
def _extract_article_types(article: ET.Element) -> list[str]:
    """Extract publication types from a PubMed article."""

    publication_types = article.findall(
        ".//PublicationTypeList/PublicationType"
    )

    return list(
        dict.fromkeys(
            " ".join("".join(item.itertext()).split())
            for item in publication_types
            if item.text
        )
    )
def _extract_study_design(
    article_title: str,
    abstract: str,
    article_types: list[str] | None = None,
) -> str:
    """Identify the likely study design from PubMed article metadata."""

    text = f"{article_title} {abstract}".lower()
    types = {item.lower() for item in (article_types or [])}

    has_meta_analysis = (
        "meta-analysis" in types
        or "meta-analysis" in text
    )
    has_systematic_review = (
        "systematic review" in types
        or "systematic review" in text
    )

    if has_meta_analysis and has_systematic_review:
        return "Systematic Review and Meta-Analysis"

    if has_meta_analysis:
        return "Meta-Analysis"

    if has_systematic_review:
        return "Systematic Review"

    if "validation study" in types:
        return "Validation Study"

    if re.search(
        r"\brandomi[sz]ed controlled trial\b"
        r"|\brandomi[sz]ed clinical trial\b",
        text,
    ):
        return "Randomized Controlled Trial"

    if re.search(r"\bcross-sectional\b", text):
        return "Cross-Sectional Study"

    if re.search(r"\bcase-control\b", text):
        return "Case-Control Study"

    if re.search(
        r"\bprospective\s+observational\s+study\b",
        text,
    ):
        return "Prospective Observational Study"

    if "observational study" in types or re.search(
        r"\bobservational study\b",
        text,
    ):
        return "Observational Study"

    if re.search(
        r"\bcohort study\b"
        r"|\bcohort studies\b"
        r"|\bprospective cohort\b"
        r"|\bretrospective cohort\b",
        text,
    ):
        return "Cohort Study"

    if re.search(r"\bcase report\b|\bcase series\b", text):
        return "Case Report/Series"

    if re.search(
        r"\bdevelopment and evaluation\b"
        r"|\bdeveloped and evaluated\b",
        text,
    ):
        return "Development and Evaluation Study"

    if "review" in types or re.search(r"\breview\b", text):
     return "Review"
    return "Not specified"


def _extract_age_ranges(text: str) -> list[tuple[int, int]]:
    """Extract all explicit participant age ranges."""

    patterns = (
        r"\baged\s+(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\s*years?\b",
        r"\bages?\s+(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\b",
        r"\bfrom\s+(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\s*years?\b",
    )

    ranges = []

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text or "",
            flags=re.IGNORECASE,
        ):
            lower_age = int(match.group(1))
            upper_age = int(match.group(2))

            if 0 <= lower_age < upper_age <= 120:
                age_range = (lower_age, upper_age)

                if age_range not in ranges:
                    ranges.append(age_range)

    return ranges

def _study_population_relevance(
    article_age_groups: list[str],
    target_population: str,
    article_title: str = "",
    abstract: str = "",
    article_types: list[str] | None = None,
) -> str:
    """Describe study population using article text and MeSH indexing."""

    text = f"{article_title} {abstract}".lower()
    population_text = re.sub(
    r"\badult-onset\b",
    "",
    text,
)

    title_text = article_title.lower()

    
    age_ranges = _extract_age_ranges(text)

    if age_ranges:
        has_child_range = any(
            lower_age < 12
            for lower_age, upper_age in age_ranges
        )

        has_adolescent_range = any(
            lower_age >= 12 and lower_age < 18
            for lower_age, upper_age in age_ranges
        )

        has_adult_range = any(
            lower_age >= 18 or upper_age > 18
            for lower_age, upper_age in age_ranges
        )

        has_adult_text = bool(
            re.search(r"\badults?\b", population_text)
            or re.search(r"\bnonpregnant adults?\b", population_text)
            or re.search(r"\bover 18 years\b", population_text)
            or re.search(r"\b18 years and older\b", population_text)
            or re.search(
                r"\b(?:elderly|geriatric)\b"
                r"|\bolder\s+(?:adults?|people|patients|individuals|persons)\b",
                population_text,
            )
        )

        has_adolescent_text = bool(
            re.search(r"\badolescents?\b", text)
        )

        has_child_text = bool(
            re.search(r"\bchild(?:ren)?\b", text)
            or re.search(r"\bpediatric\b", text)
            or re.search(r"\bpaediatric\b", text)
        )

        has_adult = has_adult_range or has_adult_text
        has_adolescent = has_adolescent_range or has_adolescent_text
        has_child = has_child_range or has_child_text

        if has_adult and (has_child or has_adolescent):
            return "Mixed"

        if has_adolescent:
            return "Adolescent"

        if has_child:
            return "Child"

        if has_adult:
            return "Adult"

    pregnancy_title_patterns = [
        r"\bpregnan\w*\b",
        r"\bgestational\b",
        r"\bantenatal\b",
        r"\bperinatal\b",
        r"\bprenatal\b",
        r"\bmaternal\b",
    ]

    if any(
        re.search(pattern, title_text)
        for pattern in pregnancy_title_patterns
    ):
        return "Pregnancy-related"

    adult_patterns = [
        r"\badults?\b",
        r"\badult population\b",
        r"\baged 18 years and above\b",
        r"\b18 years and older\b",
        r"\b(?:elderly|geriatric)\b",
        r"\bolder\s+(?:adults?|people|patients|individuals|persons)\b",
    ]

    adolescent_patterns = [
        r"\badolescents?\b",
        r"\bteenagers?\b",
        r"\bteenage\b",
    ]

    child_patterns = [
        r"\bchild(?:ren)?\b",
        r"\bchildhood\b",
        r"\bpediatric\b",
        r"\bpaediatric\b",
        r"\binfants?\b",
        r"\bchild-onset\b",
    ]

    has_adult_text = any(
        re.search(pattern, population_text) for pattern in adult_patterns
    )
    has_adolescent_text = any(
        re.search(pattern, text) for pattern in adolescent_patterns
    )
    has_child_text = any(
        re.search(pattern, text) for pattern in child_patterns
    )

    if has_adult_text and (has_child_text or has_adolescent_text):
        return "Mixed"
    if has_adult_text:
        return "Adult"
    if has_child_text:
        return "Child"
    if has_adolescent_text:
        return "Adolescent"

    if not article_age_groups:
        return "Not specified"

    adult_groups = {
        "Adult", "Young Adult", "Middle Aged",
        "Aged", "Aged, 80 and over",
    }

    child_groups = {
        "Child", "Infant",
    }

    adolescent_groups = {
        "Adolescent",
    }

    has_adult = bool(adult_groups.intersection(article_age_groups))
    has_child = bool(child_groups.intersection(article_age_groups))
    has_adolescent = bool(
        adolescent_groups.intersection(article_age_groups)
    )

    if has_adult and (has_child or has_adolescent):
        return "Mixed"
    if has_adult:
        return "Adult"
    if has_child:
        return "Child"
    if has_adolescent:
        return "Adolescent"

    return "Not specified"
def _is_pregnancy_related(article: ET.Element) -> bool:
    """Identify articles specifically related to pregnancy."""

    title = article.findtext(
        ".//ArticleTitle",
        default=""
    ).lower()

    if re.search(r"\b(pregnan(?:cy|t|cies))\b", title):
        return True

    abstract = _get_abstract(article).lower()
    article_types = _extract_article_types(article)

    pregnancy_context_patterns = [
        r"\bpregnant women with\b",
        r"\bpregnant women who\b",
        r"\bpregnant patients with\b",
        r"\bpregnant patients who\b",
        r"\bduring pregnancy\b",
        r"\bpregnancy outcomes\b",
        r"\bantenatal\b",
        r"\bprenatal\b",
        r"\bperinatal\b",
    ]

    return any(
        re.search(pattern, abstract)
        for pattern in pregnancy_context_patterns
    )



def _age_relevance_score(
    patient_age: int | None,
    article_age_groups: list[str],
    target_population: str | None = None,
    article_title: str = "",
    abstract: str = "",
) -> int:
    """Apply a conservative age-group relevance adjustment."""

    if patient_age is None and target_population is None:
        return 0

    article_text = f"{article_title} {abstract}".lower()
    age_ranges = _extract_age_ranges(article_text)

    if patient_age is not None:
        if patient_age < 1:
            expected_groups = {"Infant"}
        elif patient_age <= 12:
            expected_groups = {"Child"}
        elif patient_age <= 18:
            expected_groups = {"Adolescent"}
        elif patient_age <= 24:
            expected_groups = {"Young Adult", "Adult"}
        elif patient_age <= 44:
            expected_groups = {"Adult"}
        elif patient_age <= 64:
            expected_groups = {"Middle Aged"}
        else:
            expected_groups = {"Aged"}

    elif target_population == "adolescent":
        expected_groups = {"Adolescent"}
        mixed_age_groups = {
            "Child",
            "Adult",
            "Young Adult",
            "Middle Aged",
            "Aged",
            "Aged, 80 and over",
        }

        adult_only_patterns = (
            r"\baged\s*[â‰¥>=]+\s*18\s*years?\b",
            r"\baged\s+18\s+years?\s+or\s+older\b",
            r"\b18\s+years?\s+and\s+older\b",
            r"\b18\s+years?\s+or\s+older\b",
            r"\bat\s+least\s+18\s+years?\s+old\b",
        )

        if any(
            re.search(pattern, article_text)
            for pattern in adult_only_patterns
        ):
            return -3

        if age_ranges:
            adolescent_ranges = [
                (lower_age, upper_age)
                for lower_age, upper_age in age_ranges
                if lower_age <= 18 and upper_age >= 12
            ]

            if not adolescent_ranges or all(
              lower_age >= 18
              for lower_age, upper_age in age_ranges
            ):
             return -3

            if all(
                lower_age >= 12 and upper_age <= 18
                for lower_age, upper_age in adolescent_ranges
            ):
                return 3

            return 1

        if "Adolescent" in article_age_groups:
                return 1

        if article_age_groups:
            return -3

        if re.search(
            r"\b(adolescents?|teenagers?|teenage)\b",
            article_text,
        ):
            return 3

    elif target_population == "child":
        expected_groups = {"Child", "Infant"}

        if re.search(
            r"\b(children|child|childhood|pediatric|paediatric|infants?)\b",
            article_text,
        ):
            return 3

        if any(
            lower_age < 12
            for lower_age, upper_age in age_ranges
        ):
            return 3

    
    elif target_population == "adult":
        expected_groups = {
            "Adult",
            "Young Adult",
            "Middle Aged",
            "Aged",
            "Aged, 80 and over",
        }

        child_groups = {
            "Child",
            "Infant",
            "Adolescent",
        }

        has_adult_group = bool(
            expected_groups.intersection(article_age_groups)
        )
        has_child_group = bool(
            child_groups.intersection(article_age_groups)
        )

        # Mixed population: adults and children/adolescents.
        if has_adult_group and has_child_group:
            return 1

        # Child/adolescent-only study.
        if has_child_group:
            return -3

        # Adult-only study.
        if has_adult_group:
            return 3

        # Check explicit age ranges.
        if age_ranges:
            has_under_18 = any(
                lower_age < 18
                for lower_age, _ in age_ranges
            )
            has_adult_age = any(
                upper_age >= 18
                for _, upper_age in age_ranges
            )

            if has_under_18 and has_adult_age:
                return 1

            if has_adult_age:
                return 3

            if has_under_18:
                return -3

        has_adult_text = bool(
            re.search(r"\badults?\b", article_text)
        )
        has_child_text = bool(
            re.search(
                r"\b(children|child|childhood|pediatric|paediatric|infants?|adolescents?|teenagers?)\b",
                article_text,
            )
        )

        if has_adult_text and has_child_text:
            return 1

        if has_adult_text:
            return 3

        if has_child_text:
            return -3

    else:
        return 0
    if patient_age is not None:
        population_text = re.sub(
            r"\badult-onset\b",
            "",
            article_text,
        )

        has_adult_text = bool(
            re.search(
                r"\badults?\b|\b(?:elderly|geriatric)\b"
                r"|\bolder\s+(?:adults?|people|patients|individuals|persons)\b",
                population_text,
            )
        )

        has_child_text = bool(
            re.search(
                r"\bchild(?:ren)?\b|\bchildhood\b|\bpediatric\b"
                r"|\bpaediatric\b|\binfants?\b",
                article_text,
            )
        )

        has_adolescent_text = bool(
            re.search(
                r"\badolescents?\b|\bteenagers?\b|\bteenage\b",
                article_text,
            )
        )

        if has_adult_text and (
            has_child_text or has_adolescent_text
        ):
            return 1

        if has_adult_text:
            return 3

        if has_child_text or has_adolescent_text:
            return -3
    if expected_groups.intersection(article_age_groups):
        return 3

    if "Adult" in article_age_groups:
        if patient_age is not None and 25 <= patient_age <= 64:
            return 3
        return 0


    if article_age_groups:
        return -3

    return 0

def _build_pubmed_queries(
    topic: str,
    patient_pregnancy_status: str = "unknown",
    original_question: str | None = None,
) -> list[str]:
    """Build focused PubMed queries for supported medical research topics."""

    topic_lower = topic.lower()
    question_lower = (original_question or topic).lower()

    requests_diagnostic_accuracy = any(
        term in question_lower
        for term in (
            "diagnostic accuracy",
            "sensitivity",
            "specificity",
            "diagnostic performance",
        )
    )

    requests_causes = any(
        term in question_lower
        for term in (
            "underlying causes",
            "possible causes",
            "causes of",
            "etiology",
            "aetiology",
            "risk factors",
        )
    )
    requests_investigations = any(
        term in question_lower
        for term in (
            "further investigations",
            "additional investigations",
            "what investigations",
            "diagnostic evaluation",
            "workup",
            "work-up",
            "further evaluation",
        )
    )

    queries = []

    has_iron_evidence = any(
        term in topic_lower
        for term in (
            "iron deficiency",
            "ferritin",
            "transferrin saturation",
        )
    )

    has_b12_evidence = any(
        term in topic_lower
        for term in (
            "vitamin b12",
            "b12",
            "methylmalonic acid",
            "homocysteine",
        )
    )

    has_vitamin_d_evidence = any(
        term in topic_lower
        for term in (
            "vitamin d",
            "25-hydroxyvitamin d",
            "25(oh)d",
            "25 oh d",
        )
    )
    has_diabetic_retinopathy = (
        "diabetic retinopathy" in topic_lower
    )

    if has_diabetic_retinopathy:
        queries.extend([
            "diabetic retinopathy AND deep learning AND fundus",
            "diabetic retinopathy AND deep learning AND diagnostic accuracy",
            "diabetic retinopathy AND deep learning AND sensitivity AND specificity",
            "diabetic retinopathy AND deep learning AND systematic review",
        ])

    if has_iron_evidence:
        queries.extend([
            "iron deficiency anemia AND ferritin",
            "iron deficiency anemia AND transferrin saturation",
        ])

        if requests_causes:
            queries.extend([
                "iron deficiency anemia AND gastrointestinal bleeding",
                "iron deficiency anemia AND menstrual blood loss",
                "iron deficiency anemia AND celiac disease AND malabsorption",
                "iron deficiency anemia AND etiology",
            ])

        if requests_investigations:
            queries.extend([
                "iron deficiency anemia AND diagnostic evaluation",
                "iron deficiency anemia AND gastrointestinal endoscopy",
                "iron deficiency anemia AND celiac disease AND serology",
            ])

        if requests_diagnostic_accuracy:
            queries.extend([
                "iron deficiency anemia AND diagnostic accuracy AND ferritin",
                "iron deficiency anemia AND sensitivity AND specificity",
                "iron deficiency anemia AND diagnostic tests AND systematic review",
                "iron deficiency anemia AND diagnostic accuracy AND meta-analysis",
            ])

    if has_b12_evidence:
        queries.extend([
            "vitamin B12 deficiency AND methylmalonic acid",
            "vitamin B12 deficiency AND homocysteine",
        ])

    if has_iron_evidence and has_b12_evidence:
        queries.append(
            "iron deficiency anemia AND vitamin B12 deficiency AND MCV"
        )

    if has_vitamin_d_evidence:
        queries.extend([
            "vitamin D deficiency AND prevalence",
            "vitamin D deficiency AND risk factors",
            "vitamin D deficiency AND clinical outcomes",
            "vitamin D deficiency AND adults",
        ])

    if (
        has_iron_evidence
        and patient_pregnancy_status == "pregnant"
    ):
        queries.extend([
            "iron deficiency anemia AND pregnancy",
            "iron deficiency anemia AND pregnant women",
            "iron deficiency AND pregnancy AND ferritin",
            "iron supplementation AND pregnancy AND anemia",
        ])

    has_gi_bleeding = any(
        term in topic_lower
        for term in (
            "gastrointestinal bleeding",
            "gastrointestinal hemorrhage",
            "gastrointestinal haemorrhage",
            "gi bleeding",
        )
    )

    has_malignancy = any(
        term in topic_lower
        for term in (
            "malignancy",
            "malignancies",
            "cancer",
            "neoplasm",
            "neoplasms",
        )
    )

    if has_gi_bleeding and has_malignancy:
        queries.extend([
            "gastrointestinal bleeding AND malignancy",
            "gastrointestinal hemorrhage AND neoplasms",
            "cancer-related gastrointestinal bleeding",
        ])

    if not queries:
        queries.append(topic)

    return queries
def _select_diverse_articles(
    results: list[dict[str, Any]],
    limit: int = 5,
    patient_pregnancy_status: str = "unknown",
    target_population: str | None = None,

) -> list[dict[str, Any]]:
    """Select highly relevant articles while maintaining reasonable query diversity."""

    def article_score(item: dict[str, Any]) -> float:
        return (
            item["relevance_score"]
            + item.get("age_relevance_score", 0)
        )

    if patient_pregnancy_status == "pregnant":
        pregnancy_preference = lambda item: item.get(
            "pregnancy_related", False
        )
    else:
        pregnancy_preference = lambda item: not item.get(
            "pregnancy_related", False
        )

    def population_priority(item: dict[str, Any]) -> int:
        population = item.get(
            "study_population_relevance",
            "Not specified",
        )

        priorities_by_target = {
            "adult": {
                "Adult": 2,
                "Mixed": 1,
                "Not specified": 0,
                "Pregnancy-related": 0,
                "Child": -1,
                "Adolescent": -1,
            },
            "adolescent": {
                "Adolescent": 2,
                "Mixed": 1,
                "Not specified": 0,
                "Pregnancy-related": 0,
                "Adult": -1,
                "Child": -1,
            },
            "child": {
                "Child": 2,
                "Mixed": 1,
                "Not specified": 0,
                "Pregnancy-related": 0,
                "Adult": -1,
                "Adolescent": -1,
            },
        }

        priorities = priorities_by_target.get(target_population, {})
        return priorities.get(population, 0)
    def core_score(item: dict[str, Any]) -> float:
        return item["relevance_score"]

    best_core_score = max(
        (core_score(item) for item in results),
        default=0,
    )

    age_priority_threshold = 0.6

    ranked_results = sorted(
    results,
    key=lambda item: (
        core_score(item)
        >= best_core_score * age_priority_threshold,
        item.get("age_relevance_score", 0),
        population_priority(item),
        core_score(item),
        pregnancy_preference(item),
    ),
    reverse=True,
)
    selected_results = []
    covered_queries = set()
    rejected_by_score = []

    minimum_relative_score = 0.8

    for item in ranked_results:
        item_queries = set(item["matched_queries"])

        if not item_queries - covered_queries:
            continue

        remaining_results = [
            candidate
            for candidate in ranked_results
            if candidate not in selected_results
            and candidate.get("matched_queries")
        ]

        best_remaining_score = max(
            (article_score(candidate) for candidate in remaining_results),
            default=0,
        )

        if article_score(item) >= best_remaining_score * minimum_relative_score:
            selected_results.append(item)
            covered_queries.update(item_queries)
        else:
            rejected_by_score.append(item)

        if len(selected_results) == limit:
            break

    if len(selected_results) < limit:
        for item in ranked_results:
            if item in selected_results or item in rejected_by_score:
                continue

            selected_results.append(item)

            if len(selected_results) == limit:
                break

    return selected_results
@tool
def search_medical_evidence(
    query: str,
    patient_age: int | None = None,
    patient_pregnancy_status: str = "unknown",
    original_question: str | None = None,
) -> str:
    """Search PubMed for medical research and rank results by topic relevance.

    Returns publication metadata and original PubMed URLs. Ranking is heuristic,
    not a clinical review, quality appraisal, or treatment recommendation.
    """
    valid_pregnancy_statuses = {
        "pregnant",
        "not_pregnant",
        "unknown",
    }

    if patient_pregnancy_status not in valid_pregnancy_statuses:
        return (
            "Invalid patient_pregnancy_status. "
            "Use: pregnant, not_pregnant, or unknown."
        )

    query = (query or "").strip()

    if not query:
        return json.dumps(
            {
                "status": "INVALID_QUERY",
                "query": "",
                "search_query": "",
                "results": [],
                "limitations": [
                    "A non-empty search query is required."
                ],
            },
            ensure_ascii=False,
        )

    
    if len(query) > 300:
        return json.dumps(
            {
                "status": "INVALID_QUERY",
                "query": "",
                "search_query": "",
                "results": [],
                "limitations": [
                    "The search query must be 300 characters or fewer."
                ],
            },
            ensure_ascii=False,
        )

    context_question = (
        original_question or query
    ).strip()

    topic = extract_pubmed_topic(query)

    if not is_valid_pubmed_topic(topic):
     return json.dumps(
        {
            "status": "INVALID_QUERY",
            "query": query,
            "search_query": topic,
            "source": "PubMed / NCBI",
            "result_count": 0,
            "results": [],
            "limitations": [
                "The request does not contain a specific medical research topic.",
                "Please include a medical condition, medication, intervention, or research subject.",
            ],
        },
        ensure_ascii=False,
    )
    if patient_age is None:
        patient_age = _extract_patient_age(
            context_question
        )

    target_population = _extract_target_population(
        context_question
    )
    headers = {
        "User-Agent": "ClinicalCaseInvestigationAgent/1.0"
    }
    request_stage = "Preparing PubMed search"

    try:
# Build multiple focused queries, or fall back to the original topic.
        search_queries = _build_pubmed_queries(
            topic,
            patient_pregnancy_status=patient_pregnancy_status,
            original_question=context_question,
        ) or [topic]
        date_filter = _extract_pubmed_date_range(
            context_question
        )

        all_ids = []
        query_matches = {}
        query_diagnostics = []

        print(
            f"[PUBMED] Number of queries: {len(search_queries)}"
        )

        for query_index, search_query in enumerate(search_queries, start=1):
            request_stage = f"ESearch query {query_index}"
            for attempt in range(3):
                search_response = requests.get(
                    f"{PUBMED_BASE_URL}/esearch.fcgi",
                    params={
                        "db": "pubmed",
                        "term": search_query,
                        "retmax": 20,
                        "retmode": "json",
                        "sort": "relevance",
                        **(date_filter or {}),
                    },
                    headers=headers,
                    timeout=15,
                )

                if search_response.status_code != 429:
                    break

                if attempt < 2:
                    retry_after = search_response.headers.get("Retry-After")
                    try:
                        wait_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        wait_seconds = 5 * (2 ** attempt)
                    wait_seconds = max(0, min(wait_seconds, 60))
                    print(
                        f"[PUBMED] Query {query_index}: "
                        f"HTTP 429, retrying in {wait_seconds}s "
                        f"(attempt {attempt + 1}/3)"
                    )
                    time.sleep(wait_seconds)

            print(
                f"[PUBMED] Query {query_index}: "
                f"HTTP {search_response.status_code}"
            )

            search_response.raise_for_status()

            search_data = search_response.json()
            ids_for_query = search_data.get(
                "esearchresult", {}
            ).get("idlist", [])
            query_diagnostics.append(
                {
                    "query": search_query,
                    "http_status": search_response.status_code,
                    "pmid_count": len(ids_for_query),
                }
            )
            print(
                f"[PUBMED] Query {query_index}: "
                f"{len(ids_for_query)} PMID(s)"
            )

            for pmid in ids_for_query:
                if pmid not in query_matches:
                    all_ids.append(pmid)
                    query_matches[pmid] = []

                query_matches[pmid].append(search_query)

        print(
            f"[PUBMED] Total unique PMIDs: {len(all_ids)}"
        )
        ids = all_ids
        if not ids:
            return json.dumps(
                {
                    "candidate_count": 0,
                    "search_queries": search_queries,
                    "query_diagnostics": query_diagnostics,
                    "status": "NO_RESULTS",
                    "query": query,
                    "search_query": topic,
                    "source": "PubMed / NCBI",
                    "result_count": 0,
                    "results": [],
                    "limitations": [
                        "No matching PubMed records were returned.",
                        "No results do not prove that evidence does not exist.",
                        "Search uses a simplified extracted topic and may need refinement.",
                    ],
                },
                ensure_ascii=False,
            )


        request_stage = "EFetch article records"

        for attempt in range(3):
            fetch_response = requests.get(
                f"{PUBMED_BASE_URL}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                },
                headers=headers,
                timeout=20,
            )

            if fetch_response.status_code != 429:
                break

            if attempt < 2:
                retry_after = fetch_response.headers.get(
                    "Retry-After"
                )

                try:
                    wait_seconds = float(retry_after)
                except (TypeError, ValueError):
                    wait_seconds = 5 * (2 ** attempt)
                wait_seconds = max(0, min(wait_seconds, 60))
                print(
                    f"[PUBMED] EFetch HTTP 429, "
                    f"retrying in {wait_seconds}s "
                    f"(attempt {attempt + 1}/3)",
                    flush=True,
                )

                time.sleep(wait_seconds)

        print(
            f"[PUBMED] EFetch HTTP "
            f"{fetch_response.status_code}",
            flush=True,
        )

        fetch_response.raise_for_status()

        root = ET.fromstring(fetch_response.content)

        results = []

        for article in root.findall(".//PubmedArticle"):
            pmid = _xml_text(
                article,
                ".//MedlineCitation/PMID",
            )

            article_title = _xml_text(
                article,
                ".//MedlineCitation/Article/ArticleTitle",
            )

            journal_title = _xml_text(
                article,
                ".//MedlineCitation/Article/Journal/Title",
            )

            pub_date_element = article.find(
                ".//MedlineCitation/Article/Journal/JournalIssue/PubDate"
            )

            publication_date = ""

            if pub_date_element is not None:
                publication_date = " ".join(
                    "".join(
                        pub_date_element.itertext()
                    ).split()
                )

            abstract = _get_abstract(article)
            article_types = _extract_article_types(article)
            study_design = _extract_study_design(
               article_title,
               abstract,
               article_types,
            )

            score, matched_terms = _relevance_score(
                topic,
                article_title,
                abstract,
                original_question=context_question,
            )
            coverage_score, matched_markers = _marker_coverage_score(
               article_title,
               abstract,
                topic,
            )
            article_age_groups = _extract_age_groups(article)
            population_relevance = _study_population_relevance(
                article_age_groups,
                target_population or "",
                article_title,
                abstract,
                article_types,
            )

            age_score = _age_relevance_score(
                patient_age,
                article_age_groups,
                target_population,
                article_title,
                abstract,

            )
            pregnancy_related = _is_pregnancy_related(article)   
            results.append(
                {
                    "pmid": pmid,
                    "title": article_title or "Title unavailable",
                    "journal": journal_title or "Journal unavailable",
                    "publication_date": publication_date or "Not available",
                    "abstract": abstract or "Abstract unavailable",
                    "source_url": (
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        if pmid else ""
                    ),
                    "relevance_score": score,
                    "matched_terms": matched_terms,
                    "matched_queries": query_matches.get(pmid, []),
                    "marker_coverage_score": coverage_score,
                    "age_relevance_score": age_score,
                    "study_population_relevance": population_relevance,
                    "article_types": article_types,
                    "study_design": study_design,
                    "pregnancy_related": pregnancy_related,
                    "matched_markers": matched_markers,
                }
            )

               # Rank by relevance and marker coverage.
        results = _select_diverse_articles(
          results,
          patient_pregnancy_status=patient_pregnancy_status,
          target_population=target_population,
        )

        public_results = [
            article.copy()
            for article in results
        ]
        return json.dumps(
            {
                "query_diagnostics": query_diagnostics,
                "search_queries": search_queries,
                "status": "SUCCESS" if results else "NO_RESULTS",
                "query": query,
                "search_query": topic,
                "search_queries": search_queries,
                "source": "PubMed / NCBI",
                "result_count": len(results),
                "candidate_count": len(ids),
                "results": public_results,
                "limitations": [
                    "Relevance scores use weighted keyword/synonym heuristics; they are not a validated relevance model.",
                    "Ranking is not a clinical quality appraisal, peer-review verification, or treatment recommendation.",
                    "Studies are ranked, not filtered by relevance; related studies may use different terminology.",
                    "Only the top five of up to 20 retrieved records are displayed; relevant studies may be missed.",
                    "Some records do not include an abstract and publication dates may be incomplete.",
                ],
            },
            ensure_ascii=False,
        )

    except (
        requests.RequestException,
        ValueError,
        ET.ParseError,
    ) as exc:

        http_status = (
            exc.response.status_code
            if isinstance(exc, requests.HTTPError)
            and exc.response is not None
            else None
        )

        technical_detail = (
            f"{type(exc).__name__} during {request_stage}"
        )

        if http_status is not None:
            technical_detail += f" (HTTP {http_status})"

        print(f"[PUBMED ERROR] {technical_detail}: {exc}")

        return json.dumps(
            {
                "status": "SEARCH_ERROR",
                "query": query,
                "search_query": topic,
                "source": "PubMed / NCBI",
                "result_count": 0,
                "results": [],
                "limitations": [
                    "PubMed could not be reached or its response could not be parsed.",
                    f"Technical detail: {technical_detail}",
                ],
            },
            ensure_ascii=False,
        )

# ============================================================
# 8. GENERAL MEDICAL RESPONSE INSTRUCTIONS
# ============================================================

MEDICAL_SYSTEM_INSTRUCTIONS = """
You are an educational medical information assistant
inside a software prototype.

Safety and accuracy requirements:

1. Provide general educational information only.
2. Do not diagnose a person or prescribe medication.
3. Do not make personalized treatment decisions.
4. If asked whether someone should start, stop, or change
   a medication, explain that the decision requires advice
   from a qualified healthcare professional.
5. For questions about taking aspirin every day:
   - Do not recommend starting daily aspirin for primary
     prevention without clinician assessment.
   - Explain that benefits and bleeding risks vary by person.
   - If aspirin was prescribed after a heart attack, stroke,
     or stent procedure, tell the person not to stop it
     without consulting their prescriber.
6. Do not suggest that high blood pressure, high cholesterol,
   diabetes, or family history alone automatically means
   someone should take daily aspirin.
7. Do not claim that you checked a live clinical database.
8. Do not invent medical sources, citations, or research.
9. Clearly distinguish general information from personal advice.
10. Keep answers clear and reasonably concise.
11. If a question describes a possible emergency, advise
    seeking emergency medical care.
12. Do not claim that the project's local interaction database
    is complete or clinically validated.

This prototype is for educational and software-development
purposes, not for real clinical decision-making.

Answer the user's question below.
"""


# ============================================================
# 9. LANGGRAPH AGENT NODE
# ============================================================

def get_latest_user_question(messages: list[Any]) -> str:
    """Return the latest human message."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content

            if isinstance(content, str):
                return content

            return str(content)

    return ""

def extract_clinical_research_topic(question: str) -> str:
    """Extract a concise research topic from a clinical case."""

    normalized = " ".join(
        (question or "").lower().split()
    )

    has_iron_deficiency = any(
        term in normalized
        for term in (
            "iron deficiency",
            "ferritin",
            "serum iron",
            "transferrin saturation",
        )
    )

    has_b12_deficiency = any(
        term in normalized
        for term in (
            "vitamin b12",
            "b12 deficiency",
            "cobalamin",
        )
    )

    topics = []

    if has_iron_deficiency:
        topics.append("iron deficiency anemia")

    if has_b12_deficiency:
        topics.append("vitamin B12 deficiency")

    if len(topics) == 2:
        return "iron deficiency anemia AND vitamin B12 deficiency"

    if topics:
        return topics[0]

    return extract_pubmed_topic(question)


def is_evidence_question(question: str) -> bool:
    """Identify explicit requests for published medical evidence."""

    pattern = (
        r"\b("
        r"medical evidence|scientific evidence|evidence|"
        r"research|clinical studies|pubmed|medical literature|"
        r"clinical evidence|published studies|"
        r"systematic reviews?|meta-analyses?|"
        r"validation studies?|"
        r"prospective observational studies?|"
        r"diagnostic accuracy|"
        r"diagnostic performance|"
        r"clinical accuracy|"
        r"deep learning models"
        r")\b"
    )

    return bool(
        re.search(
            pattern,
            question,
            re.IGNORECASE,
        )
    )




def agent_node(state: MessagesState) -> dict[str, Any]:
    """
    Decide whether to search PubMed, use the drug interaction
    tool, or answer with the OpenRouter language model.
    """

    messages = state["messages"]

    question = get_latest_user_question(messages)

    if not question:
        return {
            "messages": [
                AIMessage(
                    content="Please enter a question."
                )
            ]
        }

    # Priority 1: Explicit medical evidence/research requests.
    # These requests go to PubMed even if they mention
    # two or more recognized medications.
    if is_evidence_question(question):
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_medical_evidence",
                                "args": {
                    "query": extract_clinical_research_topic(question),
                    "patient_age": _extract_patient_age(question),
                    "patient_pregnancy_status": _extract_patient_pregnancy_status(question),
                    "original_question": question,
                },
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                }
            ],
        )

        print("[ROUTING] use_tool: search_medical_evidence")

        return {
            "messages": [tool_call_message]
        }

    # Priority 2: Medication interaction questions.
    if is_medication_question(question):
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_drug_interactions",
                    "args": {
                        "question": question
                    },
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                }
            ],
        )

        print("[ROUTING] use_tool: check_drug_interactions")

        return {
            "messages": [tool_call_message]
        }

    # Priority 3: General questions go to OpenRouter.
    print("[ROUTING] use_llm")

    if not os.getenv("OPENROUTER_API_KEY"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The language model is not configured. "
                        "Please add OPENROUTER_API_KEY to the "
                        "application secrets."
                    )
                )
            ]
        }

    try:
        response = llm.invoke(
            [
                HumanMessage(
                    content=(
                        MEDICAL_SYSTEM_INSTRUCTIONS
                        + "\n\nUser question: "
                        + question
                    )
                )
            ]
        )

        return {
            "messages": [response]
        }

    except Exception as exc:
        print(
            f"[ERROR] OpenRouter request failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not generate a response because "
                        "the OpenRouter language model encountered "
                        "an error. Please try again later. "
                        "Technical detail: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            ]
        }

# ============================================================
# 10. GRAPH ROUTING
# ============================================================

def route_after_agent(state: MessagesState) -> str:
    """Choose the next graph node."""

    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"

    return "end"


# ============================================================
# 11. FORMAT TOOL RESULTS
# ============================================================

def final_answer_node(
    state: MessagesState,
) -> dict[str, Any]:
    """
    Validate and format the JSON returned by the tools.
    """

    tool_message = None

    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            tool_message = message
            break

    if tool_message is None:
        fallback = make_result(
            status="INSUFFICIENT_INFORMATION",
            medications=[],
            interaction_found=None,
            evidence=[
                "No tool result was available to validate."
            ],
            limitations=[
                "The interaction check could not be completed."
            ],
            next_step=(
                "Try the question again or consult a "
                "qualified healthcare professional."
            ),
        )

        return {
            "messages": [
                AIMessage(
                    content=serialize_result(fallback)
                )
            ]
        }

    try:
        content = tool_message.content

        if isinstance(content, str):
            result = json.loads(content)
        else:
            result = content

               # PubMed has its own result schema.
        if tool_message.name == "search_medical_evidence":

            if not validate_pubmed_result(result):
                raise ValueError(
                    "The PubMed result failed schema validation."
                )

            formatted_result = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )

            print(
                "[VALIDATION] PubMed result schema: PASS"
            )

        else:
            if not validate_result(result):
                raise ValueError(
                    "The tool result failed schema validation."
                )

            formatted_result = serialize_result(result)

            print(
                "[VALIDATION] JSON schema: PASS"
            )

        return {
            "messages": [
                AIMessage(
                    content=formatted_result
                )
            ]
        }

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:

        print(
            f"[VALIDATION] JSON schema: FAIL ({exc})"
        )

        # PubMed has its own result schema.
        if tool_message.name == "search_medical_evidence":

            fallback = {
                "status": "SEARCH_ERROR",
                "query": "",
                "search_query": "",
                "source": "PubMed / NCBI",
                "result_count": 0,
                "candidate_count": 0,
                "results": [],
                "limitations": [
                    "The PubMed tool returned a result that failed schema validation.",
                    "The returned evidence could not be safely processed.",
                    "Do not use this invalid result for medical decisions.",
                ],
            }

            return {
                "messages": [
                    AIMessage(
                        content=json.dumps(
                            fallback,
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                ]
            }

        # Medication interaction fallback.
        fallback = make_result(
            status="INSUFFICIENT_INFORMATION",
            medications=[],
            interaction_found=None,
            evidence=[
                "The tool returned a result that could not "
                "be validated against the expected JSON schema."
            ],
            limitations=[
                "The interaction result could not be validated."
            ],
            next_step=(
                "Try the question again. Do not use an invalid "
                "result to make medication decisions."
            ),
        )

        return {
            "messages": [
                AIMessage(
                    content=serialize_result(fallback)
                )
            ]
        }


# ============================================================
# 12. BUILD THE LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(MessagesState)

builder.add_node(
    "agent",
    agent_node,
)

builder.add_node(
    "tools",
    ToolNode(
        [
            check_drug_interactions,
            search_medical_evidence,
        ]
    ),
)

builder.add_node(
    "final_answer",
    final_answer_node,
)

builder.add_edge(
    START,
    "agent",
)

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "end": END,
    },
)

builder.add_edge(
    "tools",
    "final_answer",
)

builder.add_edge(
    "final_answer",
    END,
)

graph = builder.compile()


# ============================================================
# 13. PREDEFINED TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = {
    1: "Could warfarin and aspirin cause problems when used together?",
    2: "Is there an interaction between warfarin and aspirin?",
    3: "Can I take warfarin with aspirin?",
    4: "Is it safe to mix warfarin with aspirin?",
    5: "Is it risky to take simvastatin with clarithromycin?",
    6: "Check the risks of combining simvastatin and clarithromycin.",
    7: "Could paracetamol and loratadine interact?",
    8: "Can Tylenol interact with loratadine?",
    9: "Does Zocor interact with Biaxin?",
    10: "Can you check whether warfarin interacts with other medications?",
    11: "What is Machine Learning? Explain it in two simple sentences.",
    12: "What is aspirin used for?",
    13: "Explain what warfarin is.",
    14: "Should I take aspirin every day?",
    15: "Can you check whether warfarin interacts with amoxicillin?",
    16: "Find medical evidence about aspirin and bleeding risk.",
    17: "Find clinical studies about wearable devices for detecting arrhythmias.",
}


# ============================================================
# 14. INTERACTIVE CLI
# ============================================================

def run_question(question: str) -> None:
    """Run one question through the graph."""

    print("\n" + "=" * 65)
    print(f"Question: {question}")
    print("=" * 65)

    try:
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ]
            }
        )

        final_message = result["messages"][-1]

        print("\nAnswer:\n")
        print(final_message.content)

    except Exception as exc:
        print(
            "\n[ERROR] The graph could not process the question."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )


def main() -> None:
    """Display the menu and accept user questions."""

    print("=" * 65)
    print("CLINICAL CASE INVESTIGATION AGENT")
    print("=" * 65)
    print(f"Model: {MODEL_NAME}")
    print(
        "Educational prototype â€” not for clinical decision-making."
    )
    print()

    while True:
        print("\nPredefined questions:")

        for number, question in TEST_QUESTIONS.items():
            print(
                f"{number:>2}. {question}"
            )

        print("\n 0. Enter a custom question")
        print(" q. Quit")

        choice = input(
            "\nSelect an option: "
        ).strip().lower()

        if choice in {"q", "quit", "exit"}:
            print(
                "\nExiting the application. Goodbye!"
            )
            break

        if choice == "0":
            print(
                "\nEnter your question (type END on a new line when finished):"
            )

            question_lines = []

            while True:
                line = input()

                if line.strip().upper() == "END":
                    break

                question_lines.append(line)

            custom_question = "\n".join(
                question_lines
            ).strip()

            if custom_question:
                run_question(custom_question)
            else:
                print(
                    "The question cannot be empty."
                )

            continue

        try:
            selected_number = int(choice)

        except ValueError:
            print(
                "Invalid choice. Enter a number from 0 to 17, or q."
            )
            continue

        if selected_number not in TEST_QUESTIONS:
            print(
                "Invalid choice. Select a number from 0 to 17."
            )
            continue

        run_question(
            TEST_QUESTIONS[selected_number]
        )


if __name__ == "__main__":
    main()
