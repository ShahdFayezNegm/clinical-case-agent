import json
import os
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage

try:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
except Exception:
    api_key = ""

if api_key:
    os.environ["OPENROUTER_API_KEY"] = api_key

from graph_app import graph, _get_applicable_marker_groups

# ==================================================
# 1. Page Configuration
# ==================================================

st.set_page_config(
    page_title="Clinical Case Investigation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_INPUT_LENGTH = 4000

# ==================================================
# 2. Session State
# ==================================================

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "used_tools" not in st.session_state:
    st.session_state.used_tools = []

if "case_input" not in st.session_state:
    st.session_state.case_input = ""


# ==================================================
# 3. Custom Styling
# ==================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F5F7FA;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 250, 0.95);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #142B45;
    }

    [data-testid="stSidebar"] * {
        color: #F4F7FB;
    }

    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        padding: 0.8rem 0 0.3rem 0;
    }

    .sidebar-caption {
        color: #B7C7D8;
        font-size: 0.78rem;
        margin-bottom: 2rem;
    }

    .main-heading {
        font-size: 2rem;
        font-weight: 700;
        color: #172B4D;
        margin-bottom: 0.2rem;
    }

    .main-subheading {
        font-size: 0.95rem;
        color: #687B91;
        margin-bottom: 1.8rem;
    }

    .section-heading {
        color: #172B4D;
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
    }

    .section-description {
        color: #718096;
        font-size: 0.86rem;
        margin-bottom: 1rem;
    }

    .status-card {
        background: #FFFFFF;
        border: 1px solid #E3EAF1;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }

    .status-label {
        font-size: 0.76rem;
        color: #718096;
        margin-bottom: 0.35rem;
    }

    .status-value {
        color: #172B4D;
        font-size: 0.98rem;
        font-weight: 650;
    }


    .result-panel {
        background: #FFFFFF;
        border: 1px solid #E3EAF1;
        border-radius: 12px;
        padding: 1.4rem;
    }

    .footer-text {
        color: #8A98A8;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 2rem;
    }

    /* General buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 8px;
    font-weight: 600;
    min-height: 2.7rem;
}

/* Sidebar button */
[data-testid="stSidebar"] div.stButton > button {
    background-color: #1B3A59 !important;
    color: #FFFFFF !important;
    border: 1px solid #45627E !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #294D6D !important;
    color: #FFFFFF !important;
}

/* Main investigation button */
[data-testid="stFormSubmitButton"] button {
    background-color: #167D83 !important;
    color: #FFFFFF !important;
    border: 1px solid #167D83 !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background-color: #126A70 !important;
    border-color: #126A70 !important;
}

    div.stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #126A70;
        border-color: #126A70;
    }

    [data-testid="stTextArea"] textarea {
        background-color: #FFFFFF;
        border: 1px solid #DCE4EC;
        border-radius: 8px;
        color: #172B4D;
    }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E3EAF1;
        border-radius: 10px;
    }

    hr {
        border-color: #E3EAF1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 4. Sidebar
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">Clinical Research</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-caption">CASE INVESTIGATION WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Workspace")

    if st.button("＋  New Investigation", use_container_width=True):

        st.session_state.last_question = ""
        st.session_state.last_answer = None
        st.session_state.used_tools = []
        st.session_state.case_input = ""
        st.rerun()

    st.markdown("---")

    st.markdown("### About")

    st.caption(
        "An experimental clinical research assistant "
        "designed to support evidence investigation "
        "and structured case exploration."
    )

    st.markdown("---")

    st.caption("SYSTEM INFORMATION")
    st.caption("Workflow: LangGraph")
    st.caption("Cloud model: OpenRouter")
    st.caption("Mode: Research Prototype")


# ==================================================
# 5. Main Header
# ==================================================

st.markdown(
    '<div class="main-heading">Clinical Case Investigation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subheading">'
    'A structured workspace for clinical research and evidence exploration.'
    '</div>',
    unsafe_allow_html=True,
)


# ==================================================
# 6. Overview
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">WORKSPACE</div>
            <div class="status-value">Clinical Research</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">INVESTIGATION MODE</div>
            <div class="status-value">Evidence Exploration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    current_status = (
        "Investigation Completed"
        if st.session_state.last_answer
        else "Ready for Investigation"
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">CURRENT STATUS</div>
            <div class="status-value">{current_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==================================================
# 8. Clinical Case Input
# ==================================================

st.markdown(
    '<div class="section-heading">New Case Investigation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Describe a clinical case or enter a medical research question.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("clinical_case_form"):

    question = st.text_area(
        "Clinical case details",
        value=st.session_state.case_input,
        height=190,
        max_chars=MAX_INPUT_LENGTH,
        placeholder=(
            "Example: A fictional patient presents with fatigue "
            "and low hemoglobin. What medical evidence and possible "
            "explanations should be investigated?"
        ),
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button(
        "Start Investigation",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# 9. Execute Existing LangGraph Agent
# ==================================================

if submitted:

    question = question.strip()
    st.session_state.case_input = question

    if not question:

        st.warning("Please enter a clinical case or research question.")

    elif len(question) > MAX_INPUT_LENGTH:

        st.warning(
            f"Input must not exceed {MAX_INPUT_LENGTH} characters."
        )

    else:

        st.session_state.last_question = question
        st.session_state.last_answer = None
        st.session_state.used_tools = []

        try:

            with st.spinner(
                "Investigating the case and retrieving relevant information..."
            ):

                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=question)
                        ]
                    }
                )

                messages = result.get("messages", [])

                if not messages:
                    raise ValueError("The agent returned no messages.")

                final_message = messages[-1]
                answer = final_message.content

                if isinstance(answer, list):

                    answer = "\n".join(
                        block.get("text", str(block))
                        if isinstance(block, dict)
                        else str(block)
                        for block in answer
                    )

                if not isinstance(answer, str):
                    answer = str(answer)

                
                used_tools = []

                for message in messages:
                    if isinstance(message, ToolMessage):
                        tool_name = message.name

                        if (
                            tool_name
                            and tool_name not in used_tools
                        ):
                            used_tools.append(tool_name)

                st.session_state.used_tools = used_tools
                st.session_state.last_answer = answer
                st.rerun() 

        except Exception as exc:

            print(
                f"[ERROR] Investigation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            st.error(
                "The investigation could not be completed. "
                "Please check the application setup and try again."
            )
# ==================================================
# 10. Investigation Results
# ==================================================

if st.session_state.last_answer:

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Investigation Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Review the information returned by the investigation workflow.'
        '</div>',
        unsafe_allow_html=True,
    )

    answer = st.session_state.last_answer

    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        data = None

        # ----------------------------------------------
    # Structured Drug Interaction Results
    # ----------------------------------------------

    if (
        isinstance(data, dict)
        and "medications" in data
        and "interaction_found" in data
    ):

        status = data.get("status", "UNKNOWN")
        medications = data.get("medications", [])
        evidence = data.get("evidence", [])
        limitations = data.get("limitations", [])

        st.markdown("### Drug Interaction Assessment")

        st.metric("Assessment Status", status)

        st.write("**Medications:**")
        if medications:
            for medication in medications:
                st.write(f"- {medication}")
        else:
            st.write("No medications identified.")

        interaction_found = data.get("interaction_found")

        if interaction_found is True:
            st.markdown("**Potential interaction identified**")
        elif interaction_found is False:
            st.success(
                "No matching interaction was found in the local database."
            )
        else:
            st.markdown("**Interaction status: Uncertain**")

        st.markdown("### Evidence")
        if evidence:
            for item in evidence:
                st.write(f"- {item}")
        else:
            st.write("No evidence details were returned.")

        if limitations:
            with st.expander("Method Notes", expanded=False):
                for item in limitations:
                    st.write(f"- {item}")
        st.markdown("### Recommended Next Step")
        st.info(data.get("next_step", "Seek review by a qualified healthcare professional."))

    # ----------------------------------------------
    # Structured PubMed Results
    # ----------------------------------------------

    elif isinstance(data, dict) and "results" in data and "status" in data:

        status = data.get("status", "UNKNOWN")
        results = data.get("results", [])
        limitations = data.get("limitations", [])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Search Status",
                status
            )

        with col2:
            st.metric(
                "Studies Found",
                data.get("result_count", len(results))
            )

        with col3:
            st.metric(
                "Candidate Records",
                data.get("candidate_count", "N/A")
            )

        st.caption(
            f"Source: {data.get('source', 'Not specified')}"
        )

        st.markdown("### Search Details")

        st.write(
          f"**Search Query:** "
          f"{data.get('search_query', data.get('query', 'N/A'))}"
        )

        search_queries = data.get("search_queries", [])

        if search_queries:
         with st.expander("View Expanded Search Queries"):
          for index, search_query in enumerate(search_queries, start=1):
            st.write(f"{index}. {search_query}")

        if status == "NO_RESULTS":

            st.warning(
                "No matching PubMed records were returned. "
                "This does not prove that relevant medical evidence "
                "does not exist. Try refining the search query."
            )
            query_diagnostics = data.get(
                "query_diagnostics", []
            )

            if query_diagnostics:
                with st.expander(
                    "PubMed Diagnostic Details"
                ):
                    for index, diagnostic in enumerate(
                        query_diagnostics, start=1
                    ):
                        st.markdown(f"**Query {index}**")
                        st.write(
                            f"Search: {diagnostic.get('query', 'N/A')}"
                        )
                        st.write(
                            f"HTTP Status: {diagnostic.get('http_status', 'N/A')}"
                        )
                        st.write(
                            f"PMIDs returned: {diagnostic.get('pmid_count', 'N/A')}"
                        )
        elif status != "SUCCESS":

            st.warning(
                f"The search returned status: {status}"
            )

        # ------------------------------------------
        # Individual Study Cards
        # ------------------------------------------

        if results:

            st.markdown("### Retrieved Studies")

            for index, study in enumerate(results, start=1):

                title = study.get(
                    "title",
                    "Title not available"
                )

                title = study.get(
                    "title",
                    "Title not available"
                )

                pmid = str(study.get("pmid", "")).strip()
                source_url = str(study.get("source_url", "")).strip()

                journal = study.get(
                    "journal",
                    "Journal not available"
                )

                publication_date = study.get(
                    "publication_date",
                    "Date not available"
                )

                abstract = study.get(
                    "abstract",
                    "Abstract not available."
                )

                st.markdown(f"### {title}")

                st.caption(
                    f"{journal}  ·  PMID: {pmid if pmid.isdigit() else 'N/A'}"
                )

                if (
                    pmid.isdigit()
                    and source_url == f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ):
                    st.markdown(
                        f"[View publication on PubMed]({source_url})"
                    )

                with st.expander("Read Abstract"):
                    st.write(abstract)

                matched_terms = study.get("matched_terms", [])
                matched_markers = study.get("matched_markers", [])
                matched_queries = study.get("matched_queries", [])

                if matched_terms:
                    st.write(
                        "**Matched Terms:** "
                        + ", ".join(map(str, matched_terms))
                    )

                if matched_markers:
                    st.write(
                        "**Matched Markers:** "
                        + ", ".join(map(str, matched_markers))
                    )

                if matched_queries:
                    st.write(
                        "**Matched Queries:** "
                        + ", ".join(map(str, matched_queries))
                    )

                has_applicable_markers = bool(
                    _get_applicable_marker_groups(
                        st.session_state.last_question
                    )
                )

                marker_coverage = (
                    study.get("marker_coverage_score", "N/A")
                    if has_applicable_markers
                    else "N/A"
                )

                score_col1, score_col2, score_col3, score_col4 = st.columns(4)

                with score_col1:
                    st.caption(
                        f"Relevance Score: "
                        f"{study.get('relevance_score', 'N/A')}"
                    )

                with score_col2:
                    st.caption(f"Marker Coverage: {marker_coverage}")

                with score_col3:
                    st.caption(
                        f"Age Relevance: "
                        f"{study.get('age_relevance_score', 'N/A')}"
                    )

                with score_col4:
                    st.caption(
                        "Study Population: "
                        f"{study.get('study_population_relevance', 'Not specified')}"
                    )

                article_types = study.get("article_types", [])
                if article_types:
                    st.caption(
                        "Article Types: "
                        + ", ".join(map(str, article_types))
                    )
                else:
                    st.caption("Article Types: Not specified")

                study_design = study.get("study_design", "Not specified")
                st.caption(f"Study Design: {study_design}")

                if "pregnancy_related" in study:
                    st.caption(
                        "Pregnancy Related: "
                        + (
                            "Yes"
                            if study["pregnancy_related"]
                            else "No"
                        )
                    )

        # ------------------------------------------
        # Research Notes
        # ------------------------------------------

        if limitations:
            with st.expander("Research Notes", expanded=False):
                for limitation in limitations:
                    st.write(f"- {limitation}")

    # ----------------------------------------------
    # Fallback for Non-JSON Responses
    # ----------------------------------------------

    else:

        with st.container(border=True):

            st.markdown(answer)

    # ----------------------------------------------
    # Investigation Tools
    # ----------------------------------------------

    if st.session_state.used_tools:

        with st.expander(
            "Investigation Tools Used",
            expanded=False
        ):

            for tool_name in st.session_state.used_tools:
                st.write(f"• {tool_name}")


# ==================================================
# 11. Privacy and Footer
# ==================================================

st.markdown("---")

with st.expander("Privacy and Data Handling"):

    st.write(
        "This prototype does not intentionally save clinical cases "
        "to a database. The current question and result are held in "
        "temporary Streamlit session memory."
    )

    st.write(
        "Please use fictional or anonymized cases only."
    )

st.markdown(
    """
    <div class="footer-text">
        Clinical Research Workspace · Powered by LangGraph and OpenRouter
    </div>
    """,
    unsafe_allow_html=True,
)
