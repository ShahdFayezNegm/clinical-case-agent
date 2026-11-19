
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


@tool
def check_drug_interactions(medications: list[str]) -> str:
    """Check a small educational database for selected drug interactions."""

    known_interactions = {
        frozenset(["warfarin", "aspirin"]):
            "Potential interaction: using warfarin with aspirin may "
            "increase bleeding risk. Clinical review is recommended."
    }

    normalized = {
        medication.strip().lower()
        for medication in medications
    }

    for pair, result in known_interactions.items():
        if pair.issubset(normalized):
            return result

    return (
        "No matching interaction was found in this small educational "
        "database. This does not establish that the medications are safe."
    )


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

tools = [check_drug_interactions]
tools_by_name = {item.name: item for item in tools}
llm_with_tools = llm.bind_tools(tools)

messages = [
    (
        "system",
        "Use the check_drug_interactions tool for the requested "
        "drug interaction check. Do not invent medication names."
    ),
    (
        "user",
        "Check the potential interaction between warfarin and aspirin."
    ),
]

try:
    # Step 1: Request a tool call.
    response = llm_with_tools.invoke(messages)
    print("STEP 1 — TOOL CALLS")
    print(response.tool_calls)

    if not response.tool_calls:
        print("The model did not request a tool.")
        raise SystemExit(0)

    messages.append(response)

    # Step 2: Execute the tool.
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        selected_tool = tools_by_name.get(tool_name)

        if selected_tool is None:
            print(f"Unknown tool: {tool_name}")
            continue

        if tool_name == "check_drug_interactions":
            medications = tool_args.get("medications")

            if (
                not isinstance(medications, list)
                or len(medications) < 2
                or not all(
                    isinstance(item, str) and item.strip()
                    for item in medications
                )
            ):
                print("Invalid tool arguments.")
                continue

        tool_result = selected_tool.invoke(tool_args)

        print("\nSTEP 2 — TOOL RESULT")
        print(tool_result)

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

    # Step 3: Ask the model to explain the tool result.
    print("\nSTEP 3 — FINAL ANSWER")

    final_response = llm.invoke([
        (
            "system",
            "Explain the supplied educational tool result clearly. "
            "Do not add unsupported medical claims. "
            "Mention that clinical review is recommended."
        ),
        (
            "user",
            f"Question: Check the interaction between warfarin and aspirin.\n"
            f"Tool result: {tool_result}"
        ),
    ])

    print(final_response.content)

except Exception as error:
    print("\nTEST FAILED")
    print(f"{type(error).__name__}: {error}")
