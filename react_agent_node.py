from typing import Dict
from langchain.agents import Tool, initialize_agent
from langchain_community.llms import FakeListLLM  
from search_tool import search_attributes_in_text

def react_extraction_node(state: Dict) -> Dict:
    """
    ReAct Agent node that uses LangChain tools to match attributes
    in extracted content from multiple files. It returns results with
    file name and matched lines.
    """
    attributes = state.get("attributes", [])
    file_texts = state.get("extracted_by_file", {})

    # Tool function performs the actual attribute matching
    def tool_func(_: str) -> str:
        matches = search_attributes_in_text(attributes, file_texts)
        return str(matches)

    # Tool registration
    tool = Tool(
        name="AttributeMatcher",
        func=tool_func,
        description="Use this tool to find attribute-related values in extracted file text."
    )

    # 🧠 Simulated LLM with valid ReAct format
    llm = FakeListLLM(responses=[
        # Must not include anything before Action:
        "Action: AttributeMatcher\nAction Input: ''",
        "Observation: Attribute matches returned.",
        "Final Answer: Done"
    ])

    # 🧠 ReAct agent with error handling enabled
    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=False,
        handle_parsing_errors=True  # ✅ Critical to retry on parser errors
    )

    # Agent will simulate planning + tool calling
    agent.invoke("Find all attribute-related data from the uploaded airport documents.")

    # Actual tool result (same as fake observation, but real)
    matched_results = search_attributes_in_text(attributes, file_texts)

    return {
        **state,
        "matched_results": matched_results
    }
