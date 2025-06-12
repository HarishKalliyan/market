# from langgraph.graph import StateGraph
# from text_extraction_node import text_extraction_node, ExtractionState
# # Create LangGraph
# graph = StateGraph(ExtractionState)
# graph.add_node("Text_extraction", text_extraction_node)
# graph.set_entry_point("Text_extraction")
# graph.set_finish_point("Text_extraction")  # Just one node for now

# # Compile
# extraction_graph = graph.compile()


# # Sample input
# input_data = {
#     "email_content": "This is a test email with important documents attached.",
#     "folder_name": "./attachments_folder"  # Ensure this folder exists with sample files
# }

# # Execute
# result = extraction_graph.invoke(input_data)

# # Output
# print("EMAIL CONTENT:")
# print(result["email_content"])
# print("\nEXTRACTED FILE CONTENT:")
# print(result["extracted_content"])


from langgraph.graph import StateGraph
from text_extraction_node import text_extraction_node
from find_attributes_node import find_attributes_node
from react_agent_node import react_extraction_node
import json
from export_utils import save_to_json, save_to_excel


def create_graph():
    graph = StateGraph(dict)  # Shared state

    graph.add_node("Text_extraction", text_extraction_node)
    graph.add_node("Find_Attributes", find_attributes_node)
    graph.add_node("ReAct_Extraction", react_extraction_node)

    # Define the flow of the graph
    graph.set_entry_point("Text_extraction")
    graph.add_edge("Text_extraction", "Find_Attributes")
    graph.add_edge("Find_Attributes", "ReAct_Extraction")
    graph.set_finish_point("ReAct_Extraction")

    return graph.compile()



graph = create_graph()

state = {
    "email_content": "Attached are the April stats for the airport.",
    "folder_name": "./attachments_folder"
}

final_state = graph.invoke(state)


print("\n--- EMAIL CONTENT ---\n", final_state["email_content"])
print("*"*1000)
print("\n--- EXTRACTED CONTENT ---\n", final_state["extracted_by_file"])
print("*"*1000)
print("\n--- ATTRIBUTES ---\n", final_state["attributes"])
print("*"*1000)
print("\n--- MATCHED RESULTS ---\n", json.dumps(final_state["matched_results"], indent=2))
print("*"*1000)

save_to_json(final_state["matched_results"], "matched_output.json")
print("*"*1000)
save_to_excel(final_state["matched_results"], "matched_output.xlsx")
print("*"*1000)
print("\n✅ Results saved to matched_output.json and matched_output.xlsx")