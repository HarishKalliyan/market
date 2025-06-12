from typing import Dict

def find_attributes_node(state: Dict) -> Dict:
    email_content = state.get("email_content")
    
    # Dummy attribute extraction logic
    attributes = ["ENPLANING", "DEPLANING", "PAX CARRIERS", "CARGO CARRIERS"]
    
    return {
        **state,
        "attributes": attributes
    }
