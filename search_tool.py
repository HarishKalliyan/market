from typing import List, Dict
import re

def search_attributes_in_text(attributes: List[str], file_texts: Dict[str, str]) -> Dict[str, List[Dict]]:
    """
    For each attribute, search each file's text and return matches with file info.
    """
    results = {}

    for attribute in attributes:
        results[attribute] = []
        pattern = re.compile(rf"\b{re.escape(attribute)}\b", re.IGNORECASE)

        for file_name, content in file_texts.items():
            for line in content.splitlines():
                if pattern.search(line):
                    results[attribute].append({
                        "file": file_name,
                        "line": line.strip()
                    })

    return results
