import json
import pandas as pd

def save_to_json(data: dict, filename="output.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def save_to_excel(data: dict, filename="output.xlsx"):
    rows = []
    for attr, matches in data.items():
        for match in matches:
            rows.append({
                "Attribute": attr,
                "File": match["file"],
                "Matched Line": match["line"]
            })
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
