import os
import cv2, json
import time
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

import boto3
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load env + build bedrock client + instantiate Claude
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(".env")
AWS_KEY    = os.getenv("aws_access_key")
AWS_SECRET = os.getenv("aws_secret_access_key")

client_bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name="us-east-1"
)
model = ChatBedrock(
    model="anthropic.claude-sonnet-4-20250514-v1:0",
    client=client_bedrock
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Your existing helper functions: preprocess_image, encode_image, extract_entities_claude
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    # keep aspect but cap to 1024 × 720
    if w > h:
        new_w, new_h = 1024, 720
    else:
        new_h, new_w = 1024, 720

    img = cv2.resize(img, (new_w, new_h))
    out = Image.fromarray(img)
    out.save(file_path, format="JPEG")
    return file_path

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def calculate_cost(input_tokens, output_tokens):
    # Claude 3.7 Sonnet pricing (as of 2024)
    input_cost_per_1k = 0.003   # $3 per 1M tokens
    output_cost_per_1k = 0.015  # $15 per 1M tokens
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost

def extract_entities_claude(image_path):
    payload = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Extract all the information from the image provided. Give all in string not in json format."},
                {"image": {"format": "jpeg", "source": {"bytes": payload}}}
            ]
        }
    ]
    resp = client_bedrock.converse(
        # modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
        messages=messages
    )
    
    text = resp["output"]["message"]["content"][0]["text"]
    
    # Extract token usage and calculate cost
    usage = resp["usage"]
    input_tokens = usage["inputTokens"]
    output_tokens = usage["outputTokens"]
    cost = calculate_cost(input_tokens, output_tokens)
    
    return text, cost, input_tokens, output_tokens

# ─────────────────────────────────────────────────────────────────────────────
# 3) New: walk a folder of images, run your extractor, and print results
# ─────────────────────────────────────────────────────────────────────────────
def fetch_and_print(folder, max_workers=5):
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    def worker(path):
        start_time = time.time()
        
        pre = preprocess_image(path)
        out, cost, input_tokens, output_tokens = extract_entities_claude(pre)
        
        end_time = time.time()
        extraction_time = end_time - start_time
        
        print("\n" + "="*80)
        print(f"FILE: {path}")
        print(f"TIME: {extraction_time:.2f} seconds")
        print(f"COST: ${cost:.6f} (Input: {input_tokens} tokens, Output: {output_tokens} tokens)\n")
        print(out)
        print("="*80 + "\n")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        exe.map(worker, image_files)

if __name__ == "__main__":
    folder = "folder"   # <-- your folder of JPG/PNG
    fetch_and_print(folder, max_workers=10)

