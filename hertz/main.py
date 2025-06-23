import os
import boto3
import json
import base64
from dotenv import load_dotenv
import time

# AWS Bedrock setup

AWS_REGION = 'us-east-1'  # Change as needed

# Load environment variables from .env file
# Bedrock model details for Anthropic Claude 3 Sonnet
BEDROCK_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'  # Claude 3 Sonnet model ID
# Bedrock model details for Llama 4 Scout
BEDROCK_MODEL_ID = 'meta.llama4-scout-v1'  # Llama 4 Scout model ID

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BEDROCK_ENDPOINT = 'bedrock-runtime.us-east-1.amazonaws.com'

# Folder containing handwritten images
def extract_text_from_image(image_path, bedrock_client):
    with open(image_path, 'rb') as img_file:
        image_bytes = img_file.read()

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    media_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract the handwritten text from this image."
                    }
                ]
            }
        ]
    })

    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response.get('body').read())
    return response_body.get('content', [{}])[0].get('text', '')
        # You may need to parse result depending on your model's output format

def main():
    bedrock_client = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
        endpoint_url=f'https://{BEDROCK_ENDPOINT}'
    )
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            text = extract_text_from_image(image_path, bedrock_client)
            print(f"Extracted from {filename}:\n{text}\n{'-'*40}")

if __name__ == '__main__':
    # Replace with your actual image folder or set via env
    IMAGE_FOLDER = "folder"

    # Claude 3 Sonnet 3.7 token rates (per 1K tokens, USD)
    COST_PER_K_INPUT = 0.015
    COST_PER_K_OUTPUT = 0.036

    bedrock_client = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
        endpoint_url=f'https://{BEDROCK_ENDPOINT}'
    )

    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(IMAGE_FOLDER, filename)

        # measure start time
        start = time.perf_counter()
        # extractor now returns full response_body including modelStats
        response_body = extract_text_from_image(image_path, bedrock_client)
        elapsed = time.perf_counter() - start

        # pull out modelStats if present
        stats = response_body.get('modelStats', {})
        in_toks = stats.get('inputTokens', 0)
        out_toks = stats.get('outputTokens', 0)

        # compute cost
        cost = (in_toks / 1000) * COST_PER_K_INPUT + (out_toks / 1000) * COST_PER_K_OUTPUT

        # extract text
        text = response_body.get('content', [{}])[0].get('text', '')

        print(f"Image: {filename}")
        print(f" → Time taken: {elapsed:.2f}s")
        print(f" → Tokens (in/out): {in_toks}/{out_toks}")
        print(f" → Estimated cost: ${cost:.6f}")
        print(f" → Extracted text:\n{text}")
        print("-" * 40)