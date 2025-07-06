# ver.py
import base64
import os
from streamlit.components.v1 import html
import cv2
import json
import tempfile
import re
from PIL import Image
import streamlit as st
import pandas as pd
import pypdfium2 as pdfium
from dotenv import load_dotenv
import boto3
from langchain_aws import ChatBedrock

# Load environment
load_dotenv(".env")
AWS_KEY = os.getenv("aws_access_key")
AWS_SECRET = os.getenv("aws_secret_access_key")

# Bedrock LLM Client
client_bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name="us-east-1"
)

model = ChatBedrock(
    model="us.meta.llama4-scout-17b-instruct-v1:0",
    client=client_bedrock
)




def render_click_zoom_image(image_path, zoom_scale=2):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    html_code = f"""
    <style>
      .zoom-container {{
        position: relative;
        display: inline-block;
      }}
      .zoom-container img {{
        width: 100%;
        max-width: 900px;
        transition: transform 0.3s ease;
        cursor: zoom-in;
      }}
    </style>
    <div class="zoom-container">
      <img id="zoomable" src="data:image/jpeg;base64,{encoded}" onclick="handleZoom(event)" />
    </div>

    <script>
      let zoomed = false;

      function handleZoom(event) {{
        const img = event.target;
        if (!zoomed) {{
          const rect = img.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          const offsetX = x / img.width * 100;
          const offsetY = y / img.height * 100;

          img.style.transformOrigin = `${{offsetX}}% ${{offsetY}}%`;
          img.style.transform = "scale({zoom_scale})";
          img.style.cursor = "zoom-out";
          zoomed = true;
        }} else {{
          img.style.transform = "scale(1)";
          img.style.cursor = "zoom-in";
          zoomed = false;
        }}
      }}
    </script>
    """
    html(html_code, height=650)




def calculate_cost(input_tokens, output_tokens):
    return (input_tokens / 1000) * 0.00017 + (output_tokens / 1000) * 0.00066

@st.cache_data
def process_pdf(pdf_path):
    images = []
    doc = pdfium.PdfDocument(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        pil_image = page.render(scale=2).to_pil().convert('L')
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as img_file:
            pil_image.save(img_file.name, format="JPEG")
            images.append(img_file.name)
        page.close()
    doc.close()
    return images

@st.cache_data
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    h, w = img.shape[:2]
    max_dim = 1024
    new_w, new_h = (max_dim, int(h * max_dim / w)) if w > h else (int(w * max_dim / h), max_dim)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized).save(path, format="JPEG", quality=90)
    return path

def encode_image(path):
    with open(path, "rb") as f:
        return f.read()

def extract_from_image(image_path, client_bedrock):
    payload = encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"text": "Extract all VIN numbers from this image and return them as a list of strings in JSON format under key 'VINs'."},
            {"image": {"format": "jpeg", "source": {"bytes": payload}}}
        ]
    }]
    response = client_bedrock.converse(
        modelId="us.meta.llama4-scout-17b-instruct-v1:0",
        messages=messages
    )
    text = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    cost = calculate_cost(usage["inputTokens"], usage["outputTokens"])

    json_start, json_end = text.find('{'), text.rfind('}') + 1
    vins = []
    if json_start != -1 and json_end != -1:
        try:
            result = json.loads(text[json_start:json_end])
            vins = result.get("VINs", [])
            if not isinstance(vins, list):
                vins = [vins]
        except:
            pass

    vins += re.findall(r'\b[A-HJ-NPR-Z0-9]{17}\b', text, flags=re.IGNORECASE)
    return list(sorted(set(v.upper() for v in vins))), cost

def verify_vin_presence(vin, image_path, client_bedrock):
    payload = encode_image(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"text": f"Is the VIN number '{vin}' visibly present in this image? Respond with JSON: {{\"present\": true or false}}"},
            {"image": {"format": "jpeg", "source": {"bytes": payload}}}
        ]
    }]
    try:
        response = client_bedrock.converse(
            modelId="us.meta.llama4-scout-17b-instruct-v1:0",
            messages=messages
        )
        text = response["output"]["message"]["content"][0]["text"]
        json_start, json_end = text.find('{'), text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            result = json.loads(text[json_start:json_end])
            return result.get("present", False)
    except:
        pass
    return False

def process_page_background(img_path, idx, file_name, client_bedrock):
    try:
        preprocess_image(img_path)
        vins, cost = extract_from_image(img_path, client_bedrock)
        verified = {}
        for vin in vins:
            verified[vin] = verify_vin_presence(vin, img_path, client_bedrock)
        return {
            "file": file_name,
            "page": idx + 1,
            "VINs": vins,
            "cost": cost,
            "verified": verified,
            "success": True,
            "image_path": img_path
        }
    except Exception as e:
        return {
            "file": file_name,
            "page": idx + 1,
            "VINs": [],
            "cost": 0,
            "verified": {},
            "success": False,
            "error_message": f"{file_name} - Page {idx+1}: {e}",
            "image_path": img_path
        }

# ---------- Streamlit UI -------------
st.set_page_config("VIN Extractor", layout="wide", page_icon="📦")
st.markdown("<h1 style='text-align: center;font-size: 42px; color: #4A90E2;'> VIN Extractor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 28px;'>Upload your BOL PDFs or Images</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

if "extracted" not in st.session_state:
    st.session_state.extracted = []

if uploaded_files:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        if st.button("🔍 Start Extraction", use_container_width=True):
            all_extracted_data = []
            total_tasks = 0
            tasks = []

            for file in uploaded_files:
                suffix = file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                if suffix == "pdf":
                    images = process_pdf(tmp_path)
                else:
                    images = [tmp_path]
                tasks.append((file.name, images))
                total_tasks += len(images)

            progress_bar = st.progress(0)
            progress_text = st.empty()
            completed_tasks = 0

            for file_name, images in tasks:
                for idx, img in enumerate(images):
                    result = process_page_background(img, idx, file_name, client_bedrock)
                    if result["success"]:
                        all_extracted_data.append(result)
                    else:
                        st.warning(result["error_message"])
                    completed_tasks += 1
                    progress = completed_tasks / total_tasks
                    progress_bar.progress(min(progress, 1.0))
                    progress_text.text(f"Processing {completed_tasks}/{total_tasks} pages...")

            st.session_state.extracted = all_extracted_data
            st.session_state.slide_index = 0
            st.success("✅ Extraction complete!")
            progress_bar.empty()
            progress_text.empty()

# ---------- Display Results ----------
if st.session_state.extracted:
    all_extracted_data = st.session_state.extracted

    # Editable VIN table construction
    rows = []
    for item in all_extracted_data:
        if not item["VINs"]:
            rows.append({"File": item["file"], "Page": item["page"], "VIN": "NO VIN", "Verified": "❌"})
        else:
            for vin in item["VINs"]:
                verified = item.get("verified", {}).get(vin, False)
                rows.append({
                    "File": item["file"],
                    "Page": item["page"],
                    "VIN": vin,
                    "Verified": "✅" if verified else "❌"
                })

    df = pd.DataFrame(rows)

    st.divider()
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📋 All VINs Extracted")
        selected_file = st.selectbox("📁 Filter by File", ["All Files"] + df["File"].unique().tolist())
        vin_query = st.text_input("🔎 Search VIN")

        filtered_df = df.copy()
        if selected_file != "All Files":
            filtered_df = filtered_df[filtered_df["File"] == selected_file]
        if vin_query:
            filtered_df = filtered_df[filtered_df["VIN"].str.contains(vin_query.upper(), na=False)]

        edited_df = st.data_editor(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "File": st.column_config.TextColumn("File", width="small"),
                "Page": st.column_config.NumberColumn("Page", width="small"),
                "VIN": st.column_config.TextColumn("VIN (Editable)", width="medium"),
                "Verified": st.column_config.TextColumn("Verified", width="small")
            },
            key="vin_editor"
        )

        # Sync changes to session_state.extracted
        for i, row in edited_df.iterrows():
            for data in st.session_state.extracted:
                if data["file"] == row["File"] and data["page"] == row["Page"]:
                    data["VINs"] = [row["VIN"]] if row["VIN"] != "NO VIN" else []

        csv = edited_df.to_csv(index=False)
        xlsx_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        edited_df.to_excel(xlsx_tmp.name, index=False)

        exp1, exp2, exp3 = st.columns([1, 3, 1])
        with exp1:
            st.download_button("⬇️ CSV", csv, "filtered_vins.csv", mime="text/csv")
        with exp3:
            with open(xlsx_tmp.name, "rb") as f:
                st.download_button("⬇️ Excel", f, "filtered_vins.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_right:
        filtered_images = [d for d in all_extracted_data if selected_file == "All Files" or d["file"] == selected_file]

        if not filtered_images:
            st.info("No image available for selected file.")
        else:
            total = len(filtered_images)
            idx = st.session_state.slide_index
            image_info = filtered_images[idx]

            st.markdown(f"### 📄 Preview: {image_info['file']}")
            

            
            nav1, nav2, nav3 = st.columns([1, 3, 1])
            with nav1:
                if st.button("⬅️ Prev"):
                    st.session_state.slide_index = (idx - 1) % total
            with nav3:
                if st.button("➡️ Next"):
                    st.session_state.slide_index = (idx + 1) % total
            with nav2:
                st.markdown(f"<h5 style='text-align: center;font-size: 24px;'>Page {image_info['page']}</h5>", unsafe_allow_html=True)
            

            render_click_zoom_image(image_info["image_path"], zoom_scale=3.5)

