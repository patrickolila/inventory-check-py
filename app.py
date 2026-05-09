import streamlit as st
import tempfile
import os
import json
import base64
import concurrent.futures
from collections import Counter
import fitz  # PyMuPDF
from google import genai
from google.genai import types
import openai
import anthropic

# ---------- API Keys (Pulled directly from Streamlit Secrets) ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

# ---------- File & Data Helpers ----------
def safe_parse_json(text: str) -> dict:
    if not text: return {"items": []}
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try: return json.loads(raw)
    except: return {"items": []}

def pdf_to_base64_images(pdf_path: str):
    doc = fitz.open(pdf_path)
    b64_images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        b64_images.append(b64)
    doc.close()
    return b64_images

# ---------- The Crop Logic ----------
def locate_target_column(client, expected_path: str, actual_path: str):
    doc = fitz.open(actual_path)
    img_bytes_list = []
    
    # Scan the first 3 pages to ensure we catch the layout properly
    for i in range(min(3, len(doc))):
        page = doc[i]
        w, h = page.rect.width, page.rect.height
        for j in range(0, 102, 2):
            x = w * (j / 100.0)
            page.draw_line(fitz.Point(x, 0), fitz.Point(x, h), color=(1, 0, 0), width=1)
        img_bytes_list.append(page.get_pixmap(dpi=150).tobytes("png"))
    doc.close()

    prompt = """You are an expert inventory surveyor. Look at the attached ACTUAL inventory sheet images. Read the target date from the EXPECTED PDF. Find that date in the ACTUAL sheet images. Tell me the exact Red Line number (0-100) running through the CENTER of that date's column. Tell me the Red Line number where the Item Names column ends. Respond ONLY in JSON: {"target_date": "YYYY-MM-DD", "expected_header": "string", "names_end_line": 22, "target_col_center_line": 74}"""
    
    # Dynamically read the expected file bytes
    with open(expected_path, "rb") as f:
        exp_bytes = f.read()
        
    contents = [types.Part.from_bytes(data=exp_bytes, mime_type="application/pdf")]
    for img_bytes in img_bytes_list:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
    contents.append(prompt)

    response = client.models.generate_content(model="gemini-2.5-pro", contents=contents)
    return safe_parse_json(response.text)

def apply_precision_crop(input_pdf_path: str, output_pdf_path: str, col_info: dict):
    names_end = col_info.get("names_end_line", 22) / 100.0
    center = col_info.get("target_col_center_line", 74) / 100.0
    target_start, target_end = center - 0.04, center + 0.04

    src_doc = fitz.open(input_pdf_path)
    out_doc = fitz.open()

    for page in src_doc:
        w, h = page.rect.width, page.rect.height
        names_width = w * names_end
        target_width = w * (target_end - target_start)
        new_w = names_width + target_width
        
        out_page = out_doc.new_page(width=new_w, height=h)
        rect_names = fitz.Rect(0, 0, names_width, h)
        rect_target = fitz.Rect(w * target_start, 0, w * target_end, h)
        dest_names = fitz.Rect(0, 0, names_width, h)
        dest_target = fitz.Rect(names_width, 0, new_w, h)
        
        out_page.show_pdf_page(dest_names, src_doc, page.number, clip=rect_names)
        out_page.show_pdf_page(dest_target, src_doc, page.number, clip=rect_target)
        out_page.draw_line(fitz.Point(names_width, 0), fitz.Point(names_width, h), color=(0, 0, 0), width=2)
    
    out_doc.save(output_pdf_path)
    src_doc.close()
    out_doc.close()

# ---------- THE MULTI-MODEL ZOO ----------
def run_gemini(expected_path, cropped_path, prompt):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        with open(expected_path, "rb") as f1, open(cropped_path, "rb") as f2:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=f1.read(), mime_type="application/pdf"),
                    types.Part.from_bytes(data=f2.read(), mime_type="application/pdf"),
                    prompt
                ]
            )
        return safe_parse_json(response.text).get('items', [])
    except: return []

def run_openai(expected_path, cropped_path, prompt):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        exp_b64 = pdf_to_base64_images(expected_path)
        crop_b64 = pdf_to_base64_images(cropped_path)
        
        content = [{"type": "text", "text": prompt}]
        for b64 in exp_b64 + crop_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            
        response = client.chat.completions.create(
            model="gpt-5.5",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}]
        )
        return safe_parse_json(response.choices[0].message.content).get('items', [])
    except: return []

def run_claude(expected_path, cropped_path, prompt):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        exp_b64 = pdf_to_base64_images(expected_path)
        crop_b64 = pdf_to_base64_images(cropped_path)
        
        content = [{"type": "text", "text": prompt}]
        for b64 in exp_b64 + crop_b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
            
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}]
        )
        return safe_parse_json(response.content[0].text).get('items', [])
    except: return []

def ensemble_audit(expected_path, cropped_path):
    prompt = """Compare the EXPECTED master list to the ACTUAL cropped photo.
    CRITICAL RULES:
    1. DO NOT swap the Expected and Actual values. The EXPECTED quantity is printed on the master list. The ACTUAL quantity is the handwritten/printed number in the cropped image.
    2. Completely ignore the first column of the data strip because that is the column that lists the items, not the quantity. 
    3. When the actual value (act) is None or completely blank, treat it as zero (0).
    4. ONLY output items where there is a mathematical discrepancy (expected != actual). 
    Respond ONLY in JSON format EXACTLY like this: {"items": [{"item": "string", "expected": 0, "actual": 0}]}"""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_gemini = executor.submit(run_gemini, expected_path, cropped_path, prompt)
        f_openai = executor.submit(run_openai, expected_path, cropped_path, prompt)
        f_claude = executor.submit(run_claude, expected_path, cropped_path, prompt)

        gemini_items, openai_items, claude_items = f_gemini.result(), f_openai.result(), f_claude.result()

    all_findings = gemini_items + openai_items + claude_items
    vote_tally = Counter()
    item_details = {}

    for item in all_findings:
        if 'item' in item:
            name = str(item.get('item', 'Unknown')).upper().strip()
            exp = int(item.get('expected', 0))
            act = int(item.get('actual', 0))
            signature = f"{name} (Exp: {exp} | Act: {act})"
            vote_tally[signature] += 1
            item_details[signature] = {"item": name, "expected": exp, "actual": act, "difference": act - exp}

    # Verify consensus (Requires at least 2 out of 3 votes)
    verified_discrepancies = [item_details[sig] for sig, votes in vote_tally.items() if votes >= 2]
    return verified_discrepancies

# ---------- STREAMLIT USER INTERFACE ----------
st.set_page_config(page_title="Inventory Agent Pro", page_icon="🕵️‍♂️", layout="centered")
st.title("🕵️‍♂️ Inventory Agent Pro")
st.markdown("Powered by Multi-Model Consensus (Gemini + GPT-5.5 + Claude)")

if not GEMINI_API_KEY or not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    st.error("⚠️ Missing API Keys! Please configure Streamlit Secrets.")

col1, col2 = st.columns(2)
with col1:
    expected_file = st.file_uploader("1. Expected PDF", type=["pdf"])
with col2:
    actual_file = st.file_uploader("2. Actual PDF", type=["pdf"])

if st.button("Run Consensus Audit", type="primary"):
    if not expected_file or not actual_file:
        st.error("Please upload both PDFs!")
    else:
        with st.spinner("Processing documents..."):
            try:
                # Save uploaded files temporarily utilizing .getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as exp_tmp:
                    exp_tmp.write(expected_file.getvalue())
                    exp_path = exp_tmp.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as act_tmp:
                    act_tmp.write(actual_file.getvalue())
                    act_path = act_tmp.name
                
                crop_path = act_path.replace(".pdf", "_cropped.pdf")
                client = genai.Client(api_key=GEMINI_API_KEY)

                # Execute pipeline
                st.info("Step 1: AI mapping physical document layout...")
                col_info = locate_target_column(client, exp_path, act_path)
                
                st.info(f"Step 2: Cropping column for {col_info.get('target_date', 'Unknown Date')}...")
                apply_precision_crop(act_path, crop_path, col_info)
                
                st.info("Step 3: Firing queries to Gemini, GPT-5.5, and Claude simultaneously...")
                discrepancies = ensemble_audit(exp_path, crop_path)

                # Output Results
                st.success("Consensus Reached!")
                if not discrepancies:
                    st.balloons()
                    st.subheader("✅ All items match perfectly.")
                else:
                    st.subheader(f"⚠️ Discrepancies Verified ({len(discrepancies)})")
                    report_text = f"Inventory Discrepancies — {col_info.get('target_date', 'Unknown Date')}\n\n"
                    for d in discrepancies:
                        report_text += f"- {d.get('item')}: Actual {d.get('actual')}, Expected {d.get('expected')} (Var: {d.get('difference'):+d})\n"
                    st.code(report_text, language="text")

                # Cleanup temp files
                os.remove(exp_path)
                os.remove(act_path)
                os.remove(crop_path)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
