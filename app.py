import streamlit as st
import tempfile
import os
import json
from google import genai
from google.genai import types
import fitz

# --- CORE LOGIC (From your original script) ---
def safe_parse_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)

def locate_target_column(client, expected_path, actual_path):
    doc = fitz.open(actual_path)
    page = doc[0]
    w, h = page.rect.width, page.rect.height
    for i in range(0, 102, 2):
        x = w * (i / 100.0)
        page.draw_line(fitz.Point(x, 0), fitz.Point(x, h), color=(1, 0, 0), width=1)
    img_bytes = page.get_pixmap(dpi=150).tobytes("png")
    doc.close()

    prompt = """You are an expert inventory surveyor. Look at the attached ACTUAL inventory sheet. Read the target date from the EXPECTED PDF. Find that date in the ACTUAL sheet. Tell me the exact Red Line number (0-100) running through the CENTER of that date's column. Tell me the Red Line number where the Item Names column ends. Respond ONLY in JSON: {"target_date": "YYYY-MM-DD", "expected_header": "string", "names_end_line": 22, "target_col_center_line": 74}"""
    
    with open(expected_path, "rb") as f:
        exp_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(data=exp_bytes, mime_type="application/pdf"),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            prompt
        ]
    )
    return safe_parse_json(response.text)

def apply_precision_crop(input_pdf_path, output_pdf_path, col_info):
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

def run_audit(client, expected_path, cropped_path, col_info):
    prompt = f"""Compare EXPECTED to ACTUAL. We have physically cut out all empty space. Target numbers are glued next to item names. ONLY output items where there is a mathematical discrepancy. 1. If Exp=0 and Act is blank/0, MATCH. 2. If Exp=5 and Act=5, MATCH. 3. If Exp > 0 and Act is blank, this IS A DISCREPANCY (Act=0). Respond ONLY in JSON: {{"items": [{{"item": "string", "expected": 0, "actual": 0}}]}}"""
    
    with open(expected_path, "rb") as f1, open(cropped_path, "rb") as f2:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(data=f1.read(), mime_type="application/pdf"),
                types.Part.from_bytes(data=f2.read(), mime_type="application/pdf"),
                prompt
            ]
        )
    report = safe_parse_json(response.text)
    
    final_items = []
    for item in report.get('items', []):
        exp, act = item.get('expected', 0), item.get('actual', 0)
        if exp != act:
            item['difference'] = act - exp
            final_items.append(item)
    return final_items

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Inventory Agent", page_icon="🕵️‍♂️", layout="centered")
st.title("🕵️‍♂️ Inventory Agent")
st.markdown("Upload your daily sheets below to run the audit.")

# Secure API Key Input
api_key = st.text_input("Enter your Gemini API Key", type="password")

col1, col2 = st.columns(2)
with col1:
    expected_file = st.file_uploader("1. Expected PDF", type=["pdf"])
with col2:
    actual_file = st.file_uploader("2. Actual PDF", type=["pdf"])

if st.button("Run Audit", type="primary"):
    if not api_key:
        st.error("Please enter your API key!")
    elif not expected_file or not actual_file:
        st.error("Please upload both PDFs!")
    else:
        with st.spinner("Analyzing physical sheets..."):
            try:
                # Save uploaded files temporarily for PyMuPDF to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as exp_tmp:
                    exp_tmp.write(expected_file.read())
                    exp_path = exp_tmp.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as act_tmp:
                    act_tmp.write(actual_file.read())
                    act_path = act_tmp.name
                
                crop_path = act_path.replace(".pdf", "_cropped.pdf")
                client = genai.Client(api_key=api_key)

                # Execute pipeline
                st.info("Step 1: Locating target columns...")
                col_info = locate_target_column(client, exp_path, act_path)
                
                st.info(f"Step 2: Stitching columns for {col_info.get('target_date')}...")
                apply_precision_crop(act_path, crop_path, col_info)
                
                st.info("Step 3: Auditing discrepancies...")
                discrepancies = run_audit(client, exp_path, crop_path, col_info)

                # Output Results
                st.success("Audit Complete!")
                if not discrepancies:
                    st.balloons()
                    st.subheader("✅ All items match perfectly.")
                else:
                    st.subheader(f"⚠️ Discrepancies Found ({len(discrepancies)})")
                    report_text = f"Inventory Discrepancies — {col_info.get('target_date')}\n\n"
                    for d in discrepancies:
                        report_text += f"- {d.get('item')}: Actual {d.get('actual')}, Expected {d.get('expected')} (Var: {d.get('difference'):+d})\n"
                    st.code(report_text, language="text")

                # Cleanup temp files
                os.remove(exp_path)
                os.remove(act_path)
                os.remove(crop_path)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")