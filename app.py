import streamlit as st
import tempfile
import os
import json
import base64
import re
import traceback
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

# ---------- Per-model timeout for one API call ----------
MODEL_TIMEOUT_SECONDS = 180

# ---------- File & Data Helpers ----------
def safe_parse_json(text: str) -> dict:
    """Parse JSON from a model response with light cleanup. Returns {} on failure."""
    if not text:
        return {}
    raw = text.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    # Try as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fallback: pull out the first balanced {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def safe_int(value, default=0) -> int:
    """Coerce a value to int, treating None/blank/non-numeric as default."""
    if value is None:
        return default
    if isinstance(value, bool):  # bool is a subclass of int — guard against it
        return default
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", "-", ""):
        return default
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def normalize_item_name(name: str) -> str:
    """Normalize an item name for cross-model vote matching.
    Lowercase, strip, collapse whitespace, drop common punctuation that
    models inconsistently include or omit (parentheses, dashes, periods)."""
    if not name:
        return ""
    s = str(name).lower().strip()
    # Remove parenthetical groups (e.g. "Pink Violator (3.5g)" -> "pink violator")
    s = re.sub(r"\([^)]*\)", " ", s)
    # Replace common separators with space
    s = re.sub(r"[\-_/.,]+", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


# ---------- Shared audit prompt ----------
# Models return what they SEE — actual values from the cropped strip and
# the corresponding expected values from the expected PDF. All comparison/math
# happens in Python downstream, so even if a model gets confused about the
# sign of a discrepancy, our recomputation in safe_int + difference fixes it.
AUDIT_PROMPT = """You are an inventory auditor. You have two inputs:

1. The EXPECTED PDF — a typed master list with item names and their expected quantities.
2. The CROPPED IMAGE — a narrow strip showing the item names column (left) and ONE single data column (right) of values for today's date.

For every item that appears on the EXPECTED PDF, find the matching row in the cropped strip and read the value in the right-hand data column.

CRITICAL RULES:
- "expected" is the printed quantity from the EXPECTED PDF (left input). NEVER change this.
- "actual" is the handwritten/printed quantity from the CROPPED STRIP'S right-hand column. NEVER read this from the expected PDF.
- A blank, empty, missing, or unreadable cell on the cropped strip means actual = 0.
- ONLY output items where expected != actual. Skip items that match.
- Match item names tolerantly (minor spelling and punctuation differences are fine).

Respond ONLY with valid JSON, no markdown, no prose:
{"items": [{"item": "string (name as printed on the expected PDF)", "expected": <integer>, "actual": <integer>}]}"""


# ---------- Per-model runners (raise on error so caller can log) ----------
def run_gemini(expected_path, cropped_path):
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(expected_path, "rb") as f1, open(cropped_path, "rb") as f2:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(data=f1.read(), mime_type="application/pdf"),
                types.Part.from_bytes(data=f2.read(), mime_type="application/pdf"),
                AUDIT_PROMPT,
            ],
        )
    return safe_parse_json(response.text).get("items", [])


def run_openai(expected_path, cropped_path):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    exp_b64 = pdf_to_base64_images(expected_path)
    crop_b64 = pdf_to_base64_images(cropped_path)

    content = [{"type": "text", "text": AUDIT_PROMPT}]
    for b64 in exp_b64 + crop_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    response = client.chat.completions.create(
        model="gpt-5.5",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    return safe_parse_json(response.choices[0].message.content).get("items", [])


def run_claude(expected_path, cropped_path):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    exp_b64 = pdf_to_base64_images(expected_path)
    crop_b64 = pdf_to_base64_images(cropped_path)

    content = [{"type": "text", "text": AUDIT_PROMPT}]
    for b64 in exp_b64 + crop_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    return safe_parse_json(text).get("items", [])


# ---------- Ensemble with status tracking and adaptive consensus ----------
def ensemble_audit(expected_path, cropped_path):
    """Run all three models in parallel, then vote.

    Returns:
        {
            "discrepancies": [...],     # consensus discrepancies
            "model_status": {           # per-model success/failure
                "gemini": "ok" | "<error message>",
                "openai": "ok" | "<error message>",
                "claude": "ok" | "<error message>",
            },
            "models_responded": int,    # number of models that returned data
            "consensus_threshold": int, # votes required for an item
        }
    """
    runners = {
        "gemini": run_gemini,
        "openai": run_openai,
        "claude": run_claude,
    }

    model_results = {}     # name -> list of items (only for models that succeeded)
    model_status = {}      # name -> "ok" or error string

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fn, expected_path, cropped_path): name
            for name, fn in runners.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                items = future.result(timeout=MODEL_TIMEOUT_SECONDS)
                model_results[name] = items if isinstance(items, list) else []
                model_status[name] = "ok"
            except concurrent.futures.TimeoutError:
                model_status[name] = f"timeout after {MODEL_TIMEOUT_SECONDS}s"
            except Exception as e:
                # Capture the error type and message but not the full stack trace
                model_status[name] = f"{type(e).__name__}: {e}"

    models_responded = len(model_results)

    # Adaptive consensus threshold:
    # - 3 models responded -> require 2 of 3 (majority)
    # - 2 models responded -> require both to agree
    # - 1 model responded -> accept its output as-is, with a warning shown to user
    # - 0 models responded -> no results
    if models_responded >= 2:
        consensus_threshold = 2
    elif models_responded == 1:
        consensus_threshold = 1
    else:
        consensus_threshold = 0

    # Vote: signature is (normalized_name, expected, actual)
    vote_tally = Counter()
    item_details = {}

    for items in model_results.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_name = item.get("item") or item.get("name") or ""
            if not raw_name:
                continue
            norm_name = normalize_item_name(raw_name)
            exp = safe_int(item.get("expected"))
            act = safe_int(item.get("actual"))
            # Skip phantom matches (model said it's a discrepancy but exp == act)
            if exp == act:
                continue
            signature = (norm_name, exp, act)
            vote_tally[signature] += 1
            # Keep the first non-empty original name we see for display
            if signature not in item_details:
                item_details[signature] = {
                    "item": str(raw_name).strip(),
                    "expected": exp,
                    "actual": act,
                    "difference": act - exp,
                }

    if consensus_threshold == 0:
        verified = []
    else:
        verified = [
            item_details[sig]
            for sig, votes in vote_tally.items()
            if votes >= consensus_threshold
        ]

    # Sort by item name for stable output
    verified.sort(key=lambda d: d["item"].lower())

    return {
        "discrepancies": verified,
        "model_status": model_status,
        "models_responded": models_responded,
        "consensus_threshold": consensus_threshold,
    }


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
        exp_path = act_path = crop_path = None
        try:
            with st.spinner("Processing documents..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as exp_tmp:
                    exp_tmp.write(expected_file.getvalue())
                    exp_path = exp_tmp.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as act_tmp:
                    act_tmp.write(actual_file.getvalue())
                    act_path = act_tmp.name

                crop_path = act_path.replace(".pdf", "_cropped.pdf")
                client = genai.Client(api_key=GEMINI_API_KEY)

                st.info("Step 1: AI mapping physical document layout...")
                col_info = locate_target_column(client, exp_path, act_path)

                if not col_info or "target_col_center_line" not in col_info:
                    st.error("Could not locate target column on the actual sheet. "
                             "Verify the date matches the expected PDF and try again.")
                    st.stop()

                target_date = col_info.get("target_date", "Unknown Date")
                st.info(f"Step 2: Cropping column for {target_date}...")
                apply_precision_crop(act_path, crop_path, col_info)

                st.info("Step 3: Querying Gemini, GPT-5.5, and Claude in parallel...")
                result = ensemble_audit(exp_path, crop_path)

            # Always surface per-model status so silent failures aren't invisible
            status_lines = []
            for name, status in result["model_status"].items():
                icon = "✅" if status == "ok" else "❌"
                status_lines.append(f"{icon} {name}: {status}")
            st.caption("Model status — " + "  ·  ".join(status_lines))

            responded = result["models_responded"]
            threshold = result["consensus_threshold"]

            if responded == 0:
                st.error("All three models failed. See model status above. Try again or check API keys / quotas.")
                st.stop()
            elif responded == 1:
                st.warning(
                    f"Only 1 of 3 models responded. Output is from that one model alone — "
                    f"no cross-validation was performed. Treat results with extra caution."
                )
            elif responded == 2:
                st.warning(
                    f"Only 2 of 3 models responded. Showing items both remaining models agreed on."
                )

            discrepancies = result["discrepancies"]
            if not discrepancies:
                st.success("Consensus reached.")
                st.balloons()
                st.subheader("✅ All items match.")
            else:
                st.success(
                    f"Consensus reached ({responded} of 3 models responded; "
                    f"{threshold} vote{'s' if threshold > 1 else ''} required per item)."
                )
                st.subheader(f"⚠️ Discrepancies Verified ({len(discrepancies)})")
                report_text = f"Inventory Discrepancies — {target_date}\n\n"
                for d in discrepancies:
                    report_text += (
                        f"- {d['item']}: actual {d['actual']}, "
                        f"expected {d['expected']} (variance {d['difference']:+d})\n"
                    )
                st.code(report_text, language="text")

        except Exception as e:
            st.error(f"An error occurred: {type(e).__name__}: {e}")
            with st.expander("Stack trace"):
                st.code(traceback.format_exc(), language="text")

        finally:
            for path in (exp_path, act_path, crop_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
