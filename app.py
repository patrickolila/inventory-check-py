import streamlit as st
import tempfile
import os
import json
import base64
import re
import traceback
import concurrent.futures
import time
from collections import Counter
import fitz  # PyMuPDF
from pypdf import PdfReader
from google import genai
from google.genai import types
import openai
import anthropic

# ============================================================================
# Config
# ============================================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

MODEL_TIMEOUT_SECONDS = 180

# Crop window — half-width on each side of the column center.
# Tune this for your sheet layout:
#   8-day  portrait sheet -> 0.06 (12% wide window, each col is ~8.75%)
#   10-day landscape sheet -> 0.05 (10% wide, each col is ~7%)
#   14-day landscape sheet -> 0.03 (6% wide, each col is ~5.7%)
CROP_HALF_WIDTH = 0.06


# ============================================================================
# JSON / type helpers
# ============================================================================
def safe_parse_json(text: str) -> dict:
    if not text:
        return {}
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def safe_int(value, default=0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
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
    if not name:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[\-_/.,]+", " ", s)
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


# ============================================================================
# Expected PDF text extraction
# ============================================================================
def extract_expected_items(pdf_path: str):
    reader = PdfReader(pdf_path)
    raw_lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for ln in text.splitlines():
            ln = ln.strip()
            if ln:
                raw_lines.append(ln)

    if not raw_lines:
        return [], ""

    header_text = raw_lines[0]
    items = []
    current_category = ""

    NOISE_PREFIXES = ("TEMP", "*(")
    DATE_PREFIX = re.compile(
        r"^(Aug|OCT|MARCH|JAN|FEB|APR|MAY|JUN|JUL|SEP|NOV|DEC)\s",
        re.IGNORECASE,
    )
    item_pattern = re.compile(r"^(.*?)(\d+)\s*$")

    def strip_legend_marker(s: str) -> str:
        if s.endswith(" O"):
            return s[:-2].rstrip()
        if s.endswith("O") and len(s) >= 2 and not s[-2].isdigit() and not s[-2].isspace():
            return s[:-1].rstrip()
        return s

    def is_section_header(s: str) -> bool:
        if s.endswith(" O"):
            return True
        if s.endswith("O") and len(s) >= 2:
            prev = s[-2]
            if prev.isalpha() and prev != "o":
                return True
            if prev == ")":
                return True
        if s in ("BUUDA 100MG", "KITS"):
            return True
        return False

    for ln in raw_lines[1:]:
        ln_upper = ln.upper()
        if any(ln_upper.startswith(p) for p in NOISE_PREFIXES):
            continue
        if DATE_PREFIX.match(ln):
            continue
        if ln_upper == "LEGEND:":
            continue

        if is_section_header(ln):
            current_category = strip_legend_marker(ln)
            continue

        m = item_pattern.match(ln)
        if m and m.group(2):
            name_raw = m.group(1).rstrip()
            qty_str = m.group(2)
            qty = int(qty_str)
            if not name_raw:
                continue

            # Detect pypdf extraction glitches where a character of the name
            # got glued to the quantity. The signature is:
            #   - No whitespace separator between name and quantity, AND
            #   - Quantity is implausibly large (>= 30) for a glued-letter
            #     parse to be the real qty
            # Why both conditions? pypdf often drops the space between name
            # and qty even when the parse is correct (e.g. "Raspberry3" really
            # is qty 3). But when the qty is large AND glued, it's almost
            # certainly a lost-character glitch like "Confectionary 0" being
            # mangled to "Confectionar70" (the "y" got absorbed into "70").
            # We default to qty=0 to avoid false discrepancies and surface
            # the suspicious item to the user.
            suspicious_reason = None
            full_match_end_of_name_idx = m.start(2)
            char_before_qty = ln[full_match_end_of_name_idx - 1] if full_match_end_of_name_idx > 0 else ""
            glued_to_letter = char_before_qty.isalpha()

            if glued_to_letter and qty >= 30:
                suspicious_reason = (
                    f"name '{name_raw}' has no space before quantity '{qty_str}' "
                    f"and qty is implausibly large — pypdf may have lost a character "
                    f"(parsed qty defaulted to 0)"
                )

            entry = {
                "name": name_raw,
                "qty": 0 if suspicious_reason else qty,
                "category": current_category,
            }
            if suspicious_reason:
                entry["suspicious"] = suspicious_reason
                entry["raw_qty"] = qty
            items.append(entry)

    return items, header_text


def format_expected_for_prompt(items):
    if not items:
        return "(no expected items extracted)"
    lines = []
    current_cat = None
    for it in items:
        if it["category"] != current_cat:
            current_cat = it["category"]
            lines.append("")
            lines.append(f"## {current_cat}")
        lines.append(f"- {it['name']}: {it['qty']}")
    return "\n".join(lines).strip()


# ============================================================================
# Surveyor + Crop
# ============================================================================
def locate_target_column(client, expected_path: str, actual_path: str):
    doc = fitz.open(actual_path)
    if len(doc) == 0:
        doc.close()
        return {}

    page = doc[0]
    w, h = page.rect.width, page.rect.height
    for j in range(0, 102, 2):
        x = w * (j / 100.0)
        page.draw_line(fitz.Point(x, 0), fitz.Point(x, h), color=(1, 0, 0), width=1)
    img_bytes = page.get_pixmap(dpi=150).tobytes("png")
    doc.close()

    prompt = """You are an expert inventory surveyor analyzing a multi-day inventory pack sheet.

The sheet has TWO header rows at the top:
- Row 1 (printed): "PACK:" then "DATE  DATE  DATE  DATE  DATE  DATE  DATE  DATE" (or similar)
- Row 2 (handwritten): the actual day-of-month numbers, e.g. "7  8  9  10  11  12  13  14"

You will receive an image with vertical RED LINES drawn every 2 units from 0 to 100. Each red line is implicitly numbered by its position.

Your job:
1. Read the target date from the EXPECTED PDF — extract the day-of-month (e.g. for "MAY 7 2026" the day is "7").
2. Look at ROW 2 of the actual sheet's header (the HANDWRITTEN day numbers, NOT the printed "DATE" labels). Find the cell containing the target day.
3. Determine the Red Line number (0-100) running through the CENTER of that cell's column.
4. Determine the Red Line number where the Item Names column (the leftmost column with item names like "Ice Breath", "Pink Petrol") ENDS — i.e. where the data columns begin.

CRITICAL POSITIONING NOTES:
- The Item Names column is wide (typically 30-40% of page width). The names_end_line is usually somewhere between 28 and 42.
- The data columns split the remaining width. Column #1 (leftmost data column) center is typically around line 35-42. The rightmost column center is typically around line 90-96. Do not assume the answer is in the right half of the page — today's date may be in the FIRST data column.
- The date numbers are HANDWRITTEN. Ignore any printed "DATE" labels above them; they are placeholders.
- If the target day appears in column #1 (leftmost), expect target_col_center_line to be in the 33-44 range. If it's the last column, expect 88-96.

If you cannot read the target day on the sheet, set "found": false and explain in "note".

Respond ONLY in JSON, no prose, no markdown:
{
  "target_date": "YYYY-MM-DD",
  "expected_header": "string (the handwritten day number you found, e.g. '7')",
  "found": true,
  "names_end_line": <integer 0-100>,
  "target_col_center_line": <integer 0-100>,
  "note": "string (any caveats; explanation if found=false)"
}"""

    with open(expected_path, "rb") as f:
        exp_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=exp_bytes, mime_type="application/pdf"),
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        prompt,
    ]
    response = client.models.generate_content(model="gemini-2.5-pro", contents=contents)
    return safe_parse_json(response.text)


def apply_precision_crop(input_pdf_path: str, output_pdf_path: str, col_info: dict):
    names_end = col_info.get("names_end_line", 30) / 100.0
    center = col_info.get("target_col_center_line", 38) / 100.0
    target_start = max(names_end, center - CROP_HALF_WIDTH)
    target_end = min(1.0, center + CROP_HALF_WIDTH)

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


def render_pdf_first_page_png(pdf_path: str) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=120)
    png = pix.tobytes("png")
    doc.close()
    return png


# ============================================================================
# Audit prompt
# ============================================================================
def build_audit_prompt(expected_text: str, target_date: str) -> str:
    return f"""You are an inventory auditor. Today's target date is {target_date}.

EXPECTED INVENTORY (from the typed master list — these are the source-of-truth quantities):
{expected_text}

ACTUAL SHEET: attached as a cropped image showing the item-name column on the left and ONE single data column on the right. The right column holds the handwritten/printed actual quantities for today.

For each item in the EXPECTED list above, find the matching row on the cropped image and read the value in the right-hand data column.

CRITICAL RULES:
- "expected" is the quantity from the EXPECTED list above. NEVER change this.
- "actual" is the quantity from the CROPPED IMAGE'S right-hand column. NEVER read this from the expected list.
- A blank, empty, missing, or unreadable cell on the cropped image means actual = 0.
- ONLY output items where expected != actual. Skip items that match.
- Match item names tolerantly (minor spelling and punctuation differences are fine).

Respond ONLY with valid JSON, no markdown, no prose:
{{"items": [{{"item": "string (name as in the expected list)", "expected": <integer>, "actual": <integer>}}]}}"""


# ============================================================================
# Per-model runners
# ============================================================================
def run_gemini(expected_text, target_date, cropped_path):
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_audit_prompt(expected_text, target_date)
    with open(cropped_path, "rb") as f:
        cropped_bytes = f.read()
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(data=cropped_bytes, mime_type="application/pdf"),
            prompt,
        ],
    )
    return safe_parse_json(response.text).get("items", [])


def run_openai(expected_text, target_date, cropped_path):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_audit_prompt(expected_text, target_date)
    crop_b64 = pdf_to_base64_images(cropped_path)

    content = [{"type": "text", "text": prompt}]
    for b64 in crop_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    return safe_parse_json(response.choices[0].message.content).get("items", [])


def run_claude(expected_text, target_date, cropped_path):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_audit_prompt(expected_text, target_date)
    crop_b64 = pdf_to_base64_images(cropped_path)

    content = [{"type": "text", "text": prompt}]
    for b64 in crop_b64:
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


# ============================================================================
# Ensemble
# ============================================================================
def ensemble_audit(expected_text, target_date, cropped_path, progress_callback=None):
    runners = {
        "gemini": run_gemini,
        "openai": run_openai,
        "claude": run_claude,
    }

    model_results = {}
    model_status = {}
    start_times = {}

    def fire(name, status, elapsed=None):
        if progress_callback is not None:
            try:
                progress_callback(name, status, elapsed)
            except Exception:
                pass

    def make_wrapped(name, fn):
        def wrapped(*args, **kwargs):
            start_times[name] = time.time()
            fire(name, "started", 0.0)
            return fn(*args, **kwargs)
        return wrapped

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(make_wrapped(name, fn), expected_text, target_date, cropped_path): name
            for name, fn in runners.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            elapsed = time.time() - start_times.get(name, time.time())
            try:
                items = future.result(timeout=MODEL_TIMEOUT_SECONDS)
                model_results[name] = items if isinstance(items, list) else []
                model_status[name] = "ok"
                fire(name, "ok", elapsed)
            except concurrent.futures.TimeoutError:
                msg = f"timeout after {MODEL_TIMEOUT_SECONDS}s"
                model_status[name] = msg
                fire(name, msg, elapsed)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                model_status[name] = msg
                fire(name, msg, elapsed)

    models_responded = len(model_results)

    if models_responded >= 2:
        consensus_threshold = 2
    elif models_responded == 1:
        consensus_threshold = 1
    else:
        consensus_threshold = 0

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
            if exp == act:
                continue
            signature = (norm_name, exp, act)
            vote_tally[signature] += 1
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

    verified.sort(key=lambda d: d["item"].lower())

    return {
        "discrepancies": verified,
        "model_status": model_status,
        "models_responded": models_responded,
        "consensus_threshold": consensus_threshold,
    }


# ============================================================================
# Streamlit UI
# ============================================================================
st.set_page_config(page_title="Inventory Agent Pro", page_icon="🕵️‍♂️", layout="centered")
st.title("🕵️‍♂️ Inventory Agent Pro")
st.markdown("Powered by Multi-Model Consensus (Gemini + GPT-4o + Claude)")

if not GEMINI_API_KEY or not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    st.error("⚠️ Missing API Keys! Please configure Streamlit Secrets.")

if "stage" not in st.session_state:
    st.session_state.stage = "IDLE"
for k in ("exp_path", "act_path", "crop_path", "col_info", "expected_items",
          "expected_text", "target_date", "preview_png", "audit_result",
          "surveyor_warning"):
    if k not in st.session_state:
        st.session_state[k] = None


def reset_session():
    for k in ("exp_path", "act_path", "crop_path"):
        path = st.session_state.get(k)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    for k in ("exp_path", "act_path", "crop_path", "col_info", "expected_items",
              "expected_text", "target_date", "preview_png", "audit_result",
              "surveyor_warning"):
        st.session_state[k] = None
    st.session_state.stage = "IDLE"


# ----- Stage 1: Upload + prepare -----
if st.session_state.stage == "IDLE":
    col1, col2 = st.columns(2)
    with col1:
        expected_file = st.file_uploader("1. Expected PDF", type=["pdf"])
    with col2:
        actual_file = st.file_uploader("2. Actual PDF", type=["pdf"])

    if st.button("Prepare Audit", type="primary"):
        if not expected_file or not actual_file:
            st.error("Please upload both PDFs!")
        else:
            try:
                with st.status("Preparing audit...", expanded=True) as status:
                    st.write("📥 Saving uploaded files...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as exp_tmp:
                        exp_tmp.write(expected_file.getvalue())
                        st.session_state.exp_path = exp_tmp.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as act_tmp:
                        act_tmp.write(actual_file.getvalue())
                        st.session_state.act_path = act_tmp.name

                    st.write("📄 Extracting items from expected PDF (no AI required)...")
                    items, _header = extract_expected_items(st.session_state.exp_path)
                    if not items:
                        status.update(label="Failed", state="error", expanded=True)
                        st.error("Could not extract any items from the expected PDF. "
                                 "Check that it's the typed master list, not a scan.")
                        reset_session()
                        st.stop()
                    st.session_state.expected_items = items
                    st.session_state.expected_text = format_expected_for_prompt(items)
                    st.write(f"✅ Extracted {len(items)} items across "
                             f"{len({it['category'] for it in items})} categories.")

                    # Surface any suspicious extractions (likely pypdf parse glitches)
                    suspicious = [it for it in items if it.get("suspicious")]
                    if suspicious:
                        st.warning(
                            f"⚠️ {len(suspicious)} item(s) had suspicious quantities and "
                            f"were defaulted to 0 to avoid false discrepancies:"
                        )
                        for it in suspicious:
                            st.write(
                                f"  • **{it['name']}** — parsed qty `{it['raw_qty']}` "
                                f"({it['suspicious']})"
                            )

                    st.write("🔍 Locating today's column on the actual sheet (Gemini surveyor)...")
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    col_info = locate_target_column(
                        client, st.session_state.exp_path, st.session_state.act_path
                    )
                    if not col_info or "target_col_center_line" not in col_info:
                        status.update(label="Failed", state="error", expanded=True)
                        st.error("Could not locate target column on the actual sheet. "
                                 "Verify the date matches the expected PDF and try again.")
                        reset_session()
                        st.stop()

                    if col_info.get("found") is False:
                        status.update(label="Failed", state="error", expanded=True)
                        st.error(f"Surveyor could not find the target date on the actual sheet: "
                                 f"{col_info.get('note', 'no explanation')}. "
                                 f"Verify the day numbers are clearly written in the date row.")
                        reset_session()
                        st.stop()

                    st.session_state.col_info = col_info
                    st.session_state.target_date = col_info.get("target_date", "Unknown Date")

                    surveyor_warning = None
                    target_dt = col_info.get("target_date", "")
                    if target_dt:
                        try:
                            day_from_date = int(target_dt.split("-")[-1])
                            day_from_header = int(str(col_info.get("expected_header", "")).strip())
                            if day_from_date != day_from_header:
                                surveyor_warning = (
                                    f"⚠️ Target date is day {day_from_date} of the month, "
                                    f"but the surveyor identified the column with header '{day_from_header}'. "
                                    f"Verify the crop preview carefully before running the audit."
                                )
                        except (ValueError, AttributeError, IndexError):
                            pass

                    center = col_info.get("target_col_center_line", 0)
                    names_end = col_info.get("names_end_line", 0)
                    if center < names_end + 2 or center > 98:
                        coord_warn = (
                            f"⚠️ Surveyor returned an out-of-range coordinate "
                            f"(center={center}, names_end={names_end}). "
                            f"The crop is likely wrong — abort and re-scan."
                        )
                        surveyor_warning = (surveyor_warning + "\n" + coord_warn) if surveyor_warning else coord_warn

                    st.session_state.surveyor_warning = surveyor_warning
                    st.write(f"✅ Target date: **{st.session_state.target_date}** · "
                             f"column center at line {center} · "
                             f"header reads '{col_info.get('expected_header')}'.")

                    st.write("✂️ Applying precision crop (PyMuPDF)...")
                    crop_path = st.session_state.act_path.replace(".pdf", "_cropped.pdf")
                    apply_precision_crop(
                        st.session_state.act_path, crop_path, col_info
                    )
                    st.session_state.crop_path = crop_path

                    st.write("🖼️ Rendering preview...")
                    st.session_state.preview_png = render_pdf_first_page_png(crop_path)

                    status.update(label="Ready for verification", state="complete", expanded=False)

                st.session_state.stage = "PREVIEW"
                st.rerun()

            except Exception as e:
                st.error(f"Preparation failed: {type(e).__name__}: {e}")
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc(), language="text")
                reset_session()


# ----- Stage 2: Crop preview verification -----
elif st.session_state.stage == "PREVIEW":
    st.subheader(f"Preview — Target Date: {st.session_state.target_date}")
    st.caption(
        f"Extracted {len(st.session_state.expected_items)} items from the expected PDF. "
        f"Cropped column #{st.session_state.col_info.get('target_col_center_line')} "
        f"(header reads '{st.session_state.col_info.get('expected_header')}')."
    )

    if st.session_state.surveyor_warning:
        st.warning(st.session_state.surveyor_warning)

    st.markdown(
        "**Verify the crop below.** The right side should show only the target date's "
        "column and have visible numbers in it. If it looks misaligned (clipped numbers, "
        "wrong day, or empty when the actual sheet has values), abort and re-run."
    )
    st.image(st.session_state.preview_png,
             caption="Page 1 of cropped actual sheet",
             use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Crop looks correct — Run audit",
                     type="primary", use_container_width=True):
            st.session_state.stage = "AUDITING"
            st.rerun()
    with col_b:
        if st.button("❌ Crop is wrong — Abort", use_container_width=True):
            reset_session()
            st.rerun()


# ----- Stage 3: Run the ensemble audit (with live progress) -----
elif st.session_state.stage == "AUDITING":
    st.subheader(f"Auditing — {st.session_state.target_date}")

    overall_label = st.empty()
    overall_bar = st.progress(0)
    st.write("")
    st.caption("Models")

    model_names = ("gemini", "openai", "claude")
    model_display = {"gemini": "Gemini 2.5 Pro", "openai": "GPT-4o", "claude": "Claude Opus 4.7"}
    model_placeholders = {name: st.empty() for name in model_names}

    for name in model_names:
        model_placeholders[name].markdown(f"⚪ **{model_display[name]}** — waiting")
    overall_label.markdown("**Overall progress** — 0 of 3 models done")

    progress_state = {"completed": 0, "started": set()}

    def progress_callback(name, status, elapsed):
        if status == "started":
            progress_state["started"].add(name)
            model_placeholders[name].markdown(f"🟡 **{model_display[name]}** — running...")
        elif status == "ok":
            progress_state["completed"] += 1
            done = progress_state["completed"]
            elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
            model_placeholders[name].markdown(
                f"✅ **{model_display[name]}** — done ({elapsed_str})"
            )
            overall_label.markdown(f"**Overall progress** — {done} of 3 models done")
            overall_bar.progress(done / 3)
        else:
            progress_state["completed"] += 1
            done = progress_state["completed"]
            elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
            short = status if len(status) <= 70 else status[:67] + "..."
            model_placeholders[name].markdown(
                f"❌ **{model_display[name]}** — failed ({elapsed_str}): `{short}`"
            )
            overall_label.markdown(f"**Overall progress** — {done} of 3 models done")
            overall_bar.progress(done / 3)

    try:
        result = ensemble_audit(
            st.session_state.expected_text,
            st.session_state.target_date,
            st.session_state.crop_path,
            progress_callback=progress_callback,
        )
        overall_bar.progress(1.0)
        overall_label.markdown("**Overall progress** — tallying votes...")

        st.session_state.audit_result = result
        st.session_state.stage = "DONE"
        st.rerun()
    except Exception as e:
        st.error(f"Audit failed: {type(e).__name__}: {e}")
        with st.expander("Stack trace"):
            st.code(traceback.format_exc(), language="text")
        if st.button("Start over"):
            reset_session()
            st.rerun()


# ----- Stage 4: Results -----
elif st.session_state.stage == "DONE":
    result = st.session_state.audit_result
    target_date = st.session_state.target_date

    status_lines = []
    for name, status in result["model_status"].items():
        icon = "✅" if status == "ok" else "❌"
        status_lines.append(f"{icon} {name}: {status}")
    st.caption("Model status — " + "  ·  ".join(status_lines))

    responded = result["models_responded"]
    threshold = result["consensus_threshold"]

    if responded == 0:
        st.error("All three models failed. See model status above. "
                 "Try again or check API keys / quotas.")
    elif responded == 1:
        st.warning(
            "Only 1 of 3 models responded. Output is from that one model alone — "
            "no cross-validation was performed. Treat results with extra caution."
        )
    elif responded == 2:
        st.warning("Only 2 of 3 models responded. Showing items both remaining models agreed on.")

    discrepancies = result["discrepancies"]
    if not discrepancies and responded > 0:
        st.success("Consensus reached.")
        st.balloons()
        st.subheader("✅ All items match.")
    elif discrepancies:
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

    if st.button("Start over"):
        reset_session()
        st.rerun()
