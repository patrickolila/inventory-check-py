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
from pypdf import PdfReader
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


# ============================================================================
# JSON / type helpers
# ============================================================================
def safe_parse_json(text: str) -> dict:
    """Parse JSON from a model response with light cleanup. Returns {} on failure."""
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
# ----------------------------------------------------------------------------
# The expected PDF is typed Excel output, so pypdf can pull names and numbers
# directly. We parse line-by-line: most data lines look like "Item Name 5"
# (name then quantity). Section headers are ALL CAPS phrases without numbers.
# This avoids having to send 8 PDF pages as images to every model — saves
# tokens, and structured text is easier for models to reason over than images
# of a typed list.
# ============================================================================
def extract_expected_items(pdf_path: str):
    """Parse the typed expected PDF.

    The typed master sheet uses a trailing capital 'O' as a section-header
    legend marker. Headers either end in ' O' (with a space) or 'O' attached
    directly after a letter or close-paren (e.g. 'BARO', ')O', '4.00GO').
    Item lines end in a digit (the quantity), with or without a separating
    space (pypdf occasionally drops the space, producing 'Pink Violator20').

    Strategy:
      1. Skip noise lines: TEMP markers, dated freeform notes, *(...) annotations.
      2. If line is a section header (legend marker), update current category.
      3. Otherwise, pull the trailing digit run as the quantity, the rest as name.

    Returns (items, header_text) where:
       items = [{"name": str, "qty": int, "category": str}, ...]
       header_text = first non-empty line, usually contains the date.
    """
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
            # Header marker if previous char is alpha (uppercase, since "o" lowercase
            # would be part of a word) or close-paren. Digits before O would be ambiguous.
            if prev.isalpha() and prev != "o":
                return True
            if prev == ")":
                return True
        # Known bare ALL-CAPS headers that lack the legend marker
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
            qty = int(m.group(2))
            if name_raw:
                items.append({
                    "name": name_raw,
                    "qty": qty,
                    "category": current_category,
                })

    return items, header_text


def format_expected_for_prompt(items):
    """Format the expected items as a clean text block for the prompt."""
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
    """Page 1 only — saves cost vs the old 3-page version."""
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

    prompt = """You are an expert inventory surveyor. Look at the attached ACTUAL inventory sheet image. Read the target date from the EXPECTED PDF. Find that date in the ACTUAL sheet. Tell me the exact Red Line number (0-100) running through the CENTER of that date's column. Tell me the Red Line number where the Item Names column ends. Respond ONLY in JSON: {"target_date": "YYYY-MM-DD", "expected_header": "string", "names_end_line": 22, "target_col_center_line": 74}"""

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
    names_end = col_info.get("names_end_line", 22) / 100.0
    center = col_info.get("target_col_center_line", 74) / 100.0
    # Wider window for narrower columns. With an 8-day sheet, each column is
    # ~8% of page width — a 12% window (6% each side) gives skew tolerance
    # while still excluding adjacent columns. With a 10-day sheet use 0.05.
    # With a 14-day sheet (very narrow columns) use 0.03.
    target_start, target_end = center - 0.06, center + 0.06

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
    """Render page 1 of a PDF as PNG bytes — for showing the crop preview to the user."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=120)
    png = pix.tobytes("png")
    doc.close()
    return png


# ============================================================================
# Audit prompt — text expected + cropped image
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
# Ensemble with status tracking and adaptive consensus
# ============================================================================
# Ensemble with status tracking and adaptive consensus
# ----------------------------------------------------------------------------
# Optional `progress_callback(name, status, elapsed_sec)` is called as each
# model starts and finishes, so the UI can show live progress. status is one
# of "started", "ok", or an error string.
# ============================================================================
def ensemble_audit(expected_text, target_date, cropped_path, progress_callback=None):
    import time

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
                pass  # Never let UI errors break the audit

    # Wrap each runner so we can fire "started" the moment the thread begins
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
# Streamlit UI — multi-step flow with crop preview gate
# ----------------------------------------------------------------------------
# Streamlit reruns the whole script on every interaction, so multi-step flows
# need session_state. The flow is:
#   IDLE       → user uploads files, clicks "Prepare audit"
#   PREVIEW    → surveyor + crop done, preview shown, user confirms or aborts
#   AUDITING   → ensemble runs (during this step the spinner blocks)
#   DONE       → results shown
# ============================================================================
st.set_page_config(page_title="Inventory Agent Pro", page_icon="🕵️‍♂️", layout="centered")
st.title("🕵️‍♂️ Inventory Agent Pro")
st.markdown("Powered by Multi-Model Consensus (Gemini + GPT-4o + Claude)")

if not GEMINI_API_KEY or not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    st.error("⚠️ Missing API Keys! Please configure Streamlit Secrets.")

# Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = "IDLE"
for k in ("exp_path", "act_path", "crop_path", "col_info", "expected_items",
          "expected_text", "target_date", "preview_png", "audit_result"):
    if k not in st.session_state:
        st.session_state[k] = None


def reset_session():
    """Clean up temp files and reset state."""
    for k in ("exp_path", "act_path", "crop_path"):
        path = st.session_state.get(k)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    for k in ("exp_path", "act_path", "crop_path", "col_info", "expected_items",
              "expected_text", "target_date", "preview_png", "audit_result"):
        st.session_state[k] = None
    st.session_state.stage = "IDLE"


# ----- Stage 1: Upload -----
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
                    # Save uploads to disk
                    st.write("📥 Saving uploaded files...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as exp_tmp:
                        exp_tmp.write(expected_file.getvalue())
                        st.session_state.exp_path = exp_tmp.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as act_tmp:
                        act_tmp.write(actual_file.getvalue())
                        st.session_state.act_path = act_tmp.name

                    # Pre-extract expected items from typed PDF as text
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

                    # Surveyor pass — find the target column on actual sheet
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
                    st.session_state.col_info = col_info
                    st.session_state.target_date = col_info.get("target_date", "Unknown Date")
                    st.write(f"✅ Target date: **{st.session_state.target_date}** · "
                             f"column center at line {col_info.get('target_col_center_line')} · "
                             f"header reads '{col_info.get('expected_header')}'.")

                    # Apply the crop
                    st.write("✂️ Applying precision crop (PyMuPDF)...")
                    crop_path = st.session_state.act_path.replace(".pdf", "_cropped.pdf")
                    apply_precision_crop(
                        st.session_state.act_path, crop_path, col_info
                    )
                    st.session_state.crop_path = crop_path

                    # Render page 1 of the crop as a preview image
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

    st.markdown(
        "**Verify the crop below.** The right side should show only the target date's "
        "column. If it looks misaligned (clipped numbers, wrong day), abort and re-run."
    )
    st.image(st.session_state.preview_png, caption="Page 1 of cropped actual sheet", use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Crop looks correct — Run audit", type="primary", use_container_width=True):
            st.session_state.stage = "AUDITING"
            st.rerun()
    with col_b:
        if st.button("❌ Crop is wrong — Abort", use_container_width=True):
            reset_session()
            st.rerun()


# ----- Stage 3: Run the ensemble audit (with live progress) -----
elif st.session_state.stage == "AUDITING":
    st.subheader(f"Auditing — {st.session_state.target_date}")

    # Reserve a placeholder per model and one for the overall progress bar.
    # Streamlit lets us update these from inside the callback while
    # ensemble_audit() is still running on the main thread.
    overall_label = st.empty()
    overall_bar = st.progress(0)
    st.write("")  # spacer
    st.caption("Models")

    model_names = ("gemini", "openai", "claude")
    model_display = {"gemini": "Gemini 2.5 Pro", "openai": "GPT-4o", "claude": "Claude Opus 4.7"}
    model_placeholders = {name: st.empty() for name in model_names}

    # Initial render — all models pending
    for name in model_names:
        model_placeholders[name].markdown(
            f"⚪ **{model_display[name]}** — waiting"
        )
    overall_label.markdown("**Overall progress** — 0 of 3 models done")

    # Track completion state across the parallel runs
    progress_state = {"completed": 0, "started": set()}

    def progress_callback(name, status, elapsed):
        # Called from worker threads as each model starts/finishes.
        if status == "started":
            progress_state["started"].add(name)
            model_placeholders[name].markdown(
                f"🟡 **{model_display[name]}** — running..."
            )
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
            # Error or timeout
            progress_state["completed"] += 1
            done = progress_state["completed"]
            elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
            # Truncate very long error messages
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
        # Final tick — make sure the bar shows 100% even if a model errored
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

    # Per-model status
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
