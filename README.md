# Inventory Agent Pro

A daily inventory discrepancy checker. You upload two PDFs — a typed expected-quantities master list and a scanned handwritten pack sheet — and the app returns the discrepancies between them, formatted for WhatsApp.

Three vision-capable AI models read the handwritten counts in parallel and only findings that at least two of them agree on are reported.

## How it works

```
┌──────────────────────┐    ┌──────────────────────────┐
│  EXPECTED PDF        │    │  ACTUAL PDF              │
│  (typed master list, │    │  (scanned pack sheet,    │
│   single date)       │    │   multi-day, handwritten)│
└──────────┬───────────┘    └──────────┬───────────────┘
           │                            │
           │                            ▼
           │             ┌─────────────────────────────┐
           │             │  Stage 1: Surveyor          │
           │             │  Gemini reads coordinates   │
           │             │  off a numbered red ruler   │
           │             │  drawn on page 1 to find    │
           │             │  today's column.            │
           │             └──────────┬──────────────────┘
           │                        │
           │                        ▼
           │             ┌─────────────────────────────┐
           │             │  Stage 2: Precision Crop    │
           │             │  PyMuPDF reconstructs a     │
           │             │  narrow PDF showing only    │
           │             │  the names column + today's │
           │             │  column. Adjacent days are  │
           │             │  physically removed.        │
           │             └──────────┬──────────────────┘
           │                        │
           ▼                        ▼
   ┌──────────────────────────────────────────┐
   │  Stage 3: Crop Preview Verification      │
   │  Page 1 of the cropped PDF is shown      │
   │  inline. User confirms or aborts before  │
   │  the expensive ensemble runs.            │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  Stage 4: Multi-Model Ensemble Audit     │
   │                                          │
   │  ┌─────────┐ ┌─────────┐ ┌─────────┐     │
   │  │ Gemini  │ │ GPT-4o  │ │ Claude  │     │
   │  │ 2.5 Pro │ │         │ │ Opus 4.7│     │
   │  └────┬────┘ └────┬────┘ └────┬────┘     │
   │       │           │           │          │
   │       └───────────┼───────────┘          │
   │                   ▼                      │
   │       Majority vote (2-of-3)             │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
              Discrepancy report
              (WhatsApp-formatted)
```

## Why an ensemble

Handwriting recognition is the single hardest part of this pipeline. A single model's misread (e.g. "3" read as "8" because the 3 was poorly closed) becomes a wrong inventory report. Running three different models in parallel and only reporting findings that at least two of them agree on filters out individual misreads while keeping consensus findings.

The cost is roughly 3x the API spend of a single-model setup. The accuracy gain is significant on real-world scanned handwriting.

### The math behind it

The reason this works comes from the statistics of variance. **Variance** measures how "spread out" a set of measurements is from their average — when the same quantity is measured multiple times by different methods, low variance across the methods means they all agree (high confidence), and high variance means they disagree (low confidence, likely error).

The formula for sample variance is:

$$S^2 = \frac{\sum_{i=1}^{N}(X_i - \bar{X})^2}{N - 1}$$

Where:

- **$S^2$ (Sample Variance)** — the final number representing the spread of the sample.
- **$\sum$ (Sigma)** — the mathematical symbol for "sum." Add up everything that comes after it.
- **$X_i$** — an individual data point in the set.
- **$\bar{X}$ (X-bar)** — the mean (average) of the sample.
- **$(X_i - \bar{X})^2$** — the core of the formula. Take a data point, subtract the average, square the result. Squaring serves two purposes: it makes all negative differences positive (so they don't cancel out), and it penalizes points that are far from the average more heavily than points that are close.
- **$N$** — the total number of data points.
- **$N - 1$** — the denominator. Subtracting 1 is a statistical adjustment called **Bessel's Correction**. Because we're only looking at a sample (a small piece of a larger population) rather than the full population, the sample's spread will usually be slightly smaller than the real-world spread. Dividing by $N - 1$ instead of $N$ corrects this bias and makes the estimate more accurate.

### How this maps onto handwriting recognition

Each model's reading of a handwritten cell is one $X_i$ — one data point in the sample. For a cell where the digit is clearly written, all three models read the same number, the variance is zero, and we can trust the result. For a cell where the digit is ambiguous (a smudged 3 that could be an 8, or a 1 with no foot that could be a 7), the three models tend to scatter their guesses, the variance jumps, and the disagreement itself becomes a signal that this cell needs attention.

The 2-of-3 consensus rule operationalizes this: any finding where at least two models converge on the same answer has low variance and gets reported. Findings where all three disagree (high variance) get filtered out automatically — they're the ones most likely to be wrong, and silently dropping them prevents false alarms.

This is why three models work better than one: a single model has a fixed error rate on hard cells, and you have no way to know when it's wrong. Three independent models with three different training datasets and three different vision encoders rarely produce the *same* wrong answer on the same cell. So consensus is much more likely on correct readings than on mistaken ones. Variance becomes the silent quality filter.

## Why the precision crop

Earlier versions told the model "use column 8 of 14" and hoped it could count narrow columns across a skewed scan. That failed often — column drift was the most persistent bug.

The current architecture removes the problem at the pixel level: PyMuPDF physically reconstructs a new PDF containing only the item-name column (left) and today's data column (right), with everything else discarded. The model can no longer read the wrong column because the wrong columns aren't in the image.

## Why the crop preview gate

Each ensemble run costs real money (≈$0.30–$1.00 depending on PDF size and ensemble configuration). If the surveyor's coordinate is off by a few percent, the entire crop is misaligned and the audit will be wrong. The preview shows page 1 of the cropped PDF inline before the ensemble runs, so a human can spot a bad crop and abort for the cost of just the surveyor pass.

## Setup

### Local

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables for your API keys (or use a `.streamlit/secrets.toml` file as in deployment below)
4. Run: `streamlit run app.py`

### Streamlit Cloud deployment

1. Push the repo to GitHub
2. Connect the repo on [share.streamlit.io](https://share.streamlit.io)
3. Add three secrets in the Streamlit Cloud dashboard under "Settings → Secrets":

```toml
GEMINI_API_KEY = "your-key-here"
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

4. The app will be available at `https://<your-app-name>.streamlit.app`

API keys never appear in the URL or in users' browsers — they live on the Streamlit Cloud server.

## Usage

1. Open the app
2. Upload the **expected PDF** (typed master list, single date)
3. Upload the **actual PDF** (scanned pack sheet, multi-day handwritten)
4. Click **Prepare Audit**
5. Verify the crop preview — confirm the right column is the target date's column
6. Click **Run audit**
7. Review discrepancies, copy the WhatsApp-formatted report

## Project structure

```
.
├── app.py              # The entire application (one file by design)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Optional: dark theme override
```

## Dependencies

- `streamlit` — web UI framework
- `pymupdf` (imported as `fitz`) — PDF manipulation: red ruler overlay, precision crop, page rendering
- `pypdf` — text extraction from the typed expected PDF (avoids sending it as images to every model)
- `google-genai` — Gemini API client
- `openai` — GPT-4o API client
- `anthropic` — Claude API client

## Cost considerations

Per-audit cost depends on PDF size, but typical numbers:

- Surveyor pass (Gemini): ~$0.01
- Ensemble audit (3 models, ~12 cropped pages each): ~$0.30–$0.80

The expected PDF is text-extracted with pypdf rather than sent as images, which cuts each model call's token usage by ~25k tokens.

## Known limitations

**Handwriting accuracy is bounded by legibility.** If a cell is genuinely unreadable, all three models will either guess wrong or refuse — and consensus on a wrong guess is still a wrong guess. The discrepancy list is best treated as a starting point for manual verification, not a final answer.

**No memory between runs.** Each audit is independent. The app does not learn from past corrections.

**No history.** Once you close the page, the result is gone. If you need an audit trail, copy the WhatsApp message before closing.

**The pypdf extractor occasionally drops the last character of an item name** when the typed master PDF has a quantity glued to the name with no separating space. About 5 entries in a typical master sheet are affected. Verify those manually if RSO/Mary's Confectionary items show unexpected discrepancies.

## Architecture notes

The app is a single `app.py` file by design. Inventory checking is a small, well-scoped problem; splitting it across modules would add ceremony without making anything easier to reason about. The file is organized into clearly-labeled sections: helpers, expected-text extraction, surveyor + crop, audit prompt, per-model runners, ensemble voting, and the Streamlit UI.

The Streamlit UI uses `st.session_state` to track a 4-stage flow (IDLE → PREVIEW → AUDITING → DONE). Streamlit reruns the script on every interaction, so multi-step flows need session state to persist intermediate results across reruns.

Per-model runners raise on error rather than swallowing them. The ensemble layer catches each runner's exception, records the error type and message in `model_status`, and shows that status in the UI after every audit. Silent failures are not allowed — if Gemini is down, you see "❌ gemini: TimeoutError: ..." next to the green checkmarks for the models that did work.

Consensus threshold adapts to how many models actually responded:

- 3 of 3 responded → require 2-of-3 (majority)
- 2 of 3 responded → require both to agree
- 1 of 3 responded → accept its output with a yellow warning that no cross-validation occurred
- 0 of 3 responded → error and stop

This protects against the failure mode where two providers are down and the system silently reports "all items match" because the lone surviving model can't reach a 2-vote threshold by itself.

## License

Private project. Not for redistribution.
