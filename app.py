
import os
import re
import time
import tempfile

import gradio as gr
from groq import Groq

from integrated_verifier import IntegratedVerifier


# --------------------------------------------------
# 1. INITIALIZE THE APPLICATION
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Set it in Windows Environment Variables, "
        "then restart VS Code."
    )

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=30.0,
    max_retries=0
)

GROQ_MODEL = "openai/gpt-oss-20b"

verifier = IntegratedVerifier(
    similarity_threshold=0.70
)


# --------------------------------------------------
# 2. CLEAN GENERATED TEXT
# --------------------------------------------------

def clean_answer(text):
    """Remove common Markdown formatting before verification."""

    if not text:
        return ""

    cleaned = []

    for line in text.splitlines():

        line = line.strip()

        if not line or re.fullmatch(r"[-*_]{3,}", line):
            continue

        if line.startswith("#"):
            continue

        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)

        line = line.replace("**", "").replace("__", "")
        line = line.replace("*", "").replace("_", "")

        cleaned.append(line)

    return " ".join(cleaned)


def parse_reference_facts(text):
    """Split line-separated or paragraph-style evidence into individual facts."""
    if not text or not text.strip():
        return []
    facts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*+•]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        parts = re.findall(r"[^.!?]+[.!?]?", line)
        facts.extend(part.strip() for part in parts if part.strip())
        
    return facts


# --------------------------------------------------
# 3. FORMAT THE VERIFICATION REPORT
# --------------------------------------------------

def create_report(results, timings=None):

    if not results:
        return (
            "## Verification Report\n\n"
            "No claims were available to verify. "
            "Please enter an answer containing factual sentences."
        )

    total = len(results)

    supported = sum(
        item["nli_status"] == "SUPPORTED"
        for item in results
    )

    contradicted = sum(
        item["nli_status"] == "CONTRADICTED"
        for item in results
    )

    insufficient = sum(
        item["nli_status"] == "INSUFFICIENT EVIDENCE"
        for item in results
    )

    def percentage(count):
        return (count / total) * 100 if total else 0

    def distribution_bar(count):
        filled = round((count / total) * 20)
        return "█" * filled + "░" * (20 - filled)

    report = [
        "# AI Confidence / Verification Report",
        "",
        "## Verification Overview",
        "",
        f"**Total claims evaluated:** {total}",
        "",
        "| Classification | Count | Distribution |",
        "|---|---:|---:|",
        f"| SUPPORTED | {supported} | {percentage(supported):.1f}% |",
        f"| CONTRADICTED | {contradicted} | {percentage(contradicted):.1f}% |",
        f"| INSUFFICIENT EVIDENCE | {insufficient} | {percentage(insufficient):.1f}% |",
        "",
        "## Claim Distribution",
        "",
        f"**Supported:** {distribution_bar(supported)} {percentage(supported):.1f}%",
        "",
        f"**Contradicted:** {distribution_bar(contradicted)} {percentage(contradicted):.1f}%",
        "",
        f"**Insufficient evidence:** {distribution_bar(insufficient)} {percentage(insufficient):.1f}%",
        "",
        "---",
        "",
        "## Individual Claim Analysis"
    ]

    for index, result in enumerate(results, start=1):

        similarity = result["similarity"] * 100
        status = result["nli_status"]
        score = result["nli_score"]

        score_text = (
            f"{score:.2%}"
            if score is not None
            else "Not calculated"
        )

        report.append(
            f"""
### Claim {index}

**Generated claim:** {result["claim"]}

**Reference evidence:** {result["evidence"]}

**Semantic similarity:** {similarity:.2f}%

**Verification status:** {status}

**NLI prediction score:** {score_text}
"""
        )

    report.append(
        f"""
---

## Verification Summary

- **Total claims:** {total}
- **Supported:** {supported}
- **Contradicted:** {contradicted}
- **Insufficient evidence:** {insufficient}
"""
    )

    if timings:

        report.append(
            f"""
## Performance Analysis

- **Groq generation:** {timings.get("generation", 0):.2f} seconds
- **Verification:** {timings.get("verification", 0):.2f} seconds
- **Total processing:** {timings.get("total", 0):.2f} seconds
"""
        )

    report.append(
        """
---

## Important Notes

- Semantic similarity measures semantic relatedness, not factual truth.
- NLI prediction scores are model confidence scores, not calibrated probabilities of factual correctness.
- Verification results depend on the quality and completeness of the supplied reference evidence.
- INSUFFICIENT EVIDENCE means the available evidence does not adequately establish the claim; it does not necessarily mean the claim is false.
"""
    )

    return "\n".join(report)


# --------------------------------------------------
# 4. SAVE THE VERIFICATION REPORT
# --------------------------------------------------

def save_report_file(report):
    """Save the report as a downloadable Markdown file."""

    if not report:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        prefix="AI_Verification_Report_",
        delete=False
    ) as file:

        file.write(report)

        return file.name


# --------------------------------------------------
# 5. VERIFY AN EXISTING ANSWER
# --------------------------------------------------

def verify_existing_answer(answer, facts):

    if not answer or not answer.strip():
        return "Please enter an AI-generated answer.", None

    if not facts or not facts.strip():
        return "Please enter reference facts, one fact per line.", None

    reference_facts = parse_reference_facts(facts)

    cleaned_answer = clean_answer(answer)

    if not cleaned_answer:
        return "No factual text was found in the supplied answer.", None

    start = time.perf_counter()

    try:

        results = verifier.verify_answer(
            generated_text=cleaned_answer,
            reference_facts=reference_facts
        )

        elapsed = time.perf_counter() - start

        report = create_report(
            results,
            timings={
                "verification": elapsed,
                "total": elapsed
            }
        )

        report_file = save_report_file(report)

        return report, report_file

    except Exception as error:

        print(f"Verification error: {error}")

        return f"Verification failed: {error}", None


# --------------------------------------------------
# 6. GENERATE AN ANSWER USING GROQ
# --------------------------------------------------

def generate_answer(question):
    """Generate an answer using Groq and log safe response diagnostics."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question in 3 to 5 concise sentences. "
                    "Use plain text. Provide a direct, useful answer. "
                    "Do not invent facts when uncertain."
                )
            },
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_completion_tokens=500,
        reasoning_effort="low"
    )

    choice = response.choices[0]
    message = choice.message
    answer = message.content

    print(
        "Groq response diagnostics:",
        {
            "model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "content_present": bool(answer and answer.strip()),
            "usage": str(getattr(response, "usage", None))
        }
    )

    if not answer or not answer.strip():
        finish_reason = getattr(choice, "finish_reason", "unknown")
        raise RuntimeError(
            "Groq returned no answer text "
            f"(finish_reason={finish_reason}). "
            "Check the terminal diagnostics; then retry once."
        )
    return answer.strip()


# --------------------------------------------------
# 7. GENERATE AND VERIFY AN ANSWER
# --------------------------------------------------

def ask_ai_assistant(question, facts):

    if not question or not question.strip():
        return "", "Please enter a question.", None

    total_start = time.perf_counter()

    generation_start = time.perf_counter()

    # Stage 1: AI answer generation.
    try:

        answer = generate_answer(question.strip())

        generation_time = (
            time.perf_counter() - generation_start
        )

        print(
            f"DEBUG: Groq generation took "
            f"{generation_time:.4f} seconds"
        )

    except Exception as error:

        generation_time = (
            time.perf_counter() - generation_start
        )

        print(f"Groq API error: {error}")

        return "", (
            "## AI Assistant Request Failed\n\n"
            f"**Time spent waiting:** {generation_time:.2f} seconds\n\n"
            f"**Error:** {error}\n\n"
            "Check your internet connection, Groq API key, "
            "model availability, and API usage limits."
        ), None

    # If no reference facts are supplied, skip verification.
    if not facts or not facts.strip():

        return answer, (
            "## Verification Not Performed\n\n"
            "The AI assistant generated an answer, but reference "
            "facts were not provided.\n\n"
            f"**Groq generation time:** {generation_time:.2f} seconds\n\n"
            "Enter reference facts and click **Verify AI Answer**."
        ), None

    reference_facts = parse_reference_facts(facts)

    # Stage 2: Local verification.
    verification_start = time.perf_counter()

    try:

        cleaned_answer = clean_answer(answer)

        results = verifier.verify_answer(
            generated_text=cleaned_answer,
            reference_facts=reference_facts
        )

        verification_time = (
            time.perf_counter() - verification_start
        )

        total_time = time.perf_counter() - total_start

        timings = {
            "generation": generation_time,
            "verification": verification_time,
            "total": total_time
        }

        report = create_report(results, timings)

        report_file = save_report_file(report)

        return answer, report, report_file

    except Exception as error:

        verification_time = (
            time.perf_counter() - verification_start
        )

        print(f"Verification error: {error}")

        return answer, (
            "## Verification Failed\n\n"
            f"The AI assistant generated an answer in "
            f"{generation_time:.2f} seconds, but verification "
            f"failed after {verification_time:.2f} seconds.\n\n"
            f"**Error:** {error}"
        ), None


# --------------------------------------------------
# 8. BUILD THE VERIFAI WEBSITE + PROJECT WORKSPACE
# --------------------------------------------------

VERIFAI_CSS = """
:root {
    --vf-bg: #080d18;
    --vf-panel: #111a2b;
    --vf-panel-soft: #162238;
    --vf-border: rgba(148, 163, 184, 0.16);
    --vf-text: #edf4ff;
    --vf-muted: #9aabc4;
    --vf-blue: #60a5fa;
    --vf-cyan: #67e8f9;
}

/* Overall page */
.gradio-container {
    max-width: 1220px !important;
    background:
        radial-gradient(ellipse at 12% 0%, rgba(37, 99, 235, 0.16), transparent 34%),
        radial-gradient(ellipse at 88% 8%, rgba(8, 145, 178, 0.10), transparent 30%),
        var(--vf-bg) !important;
    color: var(--vf-text) !important;
}

body, .gradio-container, .main {
    background-color: var(--vf-bg) !important;
}

.gradio-container .prose,
.gradio-container label,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    color: var(--vf-text);
}

.gradio-container .prose p,
.gradio-container .prose li {
    color: var(--vf-muted);
}

/* Landing page */
#vf-landing {
    padding: 24px 0 12px;
}

#vf-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 13px 18px;
    margin-bottom: 34px;
    border: 1px solid var(--vf-border);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.78);
    backdrop-filter: blur(12px);
}

.vf-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    color: #f8fbff;
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.13em;
}

.vf-brand-mark {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border-radius: 11px;
    color: #07111f;
    background: linear-gradient(135deg, #67e8f9, #60a5fa);
    font-size: 19px;
}

.vf-nav-links {
    display: flex;
    align-items: center;
    gap: 22px;
}

.vf-nav-links a {
    color: #b7c5d9 !important;
    text-decoration: none !important;
    font-size: 13px;
    font-weight: 600;
}

.vf-nav-links a:hover {
    color: #ffffff !important;
}

.vf-nav-cta {
    padding: 10px 15px;
    border-radius: 10px;
    color: #07111f !important;
    background: linear-gradient(110deg, #67e8f9, #60a5fa);
    font-weight: 800 !important;
}

#vf-hero {
    position: relative;
    overflow: hidden;
    padding: 56px 48px;
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 26px;
    background:
        radial-gradient(circle at 82% 28%, rgba(34, 211, 238, 0.16), transparent 28%),
        linear-gradient(125deg, #111d33 0%, #10192b 52%, #0d1c30 100%);
}

.vf-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 11px;
    border: 1px solid rgba(103, 232, 249, 0.24);
    border-radius: 999px;
    color: #a5f3fc;
    background: rgba(8, 145, 178, 0.10);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.vf-hero-title {
    max-width: 760px;
    margin: 22px 0 15px;
    color: #f8fbff;
    font-size: clamp(38px, 5.2vw, 64px);
    line-height: 1.06;
    font-weight: 850;
    letter-spacing: -0.045em;
}

.vf-gradient-text {
    color: #7dd3fc;
    background: linear-gradient(90deg, #67e8f9, #60a5fa 60%, #a5b4fc);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.vf-hero-copy {
    max-width: 680px;
    color: #b8c7dc;
    font-size: 16px;
    line-height: 1.8;
}

.vf-hero-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 13px;
    margin-top: 28px;
}

.vf-primary-link, .vf-secondary-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 43px;
    padding: 0 17px;
    border-radius: 11px;
    text-decoration: none !important;
    font-size: 13px;
    font-weight: 800;
}

.vf-primary-link {
    color: #07111f !important;
    background: linear-gradient(110deg, #67e8f9, #60a5fa);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.20);
}

.vf-secondary-link {
    color: #dce9fb !important;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: rgba(15, 23, 42, 0.45);
}

.vf-hero-note {
    margin-top: 25px;
    color: #8194af;
    font-size: 11px;
    letter-spacing: 0.04em;
}

.vf-section {
    padding: 42px 2px 8px;
}

.vf-section-kicker {
    color: #67e8f9;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.vf-section-title {
    margin: 9px 0 8px;
    color: #f3f7ff;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.025em;
}

.vf-section-copy {
    max-width: 720px;
    color: #9aabc4;
    font-size: 14px;
    line-height: 1.75;
}

.vf-feature-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-top: 22px;
}

.vf-feature-card {
    min-height: 155px;
    padding: 20px;
    border: 1px solid var(--vf-border);
    border-radius: 17px;
    background: linear-gradient(155deg, rgba(22, 34, 56, 0.94), rgba(15, 24, 40, 0.94));
}

.vf-feature-icon {
    display: grid;
    place-items: center;
    width: 39px;
    height: 39px;
    margin-bottom: 16px;
    border: 1px solid rgba(103, 232, 249, 0.18);
    border-radius: 12px;
    color: #7dd3fc;
    background: rgba(8, 145, 178, 0.11);
    font-size: 18px;
}

.vf-feature-card h3 {
    margin: 0 0 7px;
    color: #edf4ff;
    font-size: 14px;
    font-weight: 800;
}

.vf-feature-card p {
    margin: 0;
    color: #94a7c1;
    font-size: 12px;
    line-height: 1.65;
}

.vf-flow {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    margin-top: 22px;
}

.vf-flow-step {
    position: relative;
    padding: 17px 14px;
    border: 1px solid var(--vf-border);
    border-radius: 14px;
    background: rgba(17, 26, 43, 0.82);
}

.vf-flow-number {
    color: #67e8f9;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.12em;
}

.vf-flow-step strong {
    display: block;
    margin-top: 9px;
    color: #eaf2ff;
    font-size: 12px;
}

.vf-flow-step span {
    display: block;
    margin-top: 5px;
    color: #91a4bf;
    font-size: 11px;
    line-height: 1.5;
}

#vf-workspace {
    scroll-margin-top: 18px;
    margin-top: 45px;
    padding: 28px;
    border: 1px solid rgba(96, 165, 250, 0.22);
    border-radius: 22px;
    background: rgba(12, 20, 34, 0.88);
}

#vf-workspace .tab-nav {
    border-bottom: 1px solid var(--vf-border);
}

#vf-workspace button[role="tab"] {
    color: #9aabc4;
    border-radius: 10px 10px 0 0;
}

#vf-workspace button[role="tab"][aria-selected="true"] {
    color: #eaf5ff;
    border-color: #60a5fa;
    background: rgba(37, 99, 235, 0.13);
}

#vf-workspace .form,
#vf-workspace .panel,
#vf-workspace .block {
    border-color: var(--vf-border) !important;
}

#vf-workspace input,
#vf-workspace textarea {
    border-color: rgba(148, 163, 184, 0.22) !important;
    background: #0b1424 !important;
    color: #edf4ff !important;
}

#vf-workspace input:focus,
#vf-workspace textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.22) !important;
}

#vf-workspace button.primary {
    border: 0 !important;
    color: #07111f !important;
    background: linear-gradient(110deg, #67e8f9, #60a5fa) !important;
    font-weight: 800 !important;
}

#vf-workspace button.secondary {
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    color: #dce9fb !important;
    background: #162238 !important;
}

#vf-workspace .prose table {
    color: #dce9fb;
}

#vf-workspace .prose th {
    color: #f1f6ff;
    background: #17243a;
}

#vf-workspace .prose td {
    color: #c0cee1;
}

#vf-footer {
    margin: 32px 0 12px;
    padding: 22px 4px 10px;
    border-top: 1px solid var(--vf-border);
    color: #8092ac;
    font-size: 11px;
    line-height: 1.8;
}

#vf-footer strong {
    color: #c9d8ec;
}

@media (max-width: 850px) {
    .vf-feature-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .vf-flow {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    #vf-hero {
        padding: 38px 25px;
    }
    #vf-workspace {
        padding: 18px;
    }
}

@media (max-width: 560px) {
    #vf-nav {
        align-items: flex-start;
        flex-direction: column;
    }
    .vf-nav-links {
        flex-wrap: wrap;
        gap: 13px;
    }
    .vf-feature-grid,
    .vf-flow {
        grid-template-columns: 1fr;
    }
    .vf-hero-title {
        font-size: 39px;
    }
}
"""

with gr.Blocks(
    title="VERIFAI | AI Confidence & Verification",
    css=VERIFAI_CSS,
    theme=gr.themes.Base(
        primary_hue="sky",
        secondary_hue="slate",
        neutral_hue="slate"
    )
) as app:

    # ----------------------------------------------
    # LANDING PAGE
    # ----------------------------------------------

    gr.HTML(
        """
        <div id="vf-landing">
          <nav id="vf-nav">
            <div class="vf-brand">
              <span class="vf-brand-mark">✓</span>
              <span>VERIFAI</span>
            </div>
            <div class="vf-nav-links">
              <a href="#vf-about">About</a>
              <a href="#vf-features">Features</a>
              <a href="#vf-process">How it works</a>
              <a class="vf-nav-cta" href="#vf-workspace">Open Workspace&nbsp; ↗</a>
            </div>
          </nav>

          <section id="vf-hero">
            <div class="vf-eyebrow">
              <span>✦</span> AI CONFIDENCE &amp; VERIFICATION SYSTEM
            </div>
            <h1 class="vf-hero-title">
              Don't just trust AI.<br>
              <span class="vf-gradient-text">Verify it.</span>
            </h1>
            <p class="vf-hero-copy">
              VERIFAI helps examine AI-generated answers claim by claim.
              Compare generated content with reference evidence, measure
              semantic similarity, and use Natural Language Inference to
              identify supported, contradicted, and insufficiently supported claims.
            </p>
            <div class="vf-hero-actions">
              <a class="vf-primary-link" href="#vf-workspace">Launch Verification Workspace&nbsp; →</a>
              <a class="vf-secondary-link" href="#vf-process">Explore the workflow</a>
            </div>
            <div class="vf-hero-note">
              GROQ-POWERED AI ASSISTANT &nbsp;·&nbsp; SEMANTIC SIMILARITY
              &nbsp;·&nbsp; NATURAL LANGUAGE INFERENCE
            </div>
          </section>
        </div>
        """
    )

    gr.HTML(
        """
        <section id="vf-about" class="vf-section">
          <div class="vf-section-kicker">ABOUT VERIFAI</div>
          <h2 class="vf-section-title">A verification layer for AI-generated answers</h2>
          <p class="vf-section-copy">
            AI-generated content can sound convincing while containing claims
            that are unsupported by the available evidence. VERIFAI provides
            an interface to examine those claims against reference facts supplied
            by the user. It combines semantic matching with NLI-based classification
            to make the verification process easier to inspect.
          </p>
        </section>

        <section id="vf-features" class="vf-section">
          <div class="vf-section-kicker">CORE CAPABILITIES</div>
          <h2 class="vf-section-title">What you can do with VERIFAI</h2>
          <p class="vf-section-copy">
            Generate an answer or bring an existing one. Then inspect the
            evidence match and claim-level verification results.
          </p>
          <div class="vf-feature-grid">
            <article class="vf-feature-card">
              <div class="vf-feature-icon">✧</div>
              <h3>AI Assistant</h3>
              <p>Generate concise answers using the Groq-powered assistant.</p>
            </article>
            <article class="vf-feature-card">
              <div class="vf-feature-icon">⌕</div>
              <h3>Semantic Matching</h3>
              <p>Compare each extracted claim with supplied reference facts using sentence embeddings.</p>
            </article>
            <article class="vf-feature-card">
              <div class="vf-feature-icon">◎</div>
              <h3>NLI Verification</h3>
              <p>Evaluate evidence relationships and classify claims as supported, contradicted, or insufficient.</p>
            </article>
            <article class="vf-feature-card">
              <div class="vf-feature-icon">⇩</div>
              <h3>Downloadable Reports</h3>
              <p>Review claim-level results, similarity scores, NLI scores, and processing time.</p>
            </article>
          </div>
        </section>

        <section id="vf-process" class="vf-section">
          <div class="vf-section-kicker">VERIFICATION PIPELINE</div>
          <h2 class="vf-section-title">How the system works</h2>
          <p class="vf-section-copy">
            The application processes an answer through a sequence of
            claim extraction, evidence matching, and NLI evaluation.
          </p>
          <div class="vf-flow">
            <div class="vf-flow-step">
              <div class="vf-flow-number">STEP 01</div>
              <strong>Answer input</strong>
              <span>Generate an answer or paste existing AI text.</span>
            </div>
            <div class="vf-flow-step">
              <div class="vf-flow-number">STEP 02</div>
              <strong>Claim extraction</strong>
              <span>Split the response into individual claims.</span>
            </div>
            <div class="vf-flow-step">
              <div class="vf-flow-number">STEP 03</div>
              <strong>Semantic matching</strong>
              <span>Find the closest supplied reference fact.</span>
            </div>
            <div class="vf-flow-step">
              <div class="vf-flow-number">STEP 04</div>
              <strong>NLI evaluation</strong>
              <span>Evaluate evidence when the similarity threshold is met.</span>
            </div>
            <div class="vf-flow-step">
              <div class="vf-flow-number">STEP 05</div>
              <strong>Verification report</strong>
              <span>Display classifications and export a Markdown report.</span>
            </div>
          </div>
        </section>
        """
    )

    # ----------------------------------------------
    # PROJECT WORKSPACE
    # ----------------------------------------------

    with gr.Column(elem_id="vf-workspace"):
        gr.HTML(
            """
            <div style="padding: 2px 0 16px;">
              <div class="vf-section-kicker">INTERACTIVE APPLICATION</div>
              <h2 style="margin:8px 0;color:#f3f7ff;font-size:27px;font-weight:800;">
                Verification Workspace
              </h2>
              <p style="margin:0;color:#9aabc4;font-size:13px;line-height:1.7;">
                Choose an AI assistant workflow or verify an existing answer.
                Add relevant reference evidence to evaluate the claims.
              </p>
            </div>
            """
        )

        # TAB 1: AI ASSISTANT
        with gr.Tab("✧  AI Assistant"):

            gr.Markdown(
                "### Generate an answer and verify it\n"
                "Enter a question and provide reference facts, one fact per line."
            )

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Example: What is machine learning?",
                lines=3
            )

            assistant_facts = gr.Textbox(
                label="Reference Facts",
                placeholder="Paste reference facts as sentences or one fact per line...",
                lines=4
            )

            ask_button = gr.Button(
                "Generate Answer and Verify",
                variant="primary"
            )

            generated_answer = gr.Textbox(
                label="Groq's Answer",
                lines=5,
                interactive=False
            )

            assistant_report = gr.Markdown()

            assistant_download = gr.File(
                label="Download Verification Report",
                interactive=False
            )

            ask_button.click(
                fn=ask_ai_assistant,
                inputs=[
                    question_input,
                    assistant_facts
                ],
                outputs=[
                    generated_answer,
                    assistant_report,
                    assistant_download
                ]
            )

            verify_generated_button = gr.Button(
                "Verify AI Answer",
                variant="secondary"
            )

            verify_generated_button.click(
                fn=verify_existing_answer,
                inputs=[
                    generated_answer,
                    assistant_facts
                ],
                outputs=[
                    assistant_report,
                    assistant_download
                ]
            )

        # TAB 2: VERIFY EXISTING ANSWER
        with gr.Tab("◎  Verify Existing Answer"):

            gr.Markdown(
                "### Verify an existing AI-generated answer\n"
                "Paste the answer you want to inspect and provide supporting reference evidence."
            )

            existing_answer = gr.Textbox(
                label="AI-Generated Answer",
                placeholder="Paste the answer you want to verify...",
                lines=6
            )

            existing_facts = gr.Textbox(
                label="Reference Facts",
                placeholder="Paste reference facts as sentences or one fact per line...",
                lines=4
            )

            verify_button = gr.Button(
                "Verify Answer",
                variant="primary"
            )

            existing_report = gr.Markdown()

            existing_download = gr.File(
                label="Download Verification Report",
                interactive=False
            )

            verify_button.click(
                fn=verify_existing_answer,
                inputs=[
                    existing_answer,
                    existing_facts
                ],
                outputs=[
                    existing_report,
                    existing_download
                ]
            )

    # ----------------------------------------------
    # FOOTER AND DISCLAIMER
    # ----------------------------------------------

    gr.HTML(
        """
        <footer id="vf-footer">
          <strong>VERIFAI</strong> &nbsp;·&nbsp; AI Confidence / Verification System
          <br>
          Built with Python, Gradio, Groq, Sentence Transformers, and Natural Language Inference.
          <br>
          <span>
            Disclaimer: Verification results are model predictions based on the supplied evidence.
            They are not guarantees of factual correctness. Reference facts are manually provided
            in this implementation.
          </span>
        </footer>
        """
    )


# --------------------------------------------------
# 9. LAUNCH THE APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.queue(default_concurrency_limit=2)

    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True
    )
