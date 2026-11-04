# VERIFAI – AI Confidence / Verification System

## Project Overview

VERIFAI is an AI-assisted confidence and verification system designed to examine AI-generated answers against user-provided reference evidence.

It combines semantic similarity matching and Natural Language Inference (NLI) to classify extracted claims as Supported, Contradicted, or Insufficient Evidence.

## Key Features

- AI-powered answer generation using Groq.
- Automatic claim extraction from generated answers.
- Semantic similarity matching using Sentence Transformers.
- Evidence evaluation using Natural Language Inference.
- Claim-level verification reports.
- Downloadable Markdown reports.

## Technologies Used

- Python
- Gradio
- Groq API
- Sentence Transformers
- Scikit-learn
- Hugging Face Transformers
- PyTorch

## System Architecture

1. User enters a question or an existing AI-generated answer.
2. The AI assistant generates an answer using Groq, when requested.
3. The system extracts individual claims.
4. Sentence Transformers generates semantic embeddings.
5. Cosine similarity identifies the closest reference evidence.
6. Claims meeting the similarity threshold are evaluated using NLI.
7. The system generates a verification report.

## Models Used

- AI Assistant: openai/gpt-oss-20b (via Groq)
- Semantic Similarity: sentence-transformers/all-MiniLM-L6-v2
- NLI: cross-encoder/nli-deberta-v3-small

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kshitt/VERIFAI-AI-Confidence-Verification-System.git
cd VERIFAI-AI-Confidence-Verification-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Groq API key

Set the `GROQ_API_KEY` environment variable on your system.

Never upload API keys or secrets to GitHub.

### 4. Run the application

```bash
python app.py
```

### 5. Open the application

http://127.0.0.1:7861

## Project Structure

```text
VERIFAI/
├── app.py
├── integrated_verifier.py
├── verification_engine.py
├── nli_verifier.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Verification depends on the quality and completeness of manually supplied reference evidence.
- Semantic similarity measures relatedness, not factual correctness.
- NLI scores are model prediction scores, not calibrated probabilities of truth.
- The system uses a threshold-based verification pipeline.
- Claim extraction may not correctly handle every complex sentence structure.
- The system does not automatically retrieve external reference sources.

## Disclaimer

VERIFAI provides AI-assisted verification estimates. Its predictions are not guarantees of factual correctness.

