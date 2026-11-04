
"""
AI Confidence / Verification System
-----------------------------------
Uses Natural Language Inference (NLI) to estimate whether
reference evidence supports, contradicts, or provides
insufficient evidence for a generated claim.

Important:
NLI predictions are estimates, not guarantees of truth.
"""

from transformers import pipeline


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


# --------------------------------------------------
# 2. NLI VERIFICATION CLASS
# --------------------------------------------------

class NLIVerifier:

    def __init__(self, model_name=MODEL_NAME):

        print("Loading NLI model...")

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name
        )

        print("NLI model loaded successfully!")

    def verify_claim(self, claim, reference_fact):

        if not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not reference_fact.strip():
            raise ValueError("Reference evidence cannot be empty.")

        # Evidence is the premise; the generated claim is the hypothesis.
        input_text = (
            f"premise: {reference_fact} "
            f"hypothesis: {claim}"
        )

        prediction = self.classifier(
            input_text,
            truncation=True
        )[0]

        model_label = prediction["label"].lower()
        model_score = float(prediction["score"])

        # Interpret the predicted label.
        if "entail" in model_label:
            status = "SUPPORTED"

        elif "contrad" in model_label:
            status = "CONTRADICTED"

        elif "neutral" in model_label:
            status = "INSUFFICIENT EVIDENCE"

        else:
            status = "UNRECOGNIZED MODEL LABEL"

        return {
            "claim": claim,
            "reference_fact": reference_fact,
            "status": status,
            "model_label": prediction["label"],
            "prediction_score": round(model_score, 4)
        }


# --------------------------------------------------
# 3. SAMPLE EXECUTION
# --------------------------------------------------

def main():

    verifier = NLIVerifier()

    test_cases = [
        {
            "claim": "Python was created by Guido van Rossum.",
            "evidence": (
                "Guido van Rossum created the Python "
                "programming language."
            ),
            "expected": "SUPPORTED"
        },
        {
            "claim": "Python was created by Microsoft.",
            "evidence": (
                "Guido van Rossum created the Python "
                "programming language."
            ),
            "expected": "CONTRADICTED"
        },
        {
            "claim": "Python was first released in 1991.",
            "evidence": (
                "Python is a high-level, general-purpose "
                "programming language."
            ),
            "expected": "INSUFFICIENT EVIDENCE"
        }
    ]

    print("\n" + "=" * 65)
    print("       AI CONFIDENCE / VERIFICATION SYSTEM")
    print("               NLI VERIFICATION REPORT")
    print("=" * 65)

    for index, test in enumerate(test_cases, start=1):

        result = verifier.verify_claim(
            claim=test["claim"],
            reference_fact=test["evidence"]
        )

        print(f"\nTest Case {index}")
        print("-" * 65)
        print("Generated Claim :", result["claim"])
        print("Reference Fact  :", result["reference_fact"])
        print("Status          :", result["status"])
        print("Model Label     :", result["model_label"])

        print(
            "Prediction Score:",
            f'{result["prediction_score"]:.2%}'
        )

        # FIX: Keep this statement inside the for loop.
        print("Expected Status :", test["expected"])

        # Optional: Compare actual and expected results.
        if result["status"] == test["expected"]:
            print("Test Result     : PASS")
        else:
            print("Test Result     : CHECK PREDICTION")

    print("\n" + "=" * 65)
    print("NLI testing completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()

