
"""
AI Confidence / Verification System
-----------------------------------
Integrates:
1. Rule-based claim extraction
2. Semantic similarity matching
3. Natural Language Inference (NLI)

The system evaluates individual claims against
the most relevant supplied reference evidence.
"""

import re

from verification_engine import AIVerificationEngine
from nli_verifier import NLIVerifier


class IntegratedVerifier:

    # Recognized auxiliary and modal verbs.
    # These are used to identify coordinated factual clauses.
    CLAUSE_START_WORDS = (
        "is", "are", "was", "were",
        "has", "have", "had",
        "can", "could", "will", "would",
        "should", "must", "does", "do", "did"
    )

    def __init__(self, similarity_threshold=0.70):

        print("Initializing Integrated Verification System...")

        self.similarity_engine = AIVerificationEngine(
            threshold=similarity_threshold
        )

        self.nli_engine = NLIVerifier()

        print("Integrated system ready!")


    @classmethod
    def extract_claims(cls, generated_text):
        """
        Controlled rule-based claim extraction.

        Separates coordinated factual clauses while preserving
        ordinary phrases such as 'easy to learn and use'.
        """

        if not generated_text or not generated_text.strip():
            return []

        sentences = AIVerificationEngine.split_sentences(
            generated_text
        )
        
        extracted_claims = []
        clause_pattern = re.compile(
            r"\s+\band\b\s+(?=(?:is|are|was|were|"
            r"has|have|had|can|could|will|would|"
            r"should|must|does|do|did)\b)",
            re.IGNORECASE
        )
        
        subject_pattern = re.compile(
            r"^(.+?)\s+"
            r"(?:is|are|was|were|has|have|had|"
            r"can|could|will|would|should|must|"
            r"does|do|did)\b",
            re.IGNORECASE
            
        )
        # Fallback: identifies a simple sentence-opening subject.
        fallback_subject_pattern = re.compile(
            r"^(?:the|a|an)?\s*([A-Za-z][A-Za-z'-]*)\b",
            re.IGNORECASE
            
        )


        for sentence in sentences:

            parts = clause_pattern.split(sentence)
            
            if len(parts) == 1:
                extracted_claims.append(sentence.strip())
                continue
            
            current_claim = parts[0].strip() 
            
            subject_match = subject_pattern.match(current_claim)
            
            if subject_match:            
                subject = subject_match.group(1).strip()

            else:            
                fallback_match = fallback_subject_pattern.match(
                    current_claim
                 )
                
                if not fallback_match:
                    extracted_claims.append(sentence.strip())
                    continue
                
                subject = fallback_match.group(1).strip()

            for clause in parts[1:]:
                clause = clause.strip()
                
                if not clause:
                    continue
                
                # Reconstruct the next claim using the identified subject.
                reconstructed_claim = subject + " " + clause
                
                extracted_claims.append(
                current_claim.strip().rstrip(" .") + "."
                
                )
                
                current_claim = reconstructed_claim

            if current_claim:
                extracted_claims.append(
                    current_claim.strip().rstrip(" .") + "."
                )
            
        return extracted_claims



    def verify_answer(self, generated_text, reference_facts):

        if not generated_text or not generated_text.strip():
            raise ValueError(
                "Generated text cannot be empty."
            )

        if not reference_facts:
            raise ValueError(
                "At least one reference fact is required."
            )

        if any(
            not isinstance(fact, str) or not fact.strip()
            for fact in reference_facts
        ):
            raise ValueError(
                "Reference facts must be non-empty strings."
            )

        # Step 1: Extract individual claims.
        claims = self.extract_claims(generated_text)

        if not claims:
            raise ValueError(
                "No claims could be extracted from generated text."
            )

        print(f"\nExtracted {len(claims)} individual claim(s).")

        for index, claim in enumerate(claims, start=1):
            print(f"Claim {index}: {claim}")

        # Step 2: Match extracted claims against reference facts.
        # The existing semantic similarity engine is preserved.
        similarity_results = self.similarity_engine.verify(
            generated_text="\n".join(claims),
            reference_facts=reference_facts
        )

        final_results = []

        # Step 3: Validate each claim using NLI.
        for result in similarity_results:

            similarity = result.similarity_score
            claim = result.sentence
            evidence = result.matched_reference

            # Avoid treating unrelated evidence as a contradiction.
            if similarity < self.similarity_engine.threshold:

                nli_status = "INSUFFICIENT EVIDENCE"
                nli_score = None
                model_label = (
                    "NLI not run: similarity below threshold"
                )

            else:

                nli_result = self.nli_engine.verify_claim(
                    claim=claim,
                    reference_fact=evidence
                )

                nli_status = nli_result["status"]
                nli_score = nli_result["prediction_score"]
                model_label = nli_result["model_label"]

            final_results.append({
                "claim": claim,
                "evidence": evidence,
                "similarity": similarity,
                "nli_status": nli_status,
                "nli_score": nli_score,
                "model_label": model_label
            })

        return final_results

    @staticmethod
    def display_report(results):

        print("\n" + "=" * 75)
        print("       AI CONFIDENCE / VERIFICATION SYSTEM")
        print("              INTEGRATED REPORT")
        print("=" * 75)

        for index, result in enumerate(results, start=1):

            print(f"\nClaim {index}")
            print("-" * 75)

            print("Generated Claim :", result["claim"])
            print("Reference Fact  :", result["evidence"])

            print(
                "Similarity Score:",
                f'{result["similarity"]:.2%}'
            )

            print("NLI Status      :", result["nli_status"])
            print("Model Label     :", result["model_label"])

            if result["nli_score"] is not None:
                print(
                    "NLI Model Score :",
                    f'{result["nli_score"]:.2%}'
                )
            else:
                print("NLI Model Score : Not calculated")

        supported = sum(
            result["nli_status"] == "SUPPORTED"
            for result in results
        )

        contradicted = sum(
            result["nli_status"] == "CONTRADICTED"
            for result in results
        )

        insufficient = sum(
            result["nli_status"] == "INSUFFICIENT EVIDENCE"
            for result in results
        )

        print("\n" + "=" * 75)
        print("                 VERIFICATION SUMMARY")
        print("=" * 75)

        print("Total Claims             :", len(results))
        print("Supported Claims         :", supported)
        print("Contradicted Claims      :", contradicted)
        print("Insufficient Evidence    :", insufficient)

        print("\nNote:")
        print(
            "Similarity scores measure semantic relatedness, "
            "not factual truth."
        )
        print(
            "NLI classifications are model predictions and "
            "require further evaluation."
        )

        print("=" * 75)


def main():

    # Example compound answer for testing claim extraction.
    generated_text = (
        "Python was created by Guido van Rossum "
        "and is the most widely used programming language "
        "in every country."
    )

    reference_facts = [
        "Guido van Rossum created the Python programming language.",
        "Python was first released in 1991.",
        "Python is a high-level, general-purpose programming language."
    ]

    verifier = IntegratedVerifier(
        similarity_threshold=0.70
    )

    results = verifier.verify_answer(
        generated_text=generated_text,
        reference_facts=reference_facts
    )

    verifier.display_report(results)


if __name__ == "__main__":

    tests = [
         "Python was created by Guido van Rossum and is widely used.",
    "Python is easy to learn and use.",
    "Earth revolves around the Sun and has one natural satellite."
    ]

    for sentence in tests:
        print("\nInput:", sentence)
        print(
            "Extracted:",
            IntegratedVerifier.extract_claims(sentence)
        )