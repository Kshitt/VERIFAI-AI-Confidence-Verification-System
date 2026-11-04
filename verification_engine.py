"""
AI Confidence/Verification System
---------------------------------
Compares AI-generated sentences with reference facts using
Sentence Transformers and cosine similarity.

Important:
Cosine similarity measures semantic similarity, not factual truth.
A high score does not guarantee that a claim is correct.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# 2. DATA STRUCTURE
# ============================================================

@dataclass
class VerificationResult:
    sentence: str
    matched_reference: str
    similarity_score: float
    status: str


# ============================================================
# 3. AI VERIFICATION ENGINE
# ============================================================

class AIVerificationEngine:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the semantic similarity model.
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Basic sentence splitting.

        For production use, replace this with spaCy or another
        sentence tokenizer to handle abbreviations more reliably.
        """
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def verify(
        self,
        generated_text: str,
        reference_facts: List[str]
    ) -> List[VerificationResult]:
        """
        Compare each generated sentence against all reference facts.

        Returns the best matching reference and its cosine similarity.
        """

        if not generated_text.strip():
            raise ValueError("Generated text cannot be empty.")

        if not reference_facts:
            raise ValueError("At least one reference fact is required.")

        if any(not fact.strip() for fact in reference_facts):
            raise ValueError("Reference facts cannot be empty strings.")

        generated_sentences = self.split_sentences(generated_text)

        if not generated_sentences:
            raise ValueError("No sentences were found in generated text.")

        # Generate semantic embeddings.
        sentence_embeddings = self.model.encode(
            generated_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        reference_embeddings = self.model.encode(
            reference_facts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Compare every generated sentence with every reference fact.
        similarity_matrix = cosine_similarity(
            sentence_embeddings,
            reference_embeddings
        )

        results = []

        for index, sentence in enumerate(generated_sentences):

            # Identify the most similar reference fact.
            best_match_index = int(
                np.argmax(similarity_matrix[index])
            )

            best_reference = reference_facts[best_match_index]

            # Raw cosine similarity, converted to a 0-1 scale.
            # This is NOT a calibrated probability of truth.
            raw_similarity = float(
                similarity_matrix[index, best_match_index]
            )

            similarity_score = float(
                np.clip(raw_similarity, 0.0, 1.0)
            )

            # Apply the configured threshold.
            if similarity_score >= self.threshold:
                status = "MATCHED - REQUIRES FACTUAL VALIDATION"
            else:
                status = "LOW SIMILARITY - FLAGGED"

            results.append(
                VerificationResult(
                    sentence=sentence,
                    matched_reference=best_reference,
                    similarity_score=similarity_score,
                    status=status
                )
            )

        return results

    @staticmethod
    def calculate_overall_score(
        results: List[VerificationResult]
    ) -> float:
        """
        Calculate the mean best-match similarity across sentences.

        This is a similarity-based summary, not a probability
        that the entire response is factually correct.
        """
        if not results:
            return 0.0

        return float(
            np.mean([
                result.similarity_score
                for result in results
            ])
        )

    def display_report(
        self,
        results: List[VerificationResult]
    ) -> None:
        """
        Display the sentence-level verification report.
        """

        overall_score = self.calculate_overall_score(results)

        flagged_count = sum(
            result.similarity_score < self.threshold
            for result in results
        )

        print("\n" + "=" * 75)
        print("          AI CONFIDENCE / VERIFICATION SYSTEM")
        print("=" * 75)

        for index, result in enumerate(results, start=1):

            print(f"\nSentence {index}:")
            print(f"Generated Text : {result.sentence}")
            print(f"Reference Fact : {result.matched_reference}")
            print(
                f"Similarity     : "
                f"{result.similarity_score:.2%}"
            )
            print(f"Status         : {result.status}")

        print("\n" + "-" * 75)
        print("                 FINAL VERIFICATION SUMMARY")
        print("-" * 75)

        print(f"Total Sentences       : {len(results)}")
        print(f"Flagged Sentences     : {flagged_count}")
        print(f"Similarity Threshold  : {self.threshold:.2f}")
        print(f"Mean Similarity Score : {overall_score:.2%}")

        if flagged_count:
            print(
                "\nWARNING: Some sentences have low semantic "
                "similarity to the available reference facts."
            )
        else:
            print(
                "\nAll sentences passed the semantic similarity "
                "threshold. Factual correctness is NOT guaranteed."
            )

        print("=" * 75)


# ============================================================
# 4. SAMPLE EXECUTION
# ============================================================

def main():

    # Sample paragraph generated by an AI assistant.
    generated_text = (
        "Python was created by Guido van Rossum. "
        "Python was first released in 1991. "
        "Python was developed as a replacement for Java. "
        "Python supports object-oriented programming."
    )

    # Reference facts collected from a trusted knowledge source.
    reference_facts = [
        "Guido van Rossum created the Python programming language.",
        "Python was first released in 1991.",
        "Python supports object-oriented programming.",
        "Python is a high-level, general-purpose programming language."
    ]

    # Initialize the engine.
    engine = AIVerificationEngine(
        threshold=0.70
    )

    # Verify the generated response.
    results = engine.verify(
        generated_text=generated_text,
        reference_facts=reference_facts
    )

    # Display the results.
    engine.display_report(results)


if __name__ == "__main__":
    main()