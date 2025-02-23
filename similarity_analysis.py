"""
Similarity Analysis Module

This module provides functionality for computing similarity scores between feature
vectors extracted from face images. It implements multiple similarity metrics
and provides a clean interface for comparing features.

The module follows the Single Responsibility Principle by focusing solely on
similarity computation, separate from feature extraction or face detection.

Dependencies:
    - numpy
    - scipy (optional, for additional distance metrics)
"""

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
import numpy as np
from typing import (
    Dict,
    List,
    Optional,
)


@dataclass
class SimilarityScore:
    """
    Container for similarity comparison results.

    Attributes:
        score (float): Primary similarity score (0-1, where 1 is most similar)
        metric (str): Name of the similarity metric used
        confidence (float): Confidence level in the similarity score (0-1)
        metadata (Dict): Additional comparison information and metrics
    """

    score: float
    metric: str
    confidence: float
    metadata: Dict = None

    def __post_init__(self):
        """Validate the similarity score after initialization."""
        if not 0 <= self.score <= 1:
            raise ValueError("Similarity score must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.metadata is None:
            self.metadata = {}


class SimilarityMetric(ABC):
    """
    Abstract base class for similarity metric implementations.

    This class defines the interface for similarity metrics, allowing for
    different comparison strategies while maintaining a consistent interface.
    """

    @abstractmethod
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> SimilarityScore:
        """
        Compute similarity between two feature vectors.

        Args:
            vector1 (np.ndarray): First feature vector
            vector2 (np.ndarray): Second feature vector

        Returns:
            SimilarityScore: Computed similarity with metadata

        Raises:
            ValueError: If vectors are incompatible or invalid
        """
        pass

    def validate_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> None:
        """
        Validate input vectors for compatibility.

        Args:
            vector1 (np.ndarray): First feature vector
            vector2 (np.ndarray): Second feature vector

        Raises:
            ValueError: If vectors are incompatible or invalid
        """
        if vector1.shape != vector2.shape:
            raise ValueError(
                f"Vector shapes do not match: {vector1.shape} != {vector2.shape}"
            )
        if len(vector1.shape) != 1:
            raise ValueError("Vectors must be 1-dimensional")
        if not (np.isfinite(vector1).all() and np.isfinite(vector2).all()):
            raise ValueError("Vectors contain non-finite values")


class CosineSimilarity(SimilarityMetric):
    """Cosine similarity metric with enhanced discrimination."""

    def __init__(self, base_threshold: float = 0.5):
        self.base_threshold = base_threshold

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> SimilarityScore:
        self.validate_vectors(vector1, vector2)

        # Compute cosine similarity
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return SimilarityScore(
                score=0.0,
                metric="cosine",
                confidence=0.0,
                metadata={"error": "Zero magnitude vector(s)"}
            )

        # Compute raw cosine similarity
        cos_sim = np.dot(vector1, vector2) / (norm1 * norm2)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # Apply non-linear scaling to make the metric more discriminative
        scaled_score = self._scale_similarity(cos_sim)
        
        # Compute confidence
        confidence = self._compute_confidence(cos_sim, norm1, norm2)

        return SimilarityScore(
            score=float(scaled_score),
            metric="cosine",
            confidence=float(confidence),
            metadata={
                "raw_cosine": float(cos_sim),
                "magnitude1": float(norm1),
                "magnitude2": float(norm2)
            }
        )

    def _scale_similarity(self, cos_sim: float) -> float:
        """
        Apply more stringent non-linear scaling to better differentiate between faces.
        
        This implements a much stricter similarity curve:
        - Scores below 0.7 are scaled down significantly
        - Scores between 0.7 and 0.85 are scaled linearly
        - Scores above 0.85 require extremely high raw similarity
        """
        # Convert from [-1, 1] to [0, 1] range
        score = (cos_sim + 1) / 2
        
        # Apply stricter non-linear scaling
        if score < 0.7:
            # Much more aggressive scaling for low similarity
            scaled = score * 0.3  # Significantly reduce low similarity scores
        elif score < 0.85:
            # Linear scaling for medium similarity
            scaled = 0.21 + (score - 0.7) * 0.6
        else:
            # Very stringent scaling for high similarity claims
            scaled = 0.3 + (score - 0.85) * 2.0
            
        return np.clip(scaled, 0.0, 1.0)

    def _compute_confidence(self, cos_sim: float, norm1: float, norm2: float) -> float:
        """Compute more stringent confidence score based on multiple factors."""
        # Base confidence on vector magnitudes with stricter threshold
        magnitude_confidence = min(1.0, (norm1 * norm2) / (2e-10 + norm1 * norm2))
        
        # Confidence based on similarity value - more skeptical of medium scores
        similarity_confidence = 1.0 - abs(cos_sim - 0.8) * 3
        
        # Add threshold penalty
        if cos_sim < 0.7:
            similarity_confidence *= 0.5
        
        # Combine confidence scores with adjusted weights
        confidence = magnitude_confidence * 0.8 + similarity_confidence * 0.2
        
        return float(np.clip(confidence, 0.0, 1.0))


class EuclideanSimilarity(SimilarityMetric):
    """
    Euclidean distance-based similarity metric.

    This metric converts Euclidean distance to a similarity score in [0, 1]
    using a Gaussian kernel: similarity = exp(-distance²/2σ²)
    where σ (sigma) controls the sensitivity of the similarity measure.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Initialize Euclidean similarity metric.

        Args:
            sigma (float): Gaussian kernel width parameter
        """
        self.sigma = sigma

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> SimilarityScore:
        """
        Compute Euclidean similarity between two feature vectors.

        Args:
            vector1 (np.ndarray): First feature vector
            vector2 (np.ndarray): Second feature vector

        Returns:
            SimilarityScore: Normalized similarity score (0-1) with metadata
        """
        self.validate_vectors(vector1, vector2)

        # Compute Euclidean distance with numeric stability
        diff = vector1 - vector2
        distance = np.sqrt(np.sum(diff * diff))

        # Convert distance to similarity score using Gaussian kernel
        score = np.exp(-(distance**2) / (2 * self.sigma**2))

        # Compute confidence based on distance relative to sigma
        confidence = np.exp(-distance / self.sigma)

        return SimilarityScore(
            score=float(score),
            metric="euclidean",
            confidence=float(confidence),
            metadata={"distance": float(distance), "sigma": float(self.sigma)},
        )


def compare_features(
    vector1: np.ndarray, vector2: np.ndarray, metric: Optional[SimilarityMetric] = None
) -> SimilarityScore:
    """
    Compare two feature vectors using the specified similarity metric.

    This is the main interface function for similarity comparison. It handles
    input validation and provides a default metric if none is specified.

    Args:
        vector1 (np.ndarray): First feature vector
        vector2 (np.ndarray): Second feature vector
        metric (Optional[SimilarityMetric]): Similarity metric to use.
            If None, uses CosineSimilarity.

    Returns:
        SimilarityScore: Computed similarity with metadata

    Raises:
        ValueError: If vectors are incompatible or invalid
        Exception: For other unexpected errors during comparison
    """
    if metric is None:
        metric = CosineSimilarity()

    try:
        return metric.compute(vector1, vector2)
    except Exception as e:
        raise Exception(f"Similarity computation failed: {e!s}")


def batch_compare_features(
    query_vector: np.ndarray,
    reference_vectors: List[np.ndarray],
    metric: Optional[SimilarityMetric] = None,
    threshold: float = 0.0,
) -> List[SimilarityScore]:
    """
    Compare a query vector against multiple reference vectors.

    This utility function efficiently compares one vector against many,
    optionally filtering by a similarity threshold.

    Args:
        query_vector (np.ndarray): Query feature vector
        reference_vectors (List[np.ndarray]): List of reference vectors
        metric (Optional[SimilarityMetric]): Similarity metric to use
        threshold (float): Minimum similarity score to include in results

    Returns:
        List[SimilarityScore]: List of similarity scores above threshold
    """
    if metric is None:
        metric = CosineSimilarity()

    results = []
    for ref_vector in reference_vectors:
        try:
            score = metric.compute(query_vector, ref_vector)
            if score.score >= threshold:
                results.append(score)
        except Exception:
            continue

    return sorted(results, key=lambda x: x.score, reverse=True)
