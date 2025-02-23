"""
Local Analysis Integration Module

This module provides detailed face analysis using rule-based analysis
and feature-specific comparison logic.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)


logger = logging.getLogger(__name__)


@dataclass
class LocalAnalysisResponse:
    """Container for local analysis responses."""

    content: str
    metadata: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LocalAnalyzer:
    """Local face analysis system using detailed rule-based analysis."""

    def __init__(self):
        """Initialize the local analyzer with detailed comparison metrics."""
        # Detailed similarity interpretation ranges
        self.similarity_ranges = {
            (0.0, 0.3): ("very different", "likely different people", "low"),
            (0.3, 0.5): ("notably different", "probably different people", "moderate"),
            (0.5, 0.65): ("somewhat similar", "possible relation", "moderate"),
            (0.65, 0.8): ("similar", "likely related or same person", "high"),
            (0.8, 0.9): ("very similar", "very likely same person", "very high"),
            (0.9, 1.0): (
                "nearly identical",
                "almost certainly same person",
                "extremely high",
            ),
        }

        # Feature-specific analysis templates
        self.feature_analysis = {
            "high_similarity": {
                "eyes": "similarly shaped and positioned eyes with matching characteristics",
                "nose": "matching nose structure including bridge and tip shape",
                "face": "consistent facial proportions and symmetry",
                "jaw": "aligned jawline and cheekbone structure",
                "overall": "strong structural similarity across features",
            },
            "medium_similarity": {
                "eyes": "somewhat similar eye shapes with some variations",
                "nose": "comparable nose features with minor differences",
                "face": "similar overall proportions with some asymmetry",
                "jaw": "partially matching jaw and cheekbone structure",
                "overall": "noticeable similarities with some distinct differences",
            },
            "low_similarity": {
                "eyes": "distinctly different eye shapes or positioning",
                "nose": "different nose structures and characteristics",
                "face": "different facial proportions and symmetry",
                "jaw": "distinct jaw and cheekbone structures",
                "overall": "significant differences across facial features",
            },
        }

    def _get_similarity_category(self, score: float) -> tuple:
        """Get detailed similarity description and confidence."""
        for (lower, upper), (desc, relation, conf) in self.similarity_ranges.items():
            if lower <= score < upper:
                return desc, relation, conf
        return "unknown similarity", "unclear relation", "unknown confidence"

    def _analyze_quality_factors(
        self, face1_metadata: Dict, face2_metadata: Dict
    ) -> Dict:
        """Analyze various quality factors affecting comparison."""
        quality1 = face1_metadata.get("quality_score", 0.5)
        quality2 = face2_metadata.get("quality_score", 0.5)

        factors = {
            "overall_quality": (
                "good"
                if min(quality1, quality2) > 0.7
                else "acceptable" if min(quality1, quality2) > 0.5 else "poor"
            ),
            "lighting": "good" if min(quality1, quality2) > 0.6 else "suboptimal",
            "clarity": (
                "sufficient" if min(quality1, quality2) > 0.5 else "insufficient"
            ),
            "affects_comparison": quality1 < 0.5 or quality2 < 0.5,
        }

        return factors

    def _get_feature_analysis(self, similarity_score: float) -> Dict:
        """Get detailed feature analysis based on similarity score."""
        if similarity_score > 0.8:
            return self.feature_analysis["high_similarity"]
        elif similarity_score > 0.5:
            return self.feature_analysis["medium_similarity"]
        else:
            return self.feature_analysis["low_similarity"]

    def _generate_detailed_analysis(
        self, similarity_score: float, face1_metadata: Dict, face2_metadata: Dict
    ) -> str:
        """Generate comprehensive analysis with specific details."""

        # Get basic categorization
        similarity_desc, relation_desc, confidence = self._get_similarity_category(
            similarity_score
        )
        quality_factors = self._analyze_quality_factors(face1_metadata, face2_metadata)
        feature_details = self._get_feature_analysis(similarity_score)

        analysis = f"""Face Comparison Analysis:

1. Overall Assessment:
- Similarity Level: {similarity_desc.title()}
- Relationship Indication: {relation_desc.capitalize()}
- Confidence Level: {confidence.title()}
- Image Quality: {quality_factors['overall_quality'].title()}
{f'- Note: {quality_factors["lighting"]} lighting conditions' if quality_factors["affects_comparison"] else ''}

2. Specific Facial Features:
- Eyes: {feature_details['eyes'].capitalize()}
- Nose: {feature_details['nose'].capitalize()}
- Face Structure: {feature_details['face'].capitalize()}
- Jaw and Cheekbones: {feature_details['jaw'].capitalize()}

3. Detailed Comparison:
- Similarity Score: {similarity_score:.3f} out of 1.0
- Primary Match Strength: {feature_details['overall'].capitalize()}
"""

        # Add quality-specific notes
        if quality_factors["affects_comparison"]:
            analysis += "\nQuality Considerations:\n"
            analysis += f"- Image Clarity: {quality_factors['clarity'].capitalize()}\n"
            analysis += (
                f"- Lighting Conditions: {quality_factors['lighting'].capitalize()}\n"
            )
            analysis += "- These factors may affect the accuracy of the comparison"

        return analysis

    async def compare_feature_vectors(
        self,
        vector1: List[float],
        vector2: List[float],
        similarity_score: float,
        metric_name: str,
        face1_metadata: Optional[Dict] = None,
        face2_metadata: Optional[Dict] = None,
    ) -> LocalAnalysisResponse:
        """Compare faces and generate detailed natural language analysis."""

        try:
            analysis = self._generate_detailed_analysis(
                similarity_score, face1_metadata or {}, face2_metadata or {}
            )

            return LocalAnalysisResponse(
                content=analysis,
                metadata={
                    "similarity_score": similarity_score,
                    "face1_metadata": face1_metadata,
                    "face2_metadata": face2_metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error in local analysis: {e!s}")
            # Provide simplified analysis on error
            return LocalAnalysisResponse(
                content=f"Basic Analysis:\nSimilarity Score: {similarity_score:.3f}\n"
                f"Quality: {self._get_similarity_category(similarity_score)[0]}",
                metadata={"error": str(e)},
            )


def create_client() -> LocalAnalyzer:
    """Create a local analyzer instance."""
    return LocalAnalyzer()
