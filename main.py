"""
Main Application Module

This module serves as the entry point for the face comparison application,
orchestrating all components and providing both GUI and command-line interfaces.
It follows the Single Responsibility Principle by focusing solely on application
setup and coordination between modules.

Dependencies:
    - argparse (for command-line argument parsing)
    - logging (for application-wide logging)
    - All application modules (GUI, face detection, feature extraction, etc.)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

from config import Config
from face_detection import detect_faces, extract_face_regions
from feature_extraction import (
    DeepFeatureExtractor,
    HOGFeatureExtractor,
    extract_features,
)
from gui import ComparisonApp
from similarity_analysis import CosineSimilarity, EuclideanSimilarity, compare_features


def setup_logging() -> None:
    """
    Configure application-wide logging.

    Sets up logging handlers, formats, and levels based on configuration.
    Creates log directory if it doesn't exist.
    """
    # Ensure log directory exists
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Config.LOG_DIR / "app.log"),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the application.

    Returns:
        argparse.Namespace: Parsed command-line arguments

    Command-line options:
        --cli: Run in command-line mode instead of GUI
        --image1: Path to first image for CLI mode
        --image2: Path to second image for CLI mode
        --method: Feature extraction method (hog/deep)
    """
    parser = argparse.ArgumentParser(description="Face Comparison Application")

    parser.add_argument("--cli", action="store_true", help="Run in command-line mode")

    parser.add_argument(
        "--image1", type=str, help="Path to first image (required for CLI mode)"
    )

    parser.add_argument(
        "--image2", type=str, help="Path to second image (required for CLI mode)"
    )

    parser.add_argument(
        "--method",
        choices=["hog", "deep"],
        default="hog",
        help="Feature extraction method",
    )

    return parser.parse_args()


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments for CLI mode.

    Args:
        args (argparse.Namespace): Parsed command-line arguments

    Raises:
        ValueError: If required arguments are missing for CLI mode
    """
    if args.cli:
        if not args.image1 or not args.image2:
            raise ValueError("Both --image1 and --image2 are required in CLI mode")

        # Validate image paths
        if not Path(args.image1).exists():
            raise ValueError(f"Image not found: {args.image1}")
        if not Path(args.image2).exists():
            raise ValueError(f"Image not found: {args.image2}")


def compare_faces_cli(
    image1_path: str, image2_path: str, method: str = "hog"
) -> Tuple[float, float]:
    """
    Compare two face images in CLI mode.

    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        method (str): Feature extraction method ('hog' or 'deep')

    Returns:
        Tuple[float, float]: Cosine and Euclidean similarity scores

    Raises:
        Exception: If face detection or comparison fails
    """
    # Create feature extractor based on method
    extractor = HOGFeatureExtractor() if method == "hog" else DeepFeatureExtractor()

    # Process first image
    faces1 = detect_faces(image1_path)
    if not faces1:
        raise ValueError(f"No faces detected in {image1_path}")
    regions1 = extract_face_regions(image1_path, faces1)
    features1 = extract_features(regions1[0], extractor)

    # Process second image
    faces2 = detect_faces(image2_path)
    if not faces2:
        raise ValueError(f"No faces detected in {image2_path}")
    regions2 = extract_face_regions(image2_path, faces2)
    features2 = extract_features(regions2[0], extractor)

    # Compare features using both similarity metrics
    cosine_score = compare_features(
        features1.vector, features2.vector, CosineSimilarity()
    )
    euclidean_score = compare_features(
        features1.vector, features2.vector, EuclideanSimilarity()
    )

    return cosine_score.score, euclidean_score.score


def run_cli_mode(args: argparse.Namespace) -> None:
    """
    Run the application in command-line mode.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    try:
        cosine_score, euclidean_score = compare_faces_cli(
            args.image1, args.image2, args.method
        )

        print("\nFace Comparison Results:")
        print(f"Cosine Similarity: {cosine_score:.3f}")
        print(f"Euclidean Similarity: {euclidean_score:.3f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def run_gui_mode() -> None:
    """
    Run the application in GUI mode.

    Initializes and starts the graphical user interface.
    """
    try:
        app = ComparisonApp()
        app.run()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to start GUI: {str(e)}")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the application.

    Handles initialization, argument parsing, and mode selection.
    """
    try:
        # Initialize logging
        setup_logging()
        logger = logging.getLogger(__name__)

        # Validate configuration
        Config.validate_settings()

        # Parse and validate arguments
        args = parse_arguments()
        validate_cli_arguments(args)

        # Run in appropriate mode
        if args.cli:
            run_cli_mode(args)
        else:
            run_gui_mode()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
