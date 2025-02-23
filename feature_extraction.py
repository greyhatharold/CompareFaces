"""
Feature Extraction Module

This module handles the extraction of features from face images, providing a standardized
interface for different feature extraction methods. It supports both traditional computer
vision approaches and deep learning-based embeddings.

The module follows the Single Responsibility Principle by focusing solely on converting
face images into feature vectors that can be used for face comparison or recognition.

Dependencies:
    - numpy
    - opencv-python (cv2)
"""

import logging
from abc import (
    ABC,
    abstractmethod,
)
import bz2
import cv2
from dataclasses import dataclass
import dlib
import numpy as np
from pathlib import Path
import shutil
from typing import (
    Dict,
    Optional,
)
import urllib.request


logger = logging.getLogger(__name__)


@dataclass
class FaceFeatures:
    """
    Container for extracted face features.

    Attributes:
        vector (np.ndarray): The feature vector/embedding
        method (str): Name of the extraction method used
        dimension (int): Length of the feature vector
        metadata (Dict): Additional information about the extraction
    """

    vector: np.ndarray
    method: str
    dimension: int
    metadata: Dict = None

    def __post_init__(self):
        """Validate the feature vector after initialization."""
        if len(self.vector.shape) != 1:
            raise ValueError("Feature vector must be 1-dimensional")
        if self.vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension ({self.vector.shape[0]}) "
                f"doesn't match specified dimension ({self.dimension})"
            )
        if self.metadata is None:
            self.metadata = {}

    def normalize(self) -> None:
        """
        Normalize the feature vector to unit length (L2 normalization).
        This is often required for cosine similarity comparisons.
        """
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction implementations.

    This class defines the interface for feature extractors, allowing for
    different implementation strategies (HOG, deep learning embeddings, etc.)
    while maintaining a consistent interface.
    """

    @abstractmethod
    def extract(self, face_image: np.ndarray) -> FaceFeatures:
        """
        Extract features from a face image.

        Args:
            face_image (np.ndarray): Input face image (BGR format)

        Returns:
            FaceFeatures: Extracted features with metadata

        Raises:
            ValueError: If the input image is invalid
        """
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of the feature vector produced by this extractor.

        Returns:
            int: Dimension of the feature vector
        """
        pass


class HOGFeatureExtractor(FeatureExtractor):
    """
    Traditional HOG (Histogram of Oriented Gradients) feature extractor.

    This implementation uses OpenCV's HOG descriptor to extract features,
    which are effective for capturing shape and gradient information.
    """

    def __init__(
        self,
        cell_size: tuple = (8, 8),
        block_size: tuple = (2, 2),
        target_size: tuple = (64, 64),
    ):
        """
        Initialize HOG feature extractor.

        Args:
            cell_size (tuple): Size of cells for HOG computation
            block_size (tuple): Size of blocks for normalization
            target_size (tuple): Size to resize face images to
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.target_size = target_size

        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor(
            _winSize=self.target_size,
            _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
            _blockStride=(cell_size[0], cell_size[1]),
            _cellSize=cell_size,
            _nbins=9,
        )

    def get_feature_dimension(self) -> int:
        """Calculate the dimension of the HOG feature vector."""
        cells_x = self.target_size[0] // self.cell_size[0]
        cells_y = self.target_size[1] // self.cell_size[1]
        blocks_x = cells_x - self.block_size[0] + 1
        blocks_y = cells_y - self.block_size[1] + 1
        return blocks_x * blocks_y * self.block_size[0] * self.block_size[1] * 9

    def extract(self, face_image: np.ndarray) -> FaceFeatures:
        """
        Extract HOG features from a face image.

        Args:
            face_image (np.ndarray): Input face image (BGR format)

        Returns:
            FaceFeatures: HOG features with metadata

        Raises:
            ValueError: If the input image is invalid
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid input image")

        # Preprocess image
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size)

        # Extract HOG features
        features = self.hog.compute(resized)

        return FaceFeatures(
            vector=features.flatten(),
            method="HOG",
            dimension=self.get_feature_dimension(),
            metadata={
                "cell_size": self.cell_size,
                "block_size": self.block_size,
                "target_size": self.target_size,
            },
        )


class ModelDownloader:
    """Handles downloading and managing dlib model files."""

    MODELS_DIR = Path("models")
    MODELS = {
        "face_recognition": {
            "file": "dlib_face_recognition_resnet_model_v1.dat",
            "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        },
        "shape_predictor": {
            "file": "shape_predictor_68_face_landmarks.dat",
            "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        },
    }

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get the path to a model file, downloading it if necessary."""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        # Ensure models directory exists
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_info = cls.MODELS[model_name]
        model_path = cls.MODELS_DIR / model_info["file"]

        # Download and extract if not exists
        if not model_path.exists():
            logger.info(f"Downloading {model_name} model...")
            try:
                # Download compressed file
                compressed_path = model_path.with_suffix(".dat.bz2")
                urllib.request.urlretrieve(model_info["url"], compressed_path)

                # Decompress
                with bz2.BZ2File(compressed_path) as fr, open(model_path, "wb") as fw:
                    shutil.copyfileobj(fr, fw)

                # Clean up compressed file
                compressed_path.unlink()
                logger.info(f"Successfully downloaded {model_name} model")
            except Exception as e:
                logger.error(f"Failed to download {model_name} model: {e!s}")
                raise

        return model_path


class DeepFeatureExtractor(FeatureExtractor):
    """Deep learning-based feature extractor using dlib."""

    def __init__(self, threshold_multiplier: float = 0.8):
        try:
            recognition_path = ModelDownloader.get_model_path("face_recognition")
            predictor_path = ModelDownloader.get_model_path("shape_predictor")

            self.model = dlib.face_recognition_model_v1(str(recognition_path))
            self.shape_predictor = dlib.shape_predictor(str(predictor_path))
            self.embedding_size = 128
            self.detector = dlib.get_frontal_face_detector()
            self.threshold_multiplier = threshold_multiplier
        except Exception as e:
            logger.error(f"Failed to initialize deep feature extractor: {str(e)}")
            raise

    def get_feature_dimension(self) -> int:
        """Get the dimension of the deep feature embedding."""
        return self.embedding_size

    def extract(self, face_image: np.ndarray) -> FaceFeatures:
        try:
            # Enhance image quality first
            rgb_image = self._preprocess_image(face_image)
            height, width = rgb_image.shape[:2]
            rect = dlib.rectangle(0, 0, width, height)
            
            shape = self.shape_predictor(rgb_image, rect)
            
            # Extract features with balanced parameters
            face_descriptor = np.array(
                self.model.compute_face_descriptor(
                    rgb_image, 
                    shape, 
                    num_jitters=10,  # Reduced jitters for speed while maintaining quality
                    padding=0.3  # Balanced padding
                )
            )
            
            # Enhanced normalization
            face_descriptor = self._normalize_features(face_descriptor)
            
            # More balanced quality assessment
            quality_score = self._compute_quality_score(rgb_image, shape)
            
            # More lenient quality threshold
            if quality_score < 0.4:  # Reduced from 0.6
                logger.warning(f"Low quality face detected (score: {quality_score:.2f})")
            
            return FaceFeatures(
                vector=face_descriptor,
                method="DeepLearning-dlib",
                dimension=self.embedding_size,
                metadata={
                    "model": "dlib_resnet",
                    "image_size": (height, width),
                    "quality_score": quality_score,
                    "threshold_multiplier": self.threshold_multiplier
                }
            )
            
        except Exception as e:
            logger.error(f"Deep feature extraction failed: {str(e)}")
            raise ValueError(f"Feature extraction failed: {str(e)}")

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply enhanced normalization to features."""
        # Center the features
        features = features - np.mean(features)
        
        # Scale to unit variance
        features = features / (np.std(features) + 1e-10)
        
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features

    def _compute_quality_score(self, image: np.ndarray, shape) -> float:
        """Compute face quality score based on multiple factors"""
        try:
            # Check face symmetry
            left_eye = np.mean(
                np.array(
                    [
                        (shape.part(36).x, shape.part(36).y),
                        (shape.part(37).x, shape.part(37).y),
                        (shape.part(38).x, shape.part(38).y),
                        (shape.part(39).x, shape.part(39).y),
                        (shape.part(40).x, shape.part(40).y),
                        (shape.part(41).x, shape.part(41).y),
                    ]
                ),
                axis=0,
            )

            right_eye = np.mean(
                np.array(
                    [
                        (shape.part(42).x, shape.part(42).y),
                        (shape.part(43).x, shape.part(43).y),
                        (shape.part(44).x, shape.part(44).y),
                        (shape.part(45).x, shape.part(45).y),
                        (shape.part(46).x, shape.part(46).y),
                        (shape.part(47).x, shape.part(47).y),
                    ]
                ),
                axis=0,
            )

            # Calculate symmetry score
            eye_distance = np.linalg.norm(left_eye - right_eye)
            ideal_distance = image.shape[1] * 0.25  # Ideal eye distance
            symmetry_score = 1.0 - min(
                abs(eye_distance - ideal_distance) / ideal_distance, 1.0
            )

            # Calculate brightness and contrast
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            brightness_score = np.mean(gray) / 255.0
            contrast_score = np.std(gray) / 128.0

            # Combine scores
            quality_score = (
                symmetry_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3
            )

            return float(np.clip(quality_score, 0, 1))

        except Exception:
            return 0.5  # Default score if calculation fails

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve quality and compatibility.
        """
        try:
            # Ensure correct color format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # Convert to RGB for dlib
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic image enhancement
            rgb_image = self._enhance_image(rgb_image)
            
            return np.ascontiguousarray(rgb_image)
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return np.ascontiguousarray(image)

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality while preserving natural appearance.
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE on L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge((l,a,b))
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image


class EnhancedHOGFeatureExtractor(HOGFeatureExtractor):
    """Enhanced HOG feature extractor with dimensionality reduction."""

    def __init__(
        self,
        cell_size: tuple = (8, 8),
        block_size: tuple = (2, 2),
        target_size: tuple = (64, 64),
        reduction_factor: float = 0.5
    ):
        """Initialize enhanced HOG extractor."""
        super().__init__(cell_size, block_size, target_size)
        self.reduction_factor = reduction_factor

    def extract(self, face_image: np.ndarray) -> FaceFeatures:
        """Extract HOG features with dimensionality reduction."""
        # Get basic HOG features
        features = super().extract(face_image)
        
        # Apply dimensionality reduction using averaging
        orig_dim = features.vector.shape[0]
        new_dim = int(orig_dim * self.reduction_factor)
        
        # Ensure new dimension is even
        new_dim = new_dim + (new_dim % 2)
        
        # Reshape and average
        reduced_features = np.mean(
            features.vector.reshape(-1, orig_dim // new_dim), 
            axis=1
        )

        return FaceFeatures(
            vector=reduced_features,
            method="HOG-Enhanced",
            dimension=new_dim,
            metadata={
                **features.metadata,
                "reduction_factor": self.reduction_factor,
                "original_dimension": orig_dim
            },
        )


def extract_features(
    face_image: np.ndarray,
    extractor: Optional[FeatureExtractor] = None,
    method: str = "hog",
) -> FaceFeatures:
    """
    Extract features from a face image.

    Args:
        face_image (np.ndarray): Input face image
        extractor (Optional[FeatureExtractor]): Custom extractor
        method (str): Feature extraction method ('hog', 'hog-enhanced', 'deep')

    Returns:
        FaceFeatures: Extracted features

    Raises:
        ValueError: If input image is invalid
        Exception: For other extraction errors
    """
    if face_image is None or face_image.size == 0:
        raise ValueError("Invalid input image")

    if extractor is None:
        if method == "hog":
            extractor = HOGFeatureExtractor()
        elif method == "hog-enhanced":  # Changed from 'hog-pca'
            extractor = EnhancedHOGFeatureExtractor()
        elif method == "deep":
            extractor = DeepFeatureExtractor()
        else:
            raise ValueError(f"Unknown method: {method}")

    try:
        features = extractor.extract(face_image)
        features.normalize()
        return features
    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")
