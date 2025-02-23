"""
Face Detection Module

This module provides functionality for detecting faces in images using OpenCV's
Cascade Classifier. It follows the Single Responsibility Principle by focusing
solely on face detection operations and is designed to be easily extended with
different detection methods.

Dependencies:
    - OpenCV (cv2)
    - numpy
    - pathlib
"""

import logging
from abc import (
    ABC,
    abstractmethod,
)
import cv2
from dataclasses import dataclass
import dlib  # Add dlib for improved face detection
import numpy as np
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)


logger = logging.getLogger(__name__)


@dataclass
class FaceLocation:
    """Enhanced FaceLocation with additional attributes"""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    angle: Optional[float] = None
    quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Enhanced dictionary conversion"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "landmarks": self.landmarks,
            "angle": self.angle,
            "quality_score": self.quality_score,
        }


class FaceDetector(ABC):
    """
    Abstract base class for face detection implementations.

    This class defines the interface for face detectors, allowing for easy
    extension with different detection methods (e.g., deep learning models,
    different cascade classifiers, etc.).
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceLocation]:
        """
        Detect faces in the provided image.

        Args:
            image (np.ndarray): Input image as a NumPy array

        Returns:
            List[FaceLocation]: List of detected face locations
        """
        pass


class HaarCascadeDetector(FaceDetector):
    """
    Face detector implementation using OpenCV's Haar Cascade Classifier.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the Haar Cascade detector.

        Args:
            cascade_path (Optional[str]): Path to custom cascade classifier XML file.
                If None, uses the default frontal face classifier.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

    def detect(self, image: np.ndarray) -> List[FaceLocation]:
        """
        Detect faces in the provided image using Haar Cascade Classifier.

        Args:
            image (np.ndarray): Input image as a NumPy array

        Returns:
            List[FaceLocation]: List of detected face locations

        Note:
            The detection uses default parameters that can be adjusted for
            different use cases:
            - scaleFactor=1.1: How much the image size is reduced at each scale
            - minNeighbors=5: How many neighbors each candidate rectangle should have
            - minSize=(30, 30): Minimum possible face size
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        return [FaceLocation(x=x, y=y, width=w, height=h) for (x, y, w, h) in faces]


class DlibDetector(FaceDetector):
    """
    Face detector implementation using dlib's HOG face detector.
    Provides more robust detection than Haar Cascades.
    """

    def __init__(self):
        """Initialize the dlib face detector and shape predictor"""
        self.detector = dlib.get_frontal_face_detector()
        # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            self.has_predictor = True
        except:
            logger.warning(f"Landmark predictor not found at {predictor_path}")
            self.has_predictor = False

    def detect(self, image: np.ndarray) -> List[FaceLocation]:
        """
        Detect faces using dlib with balanced quality checks.
        """
        try:
            # Ensure correct image format and enhance if needed
            gray = self._prepare_image(image)

            # Detect faces with balanced threshold
            dlib_rects = self.detector(gray, 1)  # Reduced from 2 for better detection

            face_locations = []
            for rect in dlib_rects:
                # Get basic measurements
                x = rect.left()
                y = rect.top()
                width = rect.width()
                height = rect.height()

                # More lenient size threshold
                if width < 40 or height < 40:  # Reduced from 60
                    continue

                if self.has_predictor:
                    shape = self.predictor(gray, rect)
                    angle = self._calculate_face_angle(shape)
                    
                    # More lenient angle threshold
                    if abs(angle) > 30:  # Increased from 20
                        continue

                    # Balanced quality scoring
                    quality_score = self._compute_enhanced_quality_score(
                        gray, shape, rect, image.shape
                    )
                    
                    # More lenient quality threshold
                    if quality_score < 0.5:  # Reduced from 0.65
                        logger.warning(f"Low quality face detected (score: {quality_score:.2f})")
                    
                    landmarks = self._extract_landmarks(shape)
                else:
                    angle = None
                    landmarks = None
                    quality_score = 0.5

                face_locations.append(
                    FaceLocation(
                        x=x, y=y, width=width, height=height,
                        confidence=quality_score,
                        landmarks=landmarks,
                        angle=angle,
                        quality_score=quality_score,
                    )
                )

            return face_locations
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for face detection with format handling.
        """
        try:
            # Handle different color formats
            if len(image.shape) == 2:
                gray = image
            elif image.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:  # BGR/RGB
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic image normalization
            gray = cv2.equalizeHist(gray)
            
            return gray
        except Exception as e:
            logger.error(f"Image preparation failed: {str(e)}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _compute_enhanced_quality_score(
        self, gray: np.ndarray, shape, rect, image_shape: tuple
    ) -> float:
        """
        Compute a more comprehensive quality score for face detection.
        """
        # Size score
        face_area = rect.width() * rect.height()
        image_area = image_shape[0] * image_shape[1]
        size_score = min(1.0, face_area / (image_area * 0.1))  # Expect face to be at least 10% of image

        # Blur detection
        face_region = gray[rect.top():rect.bottom(), rect.left():rect.right()]
        laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500)  # Normalized blur score

        # Symmetry score
        symmetry_score = self._compute_symmetry_score(shape)

        # Combine scores with weights
        final_score = (
            size_score * 0.3 +
            blur_score * 0.4 +
            symmetry_score * 0.3
        )

        return float(np.clip(final_score, 0.0, 1.0))

    def _compute_symmetry_score(self, shape) -> float:
        """
        Compute face symmetry score using facial landmarks.
        """
        # Calculate symmetry using key facial landmarks
        left_eye = np.mean([(shape.part(36+i).x, shape.part(36+i).y) for i in range(6)], axis=0)
        right_eye = np.mean([(shape.part(42+i).x, shape.part(42+i).y) for i in range(6)], axis=0)
        
        # Check eye level alignment
        eye_y_diff = abs(left_eye[1] - right_eye[1])
        eye_alignment_score = 1.0 - min(eye_y_diff / 20.0, 1.0)
        
        return eye_alignment_score


def detect_faces(
    image_path: str, detector: Optional[FaceDetector] = None
) -> List[FaceLocation]:
    """
    Detect faces in an image file.

    Args:
        image_path (str): Path to the image file
        detector (Optional[FaceDetector]): Face detector implementation to use.
            If None, uses the default HaarCascadeDetector.

    Returns:
        List[FaceLocation]: List of detected face locations

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image file cannot be read
        Exception: For other unexpected errors during detection
    """
    try:
        # Validate image path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Use default detector if none provided
        if detector is None:
            detector = HaarCascadeDetector()

        # Perform detection
        return detector.detect(image)

    except (FileNotFoundError, ValueError):
        # Re-raise expected errors
        raise
    except Exception as e:
        # Log unexpected errors here if needed
        raise Exception(f"Face detection failed: {e!s}")


def extract_face_regions(
    image_path: str,
    face_locations: List[FaceLocation],
    align: bool = True,
    padding: float = 0.2,
) -> List[np.ndarray]:
    """
    Enhanced face region extraction with alignment and padding.

    Args:
        image_path (str): Path to the image file
        face_locations (List[FaceLocation]): List of detected face locations
        align (bool): Whether to align faces using landmarks
        padding (float): Padding around face as percentage of face size

    Returns:
        List[np.ndarray]: List of processed face region images
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    face_regions = []
    for face in face_locations:
        # Add padding
        pad_x = int(face.width * padding)
        pad_y = int(face.height * padding)

        x1 = max(0, face.x - pad_x)
        y1 = max(0, face.y - pad_y)
        x2 = min(image.shape[1], face.x + face.width + pad_x)
        y2 = min(image.shape[0], face.y + face.height + pad_y)

        region = image[y1:y2, x1:x2]

        # Align face if landmarks are available and alignment is requested
        if align and face.landmarks and face.angle is not None:
            center = (region.shape[1] // 2, region.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -face.angle, 1.0)
            region = cv2.warpAffine(
                region, rotation_matrix, (region.shape[1], region.shape[0])
            )

        face_regions.append(region)

    return face_regions
