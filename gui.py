"""
GUI Module for Face Comparison Application

This module implements the graphical user interface using PySide6, providing
a modern, visually appealing way to compare face images. It follows the Single
Responsibility Principle by focusing solely on GUI presentation and user interaction.

Dependencies:
    - PySide6
    - PIL (Python Imaging Library)
    - asyncio
    - Other application modules (face_detection, feature_extraction, etc.)
"""

import logging
import sys
from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import asyncio
from chatgpt_api_integration import create_client
import cv2
from face_detection import (
    detect_faces,
    extract_face_regions,
)
from feature_extraction import (
    DeepFeatureExtractor,
    extract_features,
)
import numpy as np
from pathlib import Path
import queue
from similarity_analysis import (
    compare_features,
    CosineSimilarity,
    EuclideanSimilarity,
)
import threading
from typing import Optional


# Configure logging
logger = logging.getLogger(__name__)


class AsyncHelper:
    """Helper class to handle async operations with Qt."""

    def __init__(self):
        self.async_loop = asyncio.new_event_loop()
        self.async_queue = queue.Queue()

        # Start async thread
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()

    def _run_async_loop(self):
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()

    def run_async(self, coro, callback):
        async def wrapped():
            try:
                result = await coro
                self.async_queue.put((callback, result))
            except Exception as e:
                self.async_queue.put((callback, e))

        asyncio.run_coroutine_threadsafe(wrapped(), self.async_loop)

    def process_queue(self):
        try:
            while True:
                callback, result = self.async_queue.get_nowait()
                callback(result)
        except queue.Empty:
            pass


class ImageDropFrame(QFrame):
    """Modern-looking frame for image display with drag & drop support."""

    def __init__(self, title: str):
        super().__init__()
        self.setAcceptDrops(True)
        self.image_path: Optional[Path] = None
        self.face_region = None

        # Setup UI
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #000000;
            }
        """
        )

        # Create image display label with fixed size and styling
        self.image_label = QLabel()
        self.image_label.setFixedSize(300, 300)  # Larger image size
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: #f5f6fa;
                border: 2px dashed #b2bec3;
                border-radius: 10px;
                padding: 5px;
                color: #000000;
            }
        """
        )

        # Add placeholder text
        self.image_label.setText("Drop image here\nor click Browse")

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setStyleSheet(
            """
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """
        )
        self.browse_button.clicked.connect(self.browse_image)

        # Layout setup
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.browse_button)
        self.layout.setAlignment(Qt.AlignCenter)

        # Set frame styling
        self.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border-radius: 15px;
                padding: 10px;
            }
        """
        )

    def browse_image(self):
        """Open file dialog to select an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.load_image(Path(file_path))

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.toLocalFile().lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                self.load_image(Path(file_path))

    def load_image(self, path: Path) -> bool:
        """
        Load and display an image, detecting faces.

        Args:
            path (Path): Path to image file

        Returns:
            bool: True if image was loaded successfully
        """
        try:
            # Validate path
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")

            # Detect faces
            faces = detect_faces(str(path))
            if not faces:
                QMessageBox.warning(self, "No Faces", "No faces detected in image")
                return False

            # Extract first face region
            regions = extract_face_regions(str(path), faces)
            if not regions:
                return False

            # Ensure the face region is C-contiguous and in the correct format
            self.face_region = np.ascontiguousarray(regions[0])

            # Convert BGR to RGB and ensure proper memory layout
            rgb_image = cv2.cvtColor(self.face_region, cv2.COLOR_BGR2RGB)
            height, width = rgb_image.shape[:2]
            bytes_per_line = width * 3

            # Create QImage from the numpy array
            image = QImage(
                rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Update the label with the new image
            self.image_label.setPixmap(scaled_pixmap)
            self.image_path = path
            return True

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {e!s}")
            logger.error(f"Image load error: {e!s}")
            return False


class SimilarityMeter(QFrame):
    """Modern animated meter for displaying similarity scores."""

    def __init__(self, label: str):
        super().__init__()
        self.value = 0
        self.target_value = 0

        # Setup UI
        self.layout = QVBoxLayout(self)
        self.label = QLabel(label)
        self.label.setStyleSheet("font-weight: bold; color: #000000;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: none;
                background-color: #f5f6fa;
                border-radius: 5px;
                text-align: center;
                color: #000000;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:1 #27ae60);
            }
        """
        )

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)

    def animate_to(self, value: float):
        self.target_value = value * 100  # Convert to percentage
        self.animation = QPropertyAnimation(self, b"current_value")
        self.animation.setDuration(1000)
        self.animation.setStartValue(self.value)
        self.animation.setEndValue(self.target_value)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()

    @Property(float)
    def current_value(self):
        return self.value

    @current_value.setter
    def current_value(self, value):
        self.value = value
        self.progress_bar.setValue(int(value))
        self._update_color(value / 100)  # Convert back to 0-1 range

    def _update_color(self, value: float):
        # Create color gradient based on value
        if value < 0.65:
            color = "#e74c3c"  # Red
        elif value < 0.80:
            color = "#f1c40f"  # Yellow
        else:
            color = "#2ecc71"  # Green

        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                background-color: #f5f6fa;
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                border-radius: 5px;
                background: {color};
            }}
        """
        )


class ResultsPanel(QFrame):
    """Panel for displaying comparison results with visual feedback."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 10px;
            }
        """
        )

        # Create main layout
        self.main_layout = QVBoxLayout(self)

        # Create meters with labels
        self.cosine_meter = SimilarityMeter("Face Similarity Score")
        self.euclidean_meter = SimilarityMeter("Feature Match Score")

        # Add interpretation guide
        self.guide_text = QTextEdit()
        self.guide_text.setReadOnly(True)
        self.guide_text.setMaximumHeight(100)
        self.guide_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f5f6fa;
                border: 1px solid #b2bec3;
                border-radius: 5px;
                padding: 5px;
                color: #000000;
            }
        """
        )
        self.guide_text.setText(
            "Score Guide:\n"
            "< 0.65: Different people\n"
            "0.65-0.80: Possible relation\n"
            "0.80-0.90: Strong similarity (likely related)\n"
            "> 0.90: Very high similarity (same person or identical twin)"
        )

        # Analysis text
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f5f6fa;
                border: 1px solid #b2bec3;
                border-radius: 5px;
                padding: 5px;
                color: #000000;
            }
        """
        )

        # Add widgets to layout
        self.main_layout.addWidget(self.cosine_meter)
        self.main_layout.addWidget(self.euclidean_meter)
        self.main_layout.addWidget(self.guide_text)
        self.main_layout.addWidget(self.analysis_text)

    def update_results(self, cosine_score: float, euclidean_score: float):
        """Update the visual results with animation."""
        self.cosine_meter.animate_to(cosine_score)
        self.euclidean_meter.animate_to(euclidean_score)

    def set_analysis(self, text: str):
        """Update the analysis text."""
        self.analysis_text.setText(text)


class ComparisonApp:
    """
    Main application class for face comparison GUI.

    This class coordinates the overall GUI layout and handles the interaction
    between different components and modules.
    """

    def __init__(self):
        """Initialize the face comparison application."""
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("Face Comparison")
        self.window.setGeometry(100, 100, 1200, 800)  # Wider window

        # Create central widget and main layout
        self.central_widget = QWidget()
        self.window.setCentralWidget(self.central_widget)

        # Set window and central widget styling
        self.window.setStyleSheet(
            """
            QMainWindow, QWidget#centralWidget {
                background-color: #000000;
            }
        """
        )
        self.central_widget.setObjectName("centralWidget")

        # Create async helper
        self.async_helper = AsyncHelper()

        # Create ChatGPT client
        try:
            self.chatgpt = create_client()
        except ValueError as e:
            QMessageBox.warning(self.window, "API Configuration", str(e))
            self.chatgpt = None

        self._create_widgets()
        self._create_layout()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Image panels
        self.left_panel = ImageDropFrame("First Image")
        self.right_panel = ImageDropFrame("Second Image")

        # Comparison controls
        self.controls_frame = QFrame()
        self.compare_button = QPushButton("Compare Faces")
        self.compare_button.setStyleSheet(
            """
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                border: 2px solid #ffffff;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:disabled {
                background-color: #666666;
                border: 2px solid #999999;
            }
        """
        )
        self.compare_button.clicked.connect(self.compare_faces)

        # Results display
        self.results_panel = ResultsPanel(self.central_widget)

    def _create_layout(self):
        """Arrange widgets in the window with faces on left, results on right."""
        # Create main horizontal layout for the central widget
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)  # Add margins
        main_layout.setSpacing(20)  # Add spacing between widgets

        # Left side container for images and compare button
        left_container = QFrame()
        left_container.setStyleSheet(
            """
            QFrame {
                background-color: transparent;
            }
        """
        )
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(20)

        # Add stretch at top to push content to vertical center
        left_layout.addStretch()

        # Controls - now above the image panels
        self.controls_frame.setStyleSheet("background-color: transparent;")
        controls_layout = QHBoxLayout(self.controls_frame)
        controls_layout.addStretch()
        controls_layout.addWidget(self.compare_button)
        controls_layout.addStretch()

        # Add controls to left layout first
        left_layout.addWidget(self.controls_frame)

        # Image panels in a horizontal layout
        image_frame = QFrame()
        image_frame.setStyleSheet("background-color: transparent;")
        image_layout = QHBoxLayout(image_frame)
        image_layout.setSpacing(20)
        image_layout.addWidget(self.left_panel)
        image_layout.addWidget(self.right_panel)

        # Add image frame after controls
        left_layout.addWidget(image_frame)

        # Add stretch at bottom to maintain vertical centering
        left_layout.addStretch()

        # Add containers to main layout
        main_layout.addWidget(left_container, 2)  # Images take up more space
        main_layout.addWidget(
            self.results_panel, 1
        )  # Results panel takes up less space

        # Set up timer for processing async queue
        self.timer = QTimer()
        self.timer.timeout.connect(self.async_helper.process_queue)
        self.timer.start(100)  # Check every 100ms

    def compare_faces(self):
        """
        Compare the two loaded face images.

        This method coordinates the comparison process:
        1. Extract features from both faces
        2. Compute similarity scores
        3. Get ChatGPT analysis
        4. Update the GUI with results
        """
        if not (
            self.left_panel.face_region is not None
            and self.right_panel.face_region is not None
        ):
            QMessageBox.warning(
                self.window,
                "Missing Images",
                "Please load two images with detected faces first.",
            )
            return

        try:
            # Show loading message while downloading models if needed
            self.compare_button.setEnabled(False)
            self.compare_button.setText("Initializing models...")
            self.window.update()

            # Extract features using deep learning by default
            extractor = DeepFeatureExtractor()

            # Reset button
            self.compare_button.setText("Compare Faces")
            self.compare_button.setEnabled(True)

            # Extract features and collect metadata
            features1 = extract_features(self.left_panel.face_region, extractor)
            features2 = extract_features(self.right_panel.face_region, extractor)

            # Compute similarities
            cosine_score = compare_features(
                features1.vector, features2.vector, CosineSimilarity()
            )
            euclidean_score = compare_features(
                features1.vector, features2.vector, EuclideanSimilarity()
            )

            # Update results with animation
            self.results_panel.update_results(cosine_score.score, euclidean_score.score)

            # Get ChatGPT analysis if available
            if self.chatgpt:
                # Combine metadata from face detection and feature extraction
                face1_metadata = {
                    **features1.metadata,
                    **getattr(self.left_panel.face_region, "metadata", {}),
                }
                face2_metadata = {
                    **features2.metadata,
                    **getattr(self.right_panel.face_region, "metadata", {}),
                }

                self.async_helper.run_async(
                    self.chatgpt.compare_feature_vectors(
                        features1.vector.tolist(),
                        features2.vector.tolist(),
                        cosine_score.score,
                        "facial",
                        face1_metadata=face1_metadata,
                        face2_metadata=face2_metadata,
                    ),
                    self._update_analysis,
                )

        except Exception as e:
            QMessageBox.warning(self.window, "Error", f"Comparison failed: {e!s}")
            logger.error(f"Comparison error: {e!s}")

    def _update_analysis(self, result):
        """
        Update the analysis text widget with ChatGPT response.

        Args:
            result: ChatGPTResponse or Exception
        """
        if isinstance(result, Exception):
            self.results_panel.set_analysis(f"Analysis failed: {result!s}")
        else:
            self.results_panel.set_analysis(result.content)

    def run(self):
        """Start the application."""
        self.window.show()
        return self.app.exec()


def main() -> None:
    """Application entry point."""
    app = ComparisonApp()
    app.run()


if __name__ == "__main__":
    main()
