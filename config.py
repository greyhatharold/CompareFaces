"""
Configuration management module for the application.

This module centralizes all configuration parameters and provides a clean interface
to access them throughout the application. It handles environment variables,
API keys, and other configuration constants.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import (
    Any,
    Dict,
)


# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """
    Configuration class that provides access to all application settings.

    This class follows the singleton pattern to ensure consistent configuration
    across the application. All configuration parameters are accessed as class
    attributes.
    """

    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    OPENAI_MODEL = "gpt-3.5-turbo"  # Default model
    OPENAI_TEMPERATURE = 0.7  # Controls randomness in responses

    # File Paths
    BASE_DIR = Path(__file__).resolve().parent
    LOG_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"

    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Application Settings
    MAX_RETRIES = 3  # Maximum number of API call retries
    TIMEOUT_SECONDS = 30  # API timeout duration
    SIMILARITY_THRESHOLD = 0.85  # Minimum similarity score for matching
    BATCH_SIZE = 100  # Default batch size for processing

    # Cache Settings
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # Time-to-live for cached items (in seconds)

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """
        Returns all configuration settings as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters
        """
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    @classmethod
    def validate_settings(cls) -> bool:
        """
        Validates that all required settings are properly configured.

        Returns:
            bool: True if all required settings are valid, False otherwise

        Raises:
            ValueError: If a required setting is missing or invalid
        """
        if cls.OPENAI_API_KEY == "your-api-key-here":
            raise ValueError("OpenAI API key not configured")

        # Ensure directories exist
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

        return True


# Create required directories on module import
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
