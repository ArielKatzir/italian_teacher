"""Error correction interface for Italian language learning."""

from abc import ABC, abstractmethod
from typing import Dict, List


class ErrorCorrectionInterface(ABC):
    """Interface for error correction systems."""

    @abstractmethod
    def correct_error(self, text: str, error_type: str) -> Dict:
        """Correct an error in Italian text."""

    @abstractmethod
    def identify_errors(self, text: str) -> List[Dict]:
        """Identify errors in Italian text."""
