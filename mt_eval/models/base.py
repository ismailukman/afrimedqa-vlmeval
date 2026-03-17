from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMTModel(ABC):
    @abstractmethod
    def translate(self, messages: list[dict]) -> str:
        """Send a list of chat messages and return the model's text response."""
        ...
