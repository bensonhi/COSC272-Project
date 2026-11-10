"""Shared paths, label order and logger. No model dependencies."""

import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "LiquidAI/d1-3B"
LABELS = ("NEI", "SUPPORT", "CONTRADICT")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
LOGGER = logging.getLogger("d1_scifact")
