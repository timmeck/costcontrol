"""Central configuration for CostControl."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "costcontrol.db"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic" if ANTHROPIC_API_KEY else "ollama")
DEFAULT_MODEL = os.getenv(
    "COSTCONTROL_MODEL",
    "claude-sonnet-4-6" if LLM_PROVIDER == "anthropic" else OLLAMA_MODEL,
)

COSTCONTROL_PORT = int(os.getenv("COSTCONTROL_PORT", "8600"))
COSTCONTROL_API_KEY = os.getenv("COSTCONTROL_API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
