"""
Configuration settings for Conjunctural library.
"""

from pathlib import Path
import os

# Default cache directory
DEFAULT_CACHE_DIR = Path(os.environ.get("CONJUNCTURAL_CACHE_DIR", Path.home() / ".cache" / "conjunctural"))

# Network settings
REQUEST_TIMEOUT = int(os.environ.get("CONJUNCTURAL_TIMEOUT", "20"))
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Conjunctural/0.1"

# Brazilian convention
ANNUAL_BUSINESS_DAYS = 252

# Ensure cache directory exists
def get_cache_dir() -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR
