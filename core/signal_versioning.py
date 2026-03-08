"""
Signal Versioning — Tracks which model version generated each signal.
Critical for A/B testing models and debugging regressions.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import MODELS_DIR, LOGS_DIR


class SignalVersioner:
    """
    Tags every signal with metadata about what generated it.
    Allows you to compare performance across model versions.
    """

    VERSION_LOG = LOGS_DIR / "signal_versions.json"

    def __init__(self):
        self.current_version = self._get_current_version()
        self.versions = self._load_log()

    def tag_signal(self, signal: dict, market: str) -> dict:
        """
        Add version metadata to a trading signal.

        Adds:
        - model_version: hash of current model files
        - timestamp: when signal was generated
        - model_files: which files are currently loaded
        """
        signal["version"] = {
            "model_version": self.current_version,
            "timestamp": datetime.now().isoformat(),
            "market": market,
            "model_files": self._get_model_files(market),
        }
        return signal

    def record_signal_outcome(self, signal_version: str, outcome: dict):
        """Record the outcome of a versioned signal."""
        entry = {
            "version": signal_version,
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
        }
        self.versions.append(entry)
        self._save_log()

    def get_version_performance(self) -> dict:
        """Compare performance across model versions."""
        if not self.versions:
            return {"message": "No version history yet"}

        from collections import defaultdict
        stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0})

        for entry in self.versions:
            version = entry.get("version", "unknown")
            outcome = entry.get("outcome", {})
            pnl = outcome.get("pnl", 0)
            stats[version]["total_pnl"] += pnl
            if pnl > 0:
                stats[version]["wins"] += 1
            else:
                stats[version]["losses"] += 1

        result = {}
        for version, data in stats.items():
            total = data["wins"] + data["losses"]
            result[version] = {
                "trades": total,
                "win_rate": round(data["wins"] / total, 3) if total > 0 else 0,
                "total_pnl": round(data["total_pnl"], 2),
            }

        return result

    def _get_current_version(self) -> str:
        """Generate a version hash from current model files."""
        model_files = sorted(MODELS_DIR.glob("*.pkl")) + sorted(MODELS_DIR.glob("*.pt"))
        if not model_files:
            return "v0_no_models"

        hasher = hashlib.md5()
        for f in model_files:
            hasher.update(f.name.encode())
            hasher.update(str(f.stat().st_mtime).encode())
        return f"v_{hasher.hexdigest()[:8]}"

    def _get_model_files(self, market: str) -> list:
        """List model files for a specific market."""
        files = []
        for pattern in [f"lstm_{market}.*", f"xgboost_{market}.*"]:
            files.extend([f.name for f in MODELS_DIR.glob(pattern)])
        return files

    def _load_log(self) -> list:
        if self.VERSION_LOG.exists():
            with open(self.VERSION_LOG) as f:
                return json.load(f)
        return []

    def _save_log(self):
        self.VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(self.VERSION_LOG, "w") as f:
            json.dump(self.versions, f, indent=2, default=str)
