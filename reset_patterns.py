"""
One-time script to archive the current trade journal and pattern memory,
then reset them to empty so the system starts rebuilding from scratch.

USE WHEN: historical data is known to be poisoned/invalid (e.g. bad initial
calibration, bugs, or insufficient sample deemed harmful) and you want a
fresh slate in demo mode.

Run: python reset_patterns.py --confirm

The script is intentionally NOT imported by the trading pipeline. It must be
invoked manually. Archived copies are written alongside the originals with
a `.archived_YYYYMMDD_HHMMSS.json` suffix so nothing is lost.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import JOURNAL_DIR, MEMORY_DIR


def main():
    if "--confirm" not in sys.argv:
        print("This will archive trade_journal.json and pattern_memory.json,")
        print("then reset them to empty. Re-run with --confirm to proceed.")
        print()
        print("  python reset_patterns.py --confirm")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    journal_file = JOURNAL_DIR / "trade_journal.json"
    memory_file = MEMORY_DIR / "pattern_memory.json"

    archived = []
    for src in (journal_file, memory_file):
        if src.exists():
            dst = src.with_name(f"{src.stem}.archived_{ts}.json")
            shutil.copy2(src, dst)
            archived.append(str(dst))
            print(f"Archived: {src} -> {dst}")
        else:
            print(f"Skipped (does not exist): {src}")

    # Reset to empty collections
    journal_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(journal_file, "w") as f:
        json.dump([], f)
    with open(memory_file, "w") as f:
        json.dump({}, f)

    print()
    print("Reset complete. Journal + pattern memory are now empty.")
    print("The system will rebuild from the next executed trade.")
    if archived:
        print(f"Archived {len(archived)} file(s). Keep them in case you need to restore.")


if __name__ == "__main__":
    main()
