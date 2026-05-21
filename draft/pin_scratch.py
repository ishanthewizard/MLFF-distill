from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime


ROOT = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application")


def refresh_access_time(path: Path) -> tuple[bool, float | None]:
    """
    Open the file for reading to refresh atime without altering contents.
    Returns (success, latest_atime_epoch).
    """
    try:
        with open(path, "rb") as fh:
            fh.read(1)  # minimal read to update atime
        # Ensure mtime unchanged (open can update atime already, but be explicit)
        stat = path.stat()
        os.utime(path, (time.time(), stat.st_mtime))
        return True, path.stat().st_atime
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Failed on {path}: {exc}")
        return False, None


def fmt_atime(epoch_seconds: float | None) -> str:
    if epoch_seconds is None:
        return "n/a"
    return datetime.fromtimestamp(epoch_seconds).isoformat(timespec="seconds")


def main() -> None:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    print(f"Found {len(files)} files under {ROOT}")
    ok = 0
    for fpath in files:
        ok_flag, atime = refresh_access_time(fpath)
        if ok_flag:
            ok += 1
            print(f"[touched] {fpath} atime={fmt_atime(atime)}")
    print(f"Refreshed access time for {ok}/{len(files)} files.")


if __name__ == "__main__":
    main()
