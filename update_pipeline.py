import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
STATUS_FILE = BASE_DIR / "model_status.json"

STEPS = [
    ("EPİAŞ 2026 verileri", "fetch_2026_data.py"),
    ("Hava durumu verileri", "fetch_weather_data.py"),
    ("Doğalgaz fiyatları", "fetch_gas_data.py"),
    ("Yan hizmetler", "fetch_ancillary_data.py"),
    ("Model veri seti", "data_processor.py"),
    ("Model eğitimi", "train_model.py"),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def tail(text, limit=4000):
    if not text:
        return ""
    return text[-limit:]


def data_end():
    data_path = BASE_DIR / "model_ready_data.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path, usecols=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"], utc=True).max().isoformat()


def read_metrics():
    metrics_path = BASE_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_status(status):
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(label, script):
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "label": label,
        "script": script,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def main():
    status = {
        "started_at": utc_now(),
        "finished_at": None,
        "ok": False,
        "data_end": None,
        "metrics": {},
        "steps": [],
    }
    write_status(status)

    for label, script in STEPS:
        result = run_step(label, script)
        status["steps"].append(result)
        write_status(status)
        if not result["ok"]:
            status["finished_at"] = utc_now()
            status["ok"] = False
            status["data_end"] = data_end()
            status["metrics"] = read_metrics()
            write_status(status)
            print(f"[-] Pipeline durdu: {label}")
            print(result["stderr_tail"] or result["stdout_tail"])
            return 1

    status["finished_at"] = utc_now()
    status["ok"] = True
    status["data_end"] = data_end()
    status["metrics"] = read_metrics()
    write_status(status)
    print("[+] Veri güncelleme ve model eğitimi tamamlandı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
