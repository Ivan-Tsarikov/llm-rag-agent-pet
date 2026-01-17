from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    index_dir = Path("data/index")
    faiss_file = index_dir / "faiss.index"
    chunks_file = index_dir / "chunks.jsonl"

    if not faiss_file.exists() or not chunks_file.exists():
        print("index missing -> building", flush=True)
        subprocess.run(["python", "-m", "scripts.build_index"], check=True)
    else:
        print("index present -> skip", flush=True)

    subprocess.run(
        [
            "uvicorn",
            "src.app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
