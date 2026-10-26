"""Fetch only the pinned, unmodified SO101 model assets (Apache-2.0)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REVISION = "cda5236f98383f6fbaa4820c6b18dba892cd087f"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/robotstudio_so101/"
DEST = ROOT / "third_party/robotstudio_so101"


def fetch(name):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                content = urllib.request.urlopen(BASE + name, timeout=40).read()
                path.write_bytes(content)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1)
    print(name, path.stat().st_size, flush=True)
    return name, hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    fetch("so101.xml")
    tree = ET.parse(DEST / "so101.xml")
    names = ["assets/" + m.attrib["file"] for m in tree.findall(".//asset/mesh")]
    names += ["README.md", "LICENSE", "CHANGELOG.md"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(fetch, names))
    hashes["so101.xml"] = hashlib.sha256((DEST / "so101.xml").read_bytes()).hexdigest()
    manifest = {"repository": "https://github.com/google-deepmind/mujoco_menagerie",
                "commit": REVISION, "subdirectory": "robotstudio_so101", "sha256": hashes}
    (DEST / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
