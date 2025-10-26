"""Simple validator: parse markdown headers into JSON and validate against schemas.
This script is intentionally small and dependency-light; it uses `jsonschema` for validation.
"""
import sys
import re
import json
from pathlib import Path

from jsonschema import validate


def parse_md(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else ""
    sections = {}
    cur = None
    buf = []
    for ln in lines[1:]:
        m = re.match(r"^##+\s+(.*)$", ln)
        if m:
            if cur:
                sections[cur] = "\n".join(buf).strip()
            cur = m.group(1).strip()
            buf = []
        else:
            if cur:
                buf.append(ln)
    if cur:
        sections[cur] = "\n".join(buf).strip()
    return {"title": title, "sections": sections}


def main():
    root = Path(".")
    docs = [("STRATEGY.md", "VALIDATION/schemas/strategy.schema.json"),
            ("PRINCIPLES.md", "VALIDATION/schemas/principles.schema.json"),
            ("DEPLOYMENT.md", "VALIDATION/schemas/deployment.schema.json")]

    ok = True
    for doc, schema in docs:
        p = root / doc
        if not p.exists():
            print(f"MISSING: {doc}")
            ok = False
            continue
        obj = parse_md(p)
        schema_obj = json.loads((root / schema).read_text(encoding="utf-8"))
        try:
            # map to the schema shapes
            if doc == "PRINCIPLES.md":
                mapped = {"title": obj["title"], "items": list(obj["sections"].keys())}
            elif doc == "DEPLOYMENT.md":
                mapped = {"title": obj["title"], "commands": [obj["sections"].get("Quick-start (development)", "")]}
            else:
                mapped = {"title": obj["title"], "sections": obj["sections"]}
            validate(instance=mapped, schema=schema_obj)
            print(f"OK: {doc}")
        except Exception as e:
            print(f"SCHEMA FAIL: {doc}: {e}")
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
