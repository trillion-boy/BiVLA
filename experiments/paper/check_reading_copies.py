#!/usr/bin/env python3
"""Does each reading copy still say what its .tex says?

RelatedWork.md fell behind relatedwork.tex three times. The third time it was
carrying a sentence whose meaning had been inverted in the .tex and corrected
there but not here, which is exactly the copy a co-author reads. This checks
the two cheaply.

Run:  python3 experiments/paper/check_reading_copies.py
Exit: 0 clean, 1 drift found.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rendered(tex: str) -> str:
    """Approximate what the .tex renders to, so it can be compared with prose."""
    t = re.sub(r"(?m)^%.*$", "", tex)
    t = re.sub(r"\\footnote\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", t)
    t = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", t)
    t = re.sub(r"\\(textbf|emph|texttt)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\section\{[^}]*\}", " ", t)
    t = t.replace("~", " ").replace("\\%", "%").replace("\\ ", " ")
    t = re.sub(r"\$[^$]*\$", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalise(s: str) -> str:
    s = s.replace("**", "").replace("*", "").replace("`", "")
    s = s.replace("\u2014", " ").replace("\u2013", " ").replace("\u2019", "'")
    s = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", s)      # markdown links
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", s.lower())).strip()


def clauses(text: str, min_words: int = 6):
    """Sentence-ish chunks long enough to be worth matching."""
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if len(part.split()) >= min_words:
            yield part


STOP = set(
    "a an the of to in on for by and or but is are was were be been it its this that "
    "these those we our us as at from with which what when than then so not no nor "
    "each their them they there here into over under about more most only also".split()
)


def content_words(s: str):
    return [w for w in normalise(s).split() if w not in STOP and len(w) > 2]


PAIRS = [("relatedwork.tex", "RelatedWork.md")]


def measure(tex: str) -> int:
    """Rendered word count, same formula used throughout the paper notes."""
    t = re.sub(r"(?m)^%.*$", "", tex)
    t = re.sub(r"\\section\{[^}]*\}", "", t)
    t = re.sub(r"\\cite\{[^}]*\}", " ", t)
    t = re.sub(r"\\ref\{[^}]*\}", " ", t)
    t = re.sub(r"\\(textbf|emph|texttt)\{([^}]*)\}", r"\2", t)
    t = t.replace("~", " ").replace("\\%", "%").replace("$\\times$", "x")
    return len(t.split())


def main() -> int:
    failures = []
    for tex_name, md_name in PAIRS:
        tex_path, md_path = HERE / tex_name, HERE / md_name
        if not tex_path.exists() or not md_path.exists():
            failures.append(f"{tex_name} or {md_name} is missing")
            continue

        tex_body = normalise(rendered(tex_path.read_text()))
        md_raw = md_path.read_text()
        # the co-author notes are not part of the section
        cut = md_raw.find("## Notes for the co-authors")
        md_body = normalise(md_raw[:cut] if cut > 0 else md_raw)

        missing = []
        for clause in clauses(rendered(tex_path.read_text())):
            words = content_words(clause)
            if len(words) < 4:
                continue
            present = sum(1 for w in words if w in md_body)
            # the .md spells citations out, so allow some slack, but a clause
            # whose wording changed will drop well below this
            if present / len(words) < 0.85:
                missing.append(f"{present}/{len(words)} words: {clause[:80]}")

        if missing:
            failures.append(
                f"{md_name} is behind {tex_name}. {len(missing)} clause(s) diverge:\n    "
                + "\n    ".join(missing)
            )

        # the .md header should quote the current word count
        wc = measure(tex_path.read_text())
        claimed = re.findall(r"(\d{3}) words of prose", md_raw)
        if claimed and abs(int(claimed[0]) - wc) > 5:
            failures.append(
                f"{md_name} says {claimed[0]} rendered words, {tex_name} measures ~{wc}"
            )

    if failures:
        print("DRIFT")
        for f in failures:
            print("  -", f)
        return 1
    print("reading copies match their .tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
