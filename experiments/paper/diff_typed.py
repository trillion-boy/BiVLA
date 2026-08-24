#!/usr/bin/env python3
"""Compare a hand-typed copy of a section against the file it came from.

Written after a retyped Related Work came back with eight differences, seven of
them single characters (than/then twice, inference=time, unchhanged, rised,
traing-free, LIVERO) and one a whole sentence dropped, the one carrying the
"crossing cell empty" claim and its two citations.

None of the other checks in this directory can catch that class. They read the
repository copy, and the defect lives in the copy pasted somewhere else. The
one-character substitutions in particular survive proofreading because they are
all real English words or plausible spellings.

Usage:  python3 diff_typed.py <typed-file> [section.tex]

With one argument the section is guessed from the \\section{...} line in the
typed file. Comments and line wrapping are ignored on both sides, so only real
differences are reported.

Exit: 0 identical, 1 differences found.
"""
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def words(text):
    return " ".join(re.sub(r"(?m)(?<!\\)%.*$", "", text).split()).split(" ")


def guess(typed):
    m = re.search(r"\\section\{([^}]*)\}", typed)
    if not m:
        return None
    slug = re.sub(r"[^a-z]", "", m.group(1).lower())
    for p in sorted(HERE.glob("*.tex")):
        if re.sub(r"[^a-z]", "", p.stem.lower()) == slug:
            return p
    return None


def main(argv):
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    typed_path = Path(argv[1])
    typed = typed_path.read_text()
    src = Path(argv[2]) if len(argv) == 3 else guess(typed)
    if src is None or not src.exists():
        print(f"cannot find the source section for {typed_path}. "
              f"Pass it as the second argument.")
        return 2

    a, b = words(src.read_text()), words(typed)
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False)
           .get_opcodes() if o[0] != "equal"]
    print(f"{src.name}  vs  {typed_path.name}")
    if not ops:
        print("identical")
        return 0
    for n, (tag, i1, i2, j1, j2) in enumerate(ops, 1):
        was, now = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        # A long deletion is a lost sentence, which is the dangerous case: a
        # missing claim reads as fluent prose, while a missing word does not.
        flag = "  <-- WHOLE CLAUSE MISSING" if tag == "delete" and i2 - i1 > 6 else ""
        print(f"\n[{n}] {tag}{flag}")
        print(f"   source: {was or '(nothing)'}")
        print(f"   typed : {now or '(nothing)'}")
    print(f"\n{len(ops)} difference(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
