#!/usr/bin/env python3
"""Compare a .bib against what introduction.tex and relatedwork.tex actually cite.

Written after a hand-assembled Overleaf bibliography turned out to be missing
eighteen entries, every one of them a Related Work citation. Nothing catches
that from this side: the checks here read the repository copy, and the gap is
in the copy pasted into Overleaf. BibTeX does warn, but the warning is one line
per key buried in a log most people never open, and the PDF just prints [?].

Usage:  python3 check_bib.py [path/to/their.bib]

With no argument it checks paper/main.bib, which should always be clean.

Exit: 0 every cited key is present, 1 otherwise.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = ["introduction.tex", "relatedwork.tex"]


def cited():
    out = {}
    for name in SECTIONS:
        text = re.sub(r"(?m)(?<!\\)%.*$", "", (HERE / name).read_text())
        for group in re.findall(r"\\cite\{([^}]*)\}", text, flags=re.S):
            for key in (k.strip() for k in group.split(",")):
                if key:
                    out.setdefault(key, []).append(name)
    return out


def main(argv):
    target = Path(argv[1]) if len(argv) > 1 else HERE / "main.bib"
    have = {m.group(1) for m in
            re.finditer(r"(?m)^@[a-zA-Z]+\{([^,]+),", target.read_text())}
    want = cited()
    missing = sorted(set(want) - have)
    unused = sorted(have - set(want))

    print(f"{target.name}: {len(have)} entries, {len(want)} keys cited")
    if missing:
        print(f"\nMISSING {len(missing)} -- each prints [?] in the PDF:")
        for k in missing:
            print(f"   {k:<20} cited in {', '.join(sorted(set(want[k])))}")
    if unused:
        print(f"\nunused ({len(unused)}) -- harmless, BibTeX skips them:")
        print("   " + ", ".join(unused))
    if not missing and not unused:
        print("exact match")
    elif not missing:
        print("\nevery cited key resolves")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
