#!/usr/bin/env python3
"""Defect injection for the J-naming checks in audit_sections.py.

Written because the last attempt to self-test this section was broken and did
not say so. It used a flat string replace against a file wrapped at 78 columns,
so the phrase it looked for never appeared on one line, no defect was ever
injected, and the "NO FINDINGS" it printed meant nothing. Every phrase here is
matched with a whitespace-flexible regex instead, and the harness FAILS if the
injection did not change the file, so a silent miss cannot recur.

Each case injects one variant of a naming pair into a single occurrence, leaving
the other occurrence alone, which is exactly the drift the check is meant to
catch. The originals are restored in a finally block.

Run:  python3 experiments/paper/selftest_naming.py
Exit: 0 all cases behaved, 1 otherwise.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit_sections.py"

# (file, phrase to find, replacement, expected substring of the finding)
CASES = [
    ("introduction.tex", "control runs and sweeps", "sweeps and re-runs",
     "what the campaign total counts beyond the grid"),
    ("introduction.tex", "training-free", "training free",
     "the hyphenation of the intervention family"),
    ("introduction.tex", "candidate window", "eligibility window",
     "the noun for the eligible-layer knob"),
]


def flexible(phrase):
    """A regex for the phrase that survives the file's line wrapping."""
    return re.compile(r"\s+".join(re.escape(w) for w in phrase.split()))


def run_audit():
    p = subprocess.run([sys.executable, str(AUDIT)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    rc, out = run_audit()
    if rc != 0:
        print("ABORT: the audit already reports findings, so an injected one "
              "would prove nothing. Fix the real findings first.")
        print(out)
        return 1

    ok = True
    for fname, phrase, replacement, expect in CASES:
        path = HERE / fname
        original = path.read_text()
        pat = flexible(phrase)
        try:
            injected, n = pat.subn(replacement, original, count=1)
            if n != 1:
                print(f"FAIL {fname}: could not inject {phrase!r}. The phrase "
                      f"is gone from the file, or the wrapping defeated the "
                      f"regex. This is the failure mode the old harness hid.")
                ok = False
                continue
            path.write_text(injected)
            rc, out = run_audit()
            if rc == 0:
                print(f"FAIL {phrase!r} -> {replacement!r}: audit stayed clean")
                ok = False
            elif expect not in out:
                print(f"FAIL {phrase!r} -> {replacement!r}: audit failed but "
                      f"not on this check")
                ok = False
            else:
                line = next(l for l in out.splitlines() if expect in l)
                print(f"ok   {phrase!r} -> {replacement!r}")
                print(f"       {line.strip()}")
        finally:
            path.write_text(original)

    rc, _ = run_audit()
    if rc != 0:
        print("FAIL: the files did not restore cleanly")
        ok = False
    else:
        print("ok   originals restored, audit clean again")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
