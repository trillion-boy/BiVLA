#!/usr/bin/env python3
"""Enumerate every checkable claim in the three documents, by error class.

WHY THIS EXISTS
---------------
`build_grid_report.py` regenerates the tables from `results/`, and comparing a
document against it catches one kind of mistake: a number typed wrong. The
correction log in `Report.md` 7.1 says that kind is 10 of 92. The other 82 are
invisible to that check -- they are sentences that are wrong while every number
in them is right, or numbers that disagree between two documents, or counts the
document makes about itself, or a script and a document sharing one bug.

Re-reading does not close those, because re-reading selects what to look at by
recall, and recall is not enumeration: whatever does not come to mind stays
unchecked, and nothing reports that it was skipped. This script replaces recall
with a list. Every check below prints a count, so "what is left" is a number
instead of a memory.

    python experiments/audit_claims.py            # all classes
    python experiments/audit_claims.py --class 3  # one class

Classes, numbered to match the type table in Report 7.1:
    1  cross-document disagreement   (same labelled value, two documents)
    2  self-referential counts       (the document counting itself)
    3  non-independent verification  (value hardcoded in a script AND in prose)
    4  universal claims              (one counterexample falsifies; needs reading)
    5  threshold wording             (">" vs ">=" and friends)

Classes 1-3 are decided here. Classes 4-5 are ENUMERATED here and decided by
reading -- but the list is complete, so coverage is countable.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DOCS = ["Report.md", "Overview.md", "RelatedWork.md"]
SCRIPTS = ["build_grid_report.py", "verify_overview_claims.py",
           "mechanism_move_near.py", "measure_foveation_roundtrip.py",
           "compare_runs.py", "make_logpolar_figure.py"]

# A number written any of the ways these documents write them.
NUM = re.compile(r"[+\-−±]?\d+(?:,\d{3})*(?:\.\d+)?")

# Words that make a sentence falsifiable by a single counterexample. These are
# exactly the sentences that were wrong while their numbers were right --
# "baseline is nowhere lower", "the only discrepancy", "all twelve settings".
UNIVERSAL = ["전부", "모두", "모조리", "어디서도", "유일", "항상", "언제나",
             "하나도", "예외 없", "전혀", "무조건", "반드시", "어느 쪽도",
             "아무", "어느 칸도", "하나뿐", "밖에 없", "뿐이다", "없다"]

# Comparison wording whose sense flips between > and >=. Every one of these has
# to be traced back to the line of code that produced the number.
THRESHOLD = ["이상", "이하", "초과", "미만", "넘는", "넘게", "넘어", "이내",
             "보다 크", "보다 작", "최소", "최대"]


def read(path):
    with open(os.path.join(_HERE, path), encoding="utf-8") as f:
        return f.read().split("\n")


def label_of(line: str, pos: int) -> str:
    """The words a number is attached to, normalised enough to match across docs.

    Cross-document comparison needs a key that survives rewording. Taking the
    nearest words is crude, but the failure mode it guards against is literal:
    the same phrase, kept in two files, updated in one.
    """
    left = re.sub(r"[*`>|#\[\]()]", " ", line[max(0, pos - 46):pos])
    words = [w for w in re.split(r"\s+", left) if w and not NUM.fullmatch(w)]
    return " ".join(words[-4:]).strip().lower()


def numbers(lines, prose_only=False):
    """-> [(lineno, raw_number, normalised_label, full_line)]

    `prose_only` drops markdown table rows. In a table the meaning of a number
    is fixed by its COLUMN, which a nearest-words label cannot see, so every
    multi-column row looks like a disagreement with every other. Tables are
    already checked cell-by-cell against `build_grid_report.py`; this pass is
    for the sentences, which nothing else checks.
    """
    out = []
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("<!--"):
            continue
        if prose_only and line.lstrip().startswith("|"):
            continue
        for m in NUM.finditer(line):
            out.append((i, m.group().replace("−", "-"),
                        label_of(line, m.start()), line))
    return out


# --------------------------------------------------------------------------
# Class 1: the same labelled value, disagreeing between two documents.
# This is the class that produced "baseline is nowhere lower" surviving in
# Overview for days after Report had already been corrected.
# --------------------------------------------------------------------------
def cross_document(doc_numbers) -> list:
    by_label = defaultdict(lambda: defaultdict(set))
    where = defaultdict(lambda: defaultdict(list))
    for doc, nums in doc_numbers.items():
        for ln, raw, label, line in nums:
            # A one-word label ("baseline", "우리") names too many different
            # quantities to compare across documents; two content words is the
            # minimum that identifies a single claim.
            if len(label) < 12 or len(label.split()) < 2:
                continue
            by_label[label][doc].add(raw)
            where[label][doc].append((ln, raw))
    hits = []
    for label, per_doc in by_label.items():
        if len(per_doc) < 2:
            continue
        values = set().union(*per_doc.values())
        if len(values) == 1:
            continue
        hits.append((label, {d: sorted(v) for d, v in per_doc.items()},
                     {d: where[label][d][:3] for d in per_doc}))
    return hits


# --------------------------------------------------------------------------
# Class 2: the document counting itself -- table rows, line totals, family
# sizes, "N papers". Every one of these drifts the moment the thing it counts
# changes, and nothing about the sentence looks wrong when it does.
# --------------------------------------------------------------------------
COUNT_PATTERNS = [
    (r"(\d+)\s*행", "rows"),
    (r"(\d+)\s*건", "items"),
    (r"~?([\d,]+)\s*줄", "lines"),
    (r"(\d+)\s*편", "papers"),
    (r"(\d+)\s*개(?:이?므로|다|이고|였|입니다|\b)", "count"),
    (r"(\d+)\s*번을?\s*쟀", "tests"),
    (r"검정은?\s*(\d+)", "tests"),
    (r"파일\s*(\d+)", "files"),
]


def log_region(doc, lines):
    """Line range of the correction log, which QUOTES superseded numbers.

    Every row in Report 7.1 names a value that used to be written and is now
    wrong. Counting those as live claims turns the log into a permanent source
    of false alarms, and a checker that cries wolf gets ignored -- which is the
    same failure as not having one.
    """
    if doc != "Report.md":
        return range(0, 0)
    starts = [i for i, l in enumerate(lines, 1) if l.startswith("## 7.1")]
    if not starts:
        return range(0, 0)
    s = starts[0]
    later = [i for i, l in enumerate(lines, 1)
             if i > s and (l.startswith("## 7.3") or l.startswith("# "))]
    return range(s, later[0] if later else len(lines) + 1)


def self_counts(doc, lines) -> list:
    out = []
    skip = log_region(doc, lines)
    for i, line in enumerate(lines, 1):
        if i in skip:
            continue
        for pat, kind in COUNT_PATTERNS:
            for m in re.finditer(pat, line):
                out.append((doc, i, kind, m.group(1), line.strip()[:150]))
    return out


def measured_counts() -> dict:
    """The facts the class-2 claims have to match. Computed, never typed."""
    facts = {}
    for d in DOCS:
        facts[f"lines:{d}"] = len(read(d))
    rep = read("Report.md")
    # 7.1 table two: the correction log. Counted the same way every time.
    try:
        s = next(i for i, l in enumerate(rep) if l.startswith("## 7.1"))
        e = next(i for i, l in enumerate(rep) if l.startswith("## 7.2"))
        seg = rep[s:e]
        i2 = next(i for i, l in enumerate(seg) if "움직인 것 ②" in l)
        rows, started = 0, False
        for l in seg[i2:]:
            if l.startswith("|") and not l.startswith("|---") \
                    and l.strip() != "| 무엇 | 무엇이 틀렸나 |":
                rows += 1
                started = True
            elif started and not l.startswith("|"):
                break
        facts["corrections"] = rows
        i1 = next(i for i, l in enumerate(seg) if "움직인 것 ①" in l)
        r1 = sum(1 for l in seg[i1:i2] if l.startswith("|")
                 and not l.startswith("|---")
                 and "| 무엇 | 전 | 후 | 이유 |" not in l)
        facts["new_measurements"] = r1
    except StopIteration:
        pass
    # The two multiple-comparison families, from the generator, not from prose.
    try:
        sys.path.insert(0, _HERE)
        import build_grid_report as B
        data = B.discover()
        cols = [c for c in [("OpenVLA", "Bridge"), ("OpenVLA", "Fractal"),
                            ("SpatialVLA", "Bridge"), ("SpatialVLA", "Fractal"),
                            ("UniVLA", "Bridge")]
                if any(k[0] == c[0] and k[1] == c[1] for k in data)]
        facts["mcnemar_family"] = B.grid_family(data, cols)
        n = 0
        backbones = sorted({b for b, _ in cols})
        for bb in backbones:
            if (bb, "Bridge") in cols and (bb, "Fractal") in cols:
                n += sum(1 for c in B.conditions_present(data)
                         if B.interaction(data, c, (bb, "Bridge"), (bb, "Fractal")))
        for bm in ("Bridge", "Fractal"):
            for i, b1 in enumerate(backbones):
                for b2 in backbones[i + 1:]:
                    if (b1, bm) in cols and (b2, bm) in cols:
                        n += sum(1 for c in B.conditions_present(data)
                                 if B.interaction(data, c, (b1, bm), (b2, bm)))
        facts["fisher_family"] = n
    except Exception as exc:                       # pragma: no cover
        facts["_generator_error"] = repr(exc)
    return facts


# --------------------------------------------------------------------------
# Class 3: a value that appears BOTH hardcoded in a script AND in the prose.
# Agreement between those two proves nothing -- they have one author and can
# share one mistake, which is exactly how "Fisher count is 36, matches" passed
# as a verification while both were wrong.
# --------------------------------------------------------------------------
def non_independent(doc_numbers) -> list:
    hard = defaultdict(list)
    for s in SCRIPTS:
        p = os.path.join(_HERE, s)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p, encoding="utf-8").read().split("\n"), 1):
            code = line.split("#")[0]
            if not code.strip() or code.lstrip().startswith(('"', "'")):
                continue
            for m in re.finditer(r"\b\d+\.\d+\b", code):
                v = m.group()
                if v in ("0.0", "1.0", "2.0", "0.5", "100.0", "0.05"):
                    continue
                hard[v].append((s, i, line.strip()[:110]))
    hits = []
    for doc, nums in doc_numbers.items():
        for ln, raw, label, line in nums:
            key = raw.lstrip("+-")
            if key in hard:
                hits.append((doc, ln, key, hard[key][0], line.strip()[:110]))
    return hits


# --------------------------------------------------------------------------
# Classes 4 and 5: enumerated, not decided. The point is a complete list.
# --------------------------------------------------------------------------
def flagged(doc, lines, words, prose_only=True) -> list:
    """Lines to READ, not lines that are wrong.

    Table rows are skipped: a table cell is checked against the generator, and
    the failure this list is for is a SENTENCE that outruns its evidence.
    """
    out = []
    skip = log_region(doc, lines)
    for i, line in enumerate(lines, 1):
        if not line.strip() or i in skip:
            continue
        if prose_only and line.lstrip().startswith("|"):
            continue
        if any(w in line for w in words) and NUM.search(line):
            out.append((doc, i, line.strip()[:170]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="only", type=int, default=0)
    ap.add_argument("--list", action="store_true",
                    help="print every enumerated line for classes 4 and 5")
    a = ap.parse_args()

    docs = {d: read(d) for d in DOCS}
    doc_numbers = {d: numbers(v) for d, v in docs.items()}
    prose_numbers = {d: numbers(v, prose_only=True) for d, v in docs.items()}
    total = sum(len(v) for v in doc_numbers.values())
    print(f"{total} numeric tokens across {len(DOCS)} documents "
          f"({', '.join(f'{d}:{len(doc_numbers[d])}' for d in DOCS)})\n")

    fail = 0

    if a.only in (0, 1):
        hits = cross_document(prose_numbers)
        print(f"[1] cross-document disagreement on a shared label: {len(hits)}")
        for label, per_doc, where in hits:
            print(f"    \"{label}\"")
            for d, vals in per_doc.items():
                loc = ", ".join(f"L{ln}" for ln, _ in where[d])
                print(f"        {d:16s} {vals}  ({loc})")
        fail += len(hits)
        print()

    if a.only in (0, 2):
        facts = measured_counts()
        print("[2] self-referential counts -- what the documents must match:")
        for k, v in sorted(facts.items()):
            print(f"      {k:20s} = {v}")
        claims = [c for d in DOCS for c in self_counts(d, docs[d])]
        print(f"    {len(claims)} count-shaped claims found in prose; the ones "
              f"naming the facts above must equal them.")
        # Each check is (fact, regex capturing the number IN ITS OWN phrasing).
        # The regex carries the cue, so a line that merely mentions a nearby
        # word cannot be mistaken for a claim about this fact.
        checks = [
            ("corrections", [r"②의\s*(\d+)\s*행", r"②\((\d+)행\)",
                             r"불일치는\s*(\d+)\s*건", r"정정 기록\s*(\d+)\s*건"]),
            ("mcnemar_family", [r"격자에서\s*\*?\*?(\d+)\s*번",
                                r"짝 검정은?\s*\*?\*?(\d+)\s*개",
                                r"격자\s*(\d+)\s*개 검정"]),
            ("fisher_family", [r"이 가족은\s*\*?\*?(\d+)\s*개",
                               r"Fisher\s*(\d+)\s*개",
                               r"보정\((\d+)개 검정"]),
            ("lines:Report.md", [r"`?Report\.md`?[^|]*~([\d,]+)줄"]),
            ("lines:RelatedWork.md", [r"`?RelatedWork\.md`?[^|]*~([\d,]+)줄"]),
        ]
        for key, pats in checks:
            want = facts.get(key)
            if want is None:
                continue
            for d in DOCS:
                skip = log_region(d, docs[d])
                for i, line in enumerate(docs[d], 1):
                    if i in skip:
                        continue
                    for pat in pats:
                        for m in re.finditer(pat, line):
                            got = int(m.group(1).replace(",", ""))
                            tol = 25 if key.startswith("lines:") else 0
                            if abs(got - want) > tol:
                                print(f"    ! {d}:{i} says {got} for {key}, "
                                      f"measured {want}")
                                print(f"        {line.strip()[:130]}")
                                fail += 1
        print()

    if a.only in (0, 3):
        hits = non_independent(doc_numbers)
        seen = {(d, v) for d, _, v, _, _ in hits}
        print(f"[3] values hardcoded in a script AND stated in prose: "
              f"{len(seen)} distinct (doc, value) pairs")
        print("    Agreement between these two is NOT verification -- one "
              "author, one possible shared mistake.")
        for d, ln, v, (s, sl, src), line in hits[:12]:
            print(f"      {v:>8s}  {d}:{ln}   <-  {s}:{sl}")
        if len(hits) > 12:
            print(f"      ... {len(hits) - 12} more")
        print()

    if a.only in (0, 4):
        rows = [r for d in DOCS for r in flagged(d, docs[d], UNIVERSAL)]
        print(f"[4] universal claims carrying a number: {len(rows)}")
        print("    One counterexample falsifies each. Numbers being right does "
              "not make these right -- this is the class re-reading misses.")
        for r in (rows if a.list else rows[:8]):
            print(f"      {r[0]}:{r[1]}  {r[2]}")
        if not a.list and len(rows) > 8:
            print(f"      ... {len(rows) - 8} more (--list for all)")
        print()

    if a.only in (0, 5):
        rows = [r for d in DOCS for r in flagged(d, docs[d], THRESHOLD)]
        print(f"[5] threshold wording near a number: {len(rows)}")
        print("    Each must be traced to the comparison operator that produced "
              "it. '2 이상' written over a `> 2` is not a rounding error.")
        for r in (rows if a.list else rows[:8]):
            print(f"      {r[0]}:{r[1]}  {r[2]}")
        if not a.list and len(rows) > 8:
            print(f"      ... {len(rows) - 8} more (--list for all)")
        print()

    print(f"decided classes (1-3) failures: {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
