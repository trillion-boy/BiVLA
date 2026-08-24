#!/usr/bin/env python3
"""Every check I know how to run on introduction.tex and relatedwork.tex.

Written after doing these piecemeal across several rounds and finding something
new each time. The point of one file is that the list stops growing silently:
whatever gets checked, gets checked here, and a clean run means the whole list
passed rather than whichever slice was on my mind.

Word count is deliberately NOT a check. It is reported as context and nothing
more. Whether a section is well written is decided by whether its claims are
sourced, its sentences parse, its terms are consistent and its argument closes,
none of which a word count measures.

The checks were themselves tested, by injecting a known defect of each kind and
confirming the audit reports it: unbalanced braces, unclosed math, an unescaped
percent both inside math and mid-sentence, a doubled word, an em-dash, a
semicolon, a prose colon, a number with no source, a malformed citation key, a
citation after a period, a citation without a non-breaking space, a footnote
marker before its punctuation, a sentence over 45 words, a sentence opening on
a bare connective, and a weasel word. Fourteen of fifteen were caught first
time. The percent test was not, and was structurally unable to be: it ran after
comment stripping, which eats everything following an unescaped percent, so it
could never see one. It now runs on the raw text and distinguishes a comment
line from a stray percent inside a sentence.

Run:  python3 experiments/paper/audit_sections.py
Exit: 0 clean, 1 findings.
"""
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SECTIONS = ["introduction.tex", "relatedwork.tex"]
SOURCE_DOCS = [
    "experiments/Report.md",
    "experiments/Overview.md",
    "experiments/RelatedWork.md",
    "experiments/paper/TableI_Cells.md",
    "experiments/paper/RelatedWork_Sources.md",
    "experiments/paper/Introduction.md",
    "experiments/paper/FiveModels_Read.md",
    "experiments/paper/EpisodeCounts.md",
    "experiments/paper/PerTaskRows.md",
    "experiments/paper/AxisClaim.md",
]

findings = []
notes = []


def finding(check, msg):
    findings.append(f"[{check}] {msg}")


def note(check, msg):
    notes.append(f"[{check}] {msg}")


def strip_comments(tex):
    return re.sub(r"(?m)(?<!\\)%.*$", "", tex)


def rendered(tex, drop_footnotes=False):
    t = strip_comments(tex)
    if drop_footnotes:
        t = re.sub(r"\\footnote\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", t)
    else:
        t = re.sub(r"\\footnote\{", " ", t)
    t = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", t)
    t = re.sub(r"\\(textbf|emph|texttt)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\section\{[^}]*\}", " ", t)
    t = t.replace("~", " ").replace("\\%", "%").replace("\\ ", " ")
    t = re.sub(r"\$([^$]*)\$", r"\1", t).replace("\\times", "x")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wordcount(tex):
    t = strip_comments(tex)
    t = re.sub(r"\\footnote\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", t)
    t = re.sub(r"\\section\{[^}]*\}", "", t)
    t = re.sub(r"\\cite\{[^}]*\}", "X", t)
    t = re.sub(r"\\ref\{[^}]*\}", "X", t)
    t = re.sub(r"\\(textbf|emph|texttt)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\$[^$]*\$", "N", t)
    return len(t.replace("~", " ").split())


# --------------------------------------------------------------- load
raw = {}
for name in SECTIONS:
    p = HERE / name
    if not p.exists():
        finding("files", f"{name} is missing")
        continue
    raw[name] = p.read_text()

corpus = ""
for d in SOURCE_DOCS:
    f = ROOT / d
    if f.exists():
        corpus += f.read_text(errors="replace")
corpus_n = corpus.replace("\u2212", "-").replace("\u2013", "-").replace("\u00d7", "x")


# =========================================================== A. mechanical
for name, tex in raw.items():
    body = strip_comments(tex)

    # A1 brace balance
    if tex.count("{") != tex.count("}"):
        finding("A1 braces", f"{name}: {tex.count('{')} open vs {tex.count('}')} close")

    # A2 math mode balance
    if body.count("$") % 2:
        finding("A2 math", f"{name}: odd number of $ ({body.count('$')})")

    # A3 unescaped specials that change meaning or break the build.
    # The % test must run on the RAW text: strip_comments() eats everything
    # after an unescaped %, so a check downstream of it can never see one.
    # Distinguish a real comment line from a stray % inside a sentence.
    for i, line in enumerate(tex.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue                      # a deliberate comment line
        code = re.sub(r"\\%", "", line)   # drop escaped percents
        j = code.find("%")
        if j >= 0 and code[:j].strip():    # text before it, so mid-sentence
            finding("A3 escaping", f"{name}:{i}: unescaped '%' comments out the rest of the line: {line.strip()[:60]!r}")

    # keys inside \\cite/\\ref/\\label are never typeset, so their underscores are fine
    scannable = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", body)
    for ch, why in [("&", "column separator"),
                    ("#", "macro parameter"),
                    ("_", "math subscript outside math mode")]:
        bad = []
        for m in re.finditer(re.escape(ch), scannable):
            k = m.start()
            if k and scannable[k - 1] == "\\":
                continue
            if ch == "_" and scannable.count("$", 0, k) % 2:
                continue          # inside math, legitimate
            bad.append(scannable[max(0, k - 30):k + 5].replace("\n", " "))
        if bad:
            finding("A3 escaping", f"{name}: unescaped {ch!r} ({why}): {bad[:3]}")

    # A4 line length
    long = [i + 1 for i, l in enumerate(tex.split("\n")) if len(l) > 79]
    if long:
        finding("A4 wrap", f"{name}: lines over 79 chars at {long}")

    # A5 citation should not follow terminal punctuation
    if re.search(r"[.,;:]\s*~?\\cite", body):
        finding("A5 cite-punct", f"{name}: a citation follows punctuation")

    # A6 non-breaking space before every cite and ref
    loose = re.findall(r"(?<![~{,\s])\s\\(cite|ref)\{", body)
    if loose:
        finding("A6 nbsp", f"{name}: {len(loose)} \\cite/\\ref without a preceding ~")

    # A7 footnote marker should follow the punctuation
    for m in re.finditer(r"(.)\\footnote\{", body):
        if m.group(1) not in ".,!?":
            finding("A7 footnote", f"{name}: marker not after punctuation near {body[max(0,m.start()-40):m.start()+12]!r}")

    # A8 doubled words
    for m in re.finditer(r"\b(\w+)\s+\1\b", rendered(tex), re.I):
        if m.group(1).lower() not in {"that", "had"}:
            finding("A8 doubled", f"{name}: '{m.group(0)}'")

    # A9 double spaces in source
    if re.search(r"[a-z]\.  +[A-Z]", body):
        note("A9 spacing", f"{name}: double space after a sentence, harmless in LaTeX")


# =========================================================== B. punctuation policy
for name, tex in raw.items():
    body = strip_comments(tex)
    b2 = re.sub(r"\\(ref|label)\{[^}]*\}", "X", body)
    counts = {
        "em-dash": body.count("---"),
        "en-dash": body.count("--") - body.count("---") * 2,
        "semicolon": b2.count(";"),
        "prose colon": len(re.findall(r"[a-z,\)\}]\s?:\s", b2)),
    }
    for k, v in counts.items():
        if v:
            finding("B punctuation", f"{name}: {v} {k}")


# =========================================================== C. numbers
for name, tex in raw.items():
    body = strip_comments(tex).replace("{,}", "").replace("--", "-")
    nums = sorted(set(re.findall(r"\d+\.\d+|\d{2,}", body)))
    missing = []
    for n in nums:
        alts = [n, n + "%", f"{int(n):,}" if n.isdigit() else n]
        if not any(a in corpus_n for a in alts):
            missing.append(n)
    if missing:
        finding("C1 provenance", f"{name}: numbers not found in source docs: {missing}")
    note("C1 provenance", f"{name}: {len(nums)} distinct numbers, all traced")


# =========================================================== D. citations
allkeys = set()
for name, tex in raw.items():
    keys = {k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", tex) for k in m.split(",")}
    allkeys |= keys
    empty = [m for m in re.findall(r"\\cite\{\s*\}", tex)]
    if empty:
        finding("D1 cite", f"{name}: empty \\cite{{}}")
    for k in keys:
        if not re.fullmatch(r"[a-z0-9_]+", k):
            finding("D2 cite-key", f"{name}: odd citation key {k!r}")

src = (ROOT / "experiments/paper/RelatedWork_Sources.md").read_text()
NAME = {
    "act": "ACT", "diffusionpolicy": "Diffusion Policy", "dqn": "DQN",
    "openvlaoft": "OpenVLA-OFT", "shortgpt": "ShortGPT", "gromov": "Gromov",
    "efficientvla": "EfficientVLA", "molevla": "MoLe-VLA", "fastv": "FastV",
    "sparsevlm": "SparseVLM", "tome": "ToMe", "vlacache": "VLA-Cache",
    "vlapruner": "VLA-Pruner", "flower": "FLOWER", "smolvla": "SmolVLA",
    "turbovla": "TurboVLA", "simplerenv": "SimplerEnv", "libero": "LIBERO",
    "vlaeval": "vla-eval", "specprune": "SpecPrune-VLA", "vlaiap": "VLA-IAP",
    "gazereg": "Gaze-Reg", "lookfocusact": "Look Focus Act",
    "effvlasurvey1": "CAS survey", "effvlasurvey2": "2510.24795",
    "schwartz": "Schwartz", "traver": "Traver",
    "bagoftricks_cnn": "bagoftricks_cnn", "bagoftricks_llm": "bagoftricks_llm",
    "starvla": "StarVLA", "openvla": "OpenVLA", "spatialvla": "SpatialVLA",
    "univla": "UniVLA",
}
unmapped = [k for k in allkeys if k not in NAME]
if unmapped:
    finding("D3 provenance", f"citation keys with no provenance entry: {unmapped}")
unrecorded = [k for k in allkeys if k in NAME and NAME[k] not in src and k not in
              {"openvla", "spatialvla", "univla"}]
if unrecorded:
    finding("D3 provenance", f"keys absent from RelatedWork_Sources.md: {unrecorded}")
note("D3 provenance", f"{len(allkeys)} distinct citation keys, all accounted for")


# =========================================================== E. undefined refs
labels = set()
for tex in raw.values():
    labels |= set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set()
for tex in raw.values():
    refs |= {r for m in re.findall(r"\\ref\{([^}]*)\}", tex) for r in [m]}
dangling = sorted(refs - labels)
if dangling:
    note("E refs", f"\\ref targets defined elsewhere in the paper: {dangling}")


# =========================================================== F. language
for name, tex in raw.items():
    text = rendered(tex, drop_footnotes=True)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 2]

    # F1 sentence length
    lens = [len(s.split()) for s in sents]
    over = [s[:70] for s in sents if len(s.split()) > 45]
    if over:
        finding("F1 length", f"{name}: sentence over 45 words: {over}")
    note("F1 length", f"{name}: {len(lens)} sentences, mean {sum(lens)/len(lens):.1f}, max {max(lens)}")

    # F2 sentence-initial connectives whose antecedent may have been lost
    risky = [s[:60] for s in sents if s.split()[0] in
             {"However", "Which", "So", "And", "But", "Whereas"}]
    if risky:
        finding("F2 antecedent", f"{name}: sentence opens on a bare connective: {risky}")

    # F3 dangling participle at sentence start
    for s in sents:
        m = re.match(r"^(\w+ing|Run|Given|Based|Using|Sweeping|Held)\b", s)
        if m and "," in s[:60]:
            head = s[s.index(",") + 1:].strip().split()
            subject = " ".join(head[:2]).lower() if head else ""
            if not subject.startswith(("we ", "we,")):
                finding("F3 participle", f"{name}: participle without a matching subject: '{s[:70]}'")

    # F4 word repetition inside one sentence
    for s in sents:
        c = Counter(w.lower() for w in re.findall(r"[a-z]{5,}", s.lower()))
        rep = [w for w, n in c.items() if n >= 3]
        if rep:
            note("F4 repetition", f"{name}: {rep} repeated in one sentence")


# =========================================================== G. cross-section
if len(raw) == 2:
    a = re.sub(r"[^a-z0-9 ]+", " ", rendered(raw["introduction.tex"]).lower()).split()
    b = re.sub(r"[^a-z0-9 ]+", " ", rendered(raw["relatedwork.tex"]).lower()).split()
    ng = lambda w, n: {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    for n in (10, 8):
        dup = ng(a, n) & ng(b, n)
        if dup:
            finding("G duplication", f"{n}-word phrase in both sections: {sorted(dup)[:3]}")
    small = sorted(ng(a, 6) & ng(b, 6))
    if small:
        note("G duplication", f"6-word overlap, judged intentional: {small}")


# =========================================================== H. structure
intro = raw.get("introduction.tex", "")
rw = raw.get("relatedwork.tex", "")

if intro:
    n_contrib = len(re.findall(r"\\textbf\{\d\)", intro))
    if n_contrib < 3:
        finding("H1 contributions", f"only {n_contrib} numbered contributions")
    note("H1 contributions", f"{n_contrib} contributions in numbered run-in form")
    if "itemize" in intro or "enumerate" in intro:
        finding("H1 contributions", "uses a list environment; Bag of Tricks uses a paragraph")

if rw:
    body = strip_comments(rw)
    runins = len(re.findall(r"\\textbf\{[A-Z][^}]*\.\}", body))
    if runins != 3:
        finding("H2 runins", f"Related Work has {runins} bold run-in headings, expected 3")
    if "tabular" in body:
        finding("H2 table", "Related Work contains a table; the model paper has none")
    blocks = [p for p in re.split(r"\n\s*\n", strip_comments(rw)) if p.strip()]
    note("H2 structure", f"{runins} run-in paragraphs, {len(blocks)} blocks, 0 tables")

for name, tex in raw.items():
    note("H3 context", f"{name}: {wordcount(tex)} words (context only, not a criterion)")


# =========================================================== I. promises
if intro:
    if "records released" in rendered(intro):
        d = ROOT / "results"
        if not d.exists() or not any(d.iterdir()):
            finding("I promises", "the draft promises per-episode records but results/ is empty")
        else:
            note("I promises", f"per-episode release backed by {len(list(d.iterdir()))} result directories")



# =========================================================== J. terminology
# One idea should have one name. Drift between synonyms is the commonest way a
# reader loses the thread across two sections.
# Reviewed 2026-08-22, when the question came up of promoting every entry here
# to a finding. Three of the four original entries were counting the wrong
# thing, so promoting them unchanged would have produced three false failures
# out of four, and a checker that cries wolf gets ignored. That is how this
# section became decorative in the first place: it only ever noted, so nobody
# acted on it.
#
# The split is now by whether drift is a DEFECT or a STYLE CHOICE, and whether
# the detection is sound enough to fail a build on.

# Drift here is a defect and the match is exact, so these fail.
# Rule: no variant may be a substring of another, and no variant may match
# inside a different word or a quotation of someone else's terminology.
NAMING_MUST_MATCH = [
    ("what the campaign total counts beyond the grid",
     ["control runs and sweeps", "control and diagnostic runs", "sweeps and re-runs"]),
    ("the hyphenation of the intervention family",
     ["training-free", "training free"]),
    ("the noun for the eligible-layer knob",
     ["candidate window", "eligibility window", "eligible-layer window", "candidate set"]),
]

# Drift here is legitimate, so these only report. Each carries the reason it
# cannot be a finding, because otherwise someone will promote it again.
TERM_SETS = [
    # "candidate window" is the noun, "eligible" is the adjective for membership
    # in it. Both belong. The noun-only check above is the one that can fail.
    ("the eligible-layer knob", ["candidate window", "eligibility", "eligible"]),
    # "matched episodes" is how we test; "per-episode records" is what we
    # release. Different things, not two names for one.
    ("the unit of pairing", ["matched episode", "episode-level", "per-episode"]),
    # "compute" is our measurement and "FLOPs" appears only where we quote the
    # claim template and VLA-Cache. Counting them as rivals counts other
    # people's vocabulary as our own. The count also over-reports: "compute"
    # matches inside "compute saved" and inside the verb "computed".
    ("the thing being spent", ["compute saved", "compute", "FLOPs"]),
]

joined = " ".join(rendered(t, drop_footnotes=False) for t in raw.values()).lower()
for label, variants in TERM_SETS:
    used = {v: joined.count(v.lower()) for v in variants if joined.count(v.lower())}
    if len(used) > 1:
        note("J terminology", f"{label}: {used}")
for label, variants in NAMING_MUST_MATCH:
    used = {v: joined.count(v.lower()) for v in variants if joined.count(v.lower())}
    if len(used) > 1:
        finding("J naming", f"{label}: two names for one set, {used}")
    elif used:
        note("J naming", f"{label}: one name throughout, {used}")

# One name carrying two ideas, which is the mirror of the check above and could
# never be caught by it. `axes` is DEFINED in Related Work paragraph 1 as the
# resource a method spends. Two later sentences used it for the dimensions of
# the backbone x benchmark grid instead, and one of those sat two sentences
# after "which spend resources our axes do not", so a single paragraph carried
# both senses. The grid dimensions are "factors" now. Flag any recurrence.
OVERLOADED = [
    ("axes", ["backbone", "benchmark", "grid", "crossing"], 60,
     "'axes' is the resource axis. Call a grid dimension a factor."),
]
for term, collide, window, advice in OVERLOADED:
    for m in re.finditer(r"\b" + term + r"\b", joined):
        near = joined[max(0, m.start() - window):m.end() + window]
        hits = [w for w in collide if w in near]
        if hits:
            finding("J overload",
                    f"'{term}' used within {window} chars of {hits}. {advice}")

# =========================================================== K. hedging and vagueness
WEASEL = ["very", "quite", "somewhat", "arguably", "clearly", "obviously",
          "of course", "it is well known", "significantly better", "a number of",
          "several studies", "many works", "in recent years", "state-of-the-art",
          "it should be noted", "notably better", "fairly", "relatively"]
for name, tex in raw.items():
    text = rendered(tex, drop_footnotes=False).lower()
    # word boundaries, so "every" does not match "very"
    hits = [w for w in WEASEL if re.search(r"\b" + re.escape(w) + r"\b", text)]
    # "rather" is fine in "rather than"; only a bare adverbial use is vague
    if re.search(r"\brather\b(?!\s+than)", text):
        hits.append("rather (adverbial)")
    if hits:
        finding("K vagueness", f"{name}: {hits}")


# =========================================================== L. unsupported quantifiers
# A quantified claim about other people's work needs a citation in the same sentence.
for name, tex in raw.items():
    body = strip_comments(tex)
    for s in re.split(r"(?<=[.!?])\s+", body):
        if re.search(r"\b(all|none|every|no|never|always|only)\b", s, re.I):
            hostile = re.search(r"\b(papers|methods|work|studies|literature|tables)\b", s, re.I)
            if hostile and "\\cite{" not in s and "\\ref{" not in s:
                finding("L quantifier", f"{name}: universal claim about others with no citation or pointer: {rendered(s)[:80]!r}")

# =========================================================== M. passive voice density
for name, tex in raw.items():
    text = rendered(tex, drop_footnotes=True)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]
    passive = [s for s in sents if re.search(r"\b(is|are|was|were|be|been|being)\s+\w+(ed|en)\b", s)]
    frac = len(passive) / max(1, len(sents))
    if frac > 0.45:
        finding("M passive", f"{name}: {frac:.0%} of sentences passive")
    note("M passive", f"{name}: {frac:.0%} passive, {len(passive)}/{len(sents)}")

# =========================================================== N. argument closure
# Every result announced in a bold run-in must be picked up by a contribution,
# and every contribution must rest on something stated.
if intro:
    r_heads = re.findall(r"\\textbf\{([A-Z][^}]*?\.)\}", strip_comments(intro))
    r_heads = [h for h in r_heads if not re.match(r"^\d\)", h) and h != "Contributions."]
    contribs = re.findall(r"\\textbf\{\d\)[^}]*\}([^\\]*)", strip_comments(intro))
    note("N closure", f"{len(r_heads)} result headings, {len(contribs)} contributions")
    if len(r_heads) < 3:
        finding("N closure", f"only {len(r_heads)} result headings found")

# =========================================================== O. orphan concepts
# A term used once and never explained or reused is usually a leftover.
for name, tex in raw.items():
    text = rendered(tex, drop_footnotes=True)
    caps = re.findall(r"\b([A-Z][a-zA-Z0-9-]{3,})\b", text)
    once = [c for c, n in Counter(caps).items() if n == 1 and c not in
            {"Running", "Such", "Applying", "That", "Three", "What", "Inside", "Success",
             "Across", "Sweeping", "Measured", "Every", "Because", "Contributions",
             "Evidence", "Four", "Layer", "VLA", "Whether", "Notably", "Results",
             "Papers", "Which", "Their", "This", "These", "There", "When", "Neither"}]
    if once:
        note("O orphans", f"{name}: capitalised terms used once: {sorted(once)[:14]}")

# =========================================================== report
print("=" * 68)
if findings:
    print(f"FINDINGS ({len(findings)})")
    for f in findings:
        print("  !", f)
else:
    print("NO FINDINGS")
print("-" * 68)
for n in notes:
    print("  .", n)
print("=" * 68)
sys.exit(1 if findings else 0)
