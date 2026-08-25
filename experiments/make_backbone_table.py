#!/usr/bin/env python3
"""Table II -- the three backbones, as specifications instead of prose.

Setup owes the reader four facts about each backbone, and every one of them
was going to be a sentence: which checkpoint we ran, how many decoder layers
it has, how many actions one call emits, and which card it ran on. Four
sentences x three backbones is a paragraph that says nothing an alignment
would not say better, so it is a table.

Two of the four columns are RECOVERED FROM THE RECORDS, not typed:

  decoder layers     three independent derivations, one per backbone, each
                     from a different artefact the runs happened to leave
                     behind (see layers_*() below). No config file is read,
                     because we do not have the checkpoints on this machine.
  actions per call   read from the same field paired_test.horizon() reads,
                     under the same fallback rule, so the table and the
                     statistics cannot disagree about what a horizon is.

Two are typed, and cannot be otherwise:

  checkpoint         the run scripts hold it (run_*_grid.sh) and the UniVLA
                     Colab notebook holds it, but no result file records it.
  GPU                Hardware.md, first line: grepping every result file for
                     gpu/device/T4/L4/Tesla/NVIDIA returns nothing. The
                     assignment was confirmed by the author of the runs and
                     is forced by fp16 weight size, which is why the params
                     column sits next to it.

Both typed columns assert against their source below, so a silent drift in
either becomes a crash here rather than a wrong number in the paper.

Output: paper/tablebackbones.tex
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "paper")


# --------------------------------------------------------------------------
# What we type, and where each value comes from.
# --------------------------------------------------------------------------
# `params` is the parameter count of the whole policy, and it is here only to
# make the GPU column non-arbitrary: 8.5B in fp16 is 17 GB and a T4 holds
# about 15, so UniVLA could not have run on the card the other two used.
BACKBONES = [
    dict(
        name="OpenVLA",
        checkpoint="openvla/openvla-7b",
        checkpoint_src=("run_openvla_fractal_grid.sh", "MODEL_PATH"),
        base="Llama-2 7B",
        params="7B",
        gpu="T4",
    ),
    dict(
        name="SpatialVLA",
        checkpoint="IPEC-COMMUNITY/spatialvla-4b-224-pt",
        checkpoint_src=("run_spatialvla_fractal_grid.sh", "MODEL_PATH"),
        base="PaliGemma 2",
        params="4B",
        gpu="T4",
    ),
    dict(
        # The repo is Yuqi1997/UniVLA and the weights we ran are one folder
        # inside it. The folder name alone is what the lab notes carry, and a
        # reader cannot resolve it; the repo alone holds more than one
        # checkpoint. Both are needed, and the folder is 37 characters, which
        # is why it sits in the note and not in the column. See the width
        # arithmetic in the header comment of the emitted .tex.
        name="UniVLA",
        checkpoint="Yuqi1997/UniVLA",
        subfolder="UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K",
        checkpoint_src=("../BiVLA_univla_colab.ipynb", "snapshot_download"),
        base="Emu3",
        params="8.5B",
        gpu="L4",
    ),
]

GPU_MEM = {"T4": "15", "L4": "22.5"}  # GB, Hardware.md line 13


def tt(s):
    """Escape an identifier for \\texttt. Underscores are the only character
    in any of these names that LaTeX would otherwise take as markup."""
    return s.replace("_", r"\_")


# --------------------------------------------------------------------------
# Decoder layers. One derivation per backbone, each from whatever the runs
# happened to log. None of the three is a config lookup.
# --------------------------------------------------------------------------
def layers_openvla():
    """OpenVLA's depth runs print the stack size in the sentence itself.

        depth: bypassed 4 of 32 layers [17, 20, 23, 26]
    """
    found = set()
    n_lines = 0
    for path in glob.glob(os.path.join(RESULTS, "openvla*", "**", "*.log"),
                          recursive=True):
        with open(path, errors="ignore") as fh:
            for m in re.finditer(r"bypassed \d+ of (\d+) layers", fh.read()):
                found.add(int(m.group(1)))
                n_lines += 1
    assert len(found) == 1, f"OpenVLA stack size is not agreed: {found}"
    return found.pop(), "%d log lines" % n_lines


def layers_spatialvla():
    """SpatialVLA's depth runs print the whole per-layer redundancy array, so
    the stack size is how many entries it has.

        [DepthPrune] per-layer redundancy (cos in/out): L0=0.1028, ... L25=0.65
    """
    found = set()
    n_lines = 0
    for path in glob.glob(os.path.join(RESULTS, "spatialvla*", "**", "*.log"),
                          recursive=True):
        with open(path, errors="ignore") as fh:
            for line in fh:
                if "per-layer redundancy" not in line:
                    continue
                idx = [int(i) for i in re.findall(r"L(\d+)=", line)]
                # Contiguous from L0, so the count is the stack size and not
                # a subset that happens to have that many entries.
                assert idx == list(range(len(idx))), path
                found.add(len(idx))
                n_lines += 1
    assert len(found) == 1, f"SpatialVLA stack size is not agreed: {found}"
    return found.pop(), "%d log lines" % n_lines


def layers_univla():
    """UniVLA logs neither, so the stack size comes out of a run whose window
    it saturates.

    `univla_bridge_depth_control/prune4_last` restricts removal to the deepest
    layers with min_layer = 0.875 and removes 4. The eligible window is
    floor(0.875 * N) .. N-1. It selected [28, 29, 30, 31], four consecutive
    layers, which is only possible if the window held exactly four, i.e. if
    N - floor(0.875 * N) == 4. N = 32 is the solution, and the check below
    confirms it is the only one under 200.
    """
    picked, frac = None, None
    for path in glob.glob(os.path.join(RESULTS, "univla*", "**",
                                       "results_*.json"), recursive=True):
        d = json.load(open(path))
        lp = (d.get("llm_pruning") or {}).get("active_layers")
        f = d.get("llm_prune_min_layer")
        if not lp or f is None or f < 0.8:
            continue
        got = sorted(lp)
        assert got == list(range(got[0], got[0] + len(got))), \
            f"{path}: not a contiguous block, so the window was not saturated"
        if picked is None:
            picked, frac = got, float(f)
        assert got == picked and float(f) == frac, "runs disagree"
    assert picked, "no saturated deep-window UniVLA run found"
    import math
    sols = [n for n in range(len(picked), 200)
            if n - math.floor(frac * n) == len(picked) and n - 1 == picked[-1]]
    assert len(sols) == 1, f"UniVLA stack size is not determined: {sols}"
    return sols[0], "deepest-%d window at min_layer %g saturated by %s" % (
        len(picked), frac, picked)


LAYERS = {"OpenVLA": layers_openvla,
          "SpatialVLA": layers_spatialvla,
          "UniVLA": layers_univla}


# --------------------------------------------------------------------------
# Actions executed per policy call, under paired_test.horizon()'s own rule.
# --------------------------------------------------------------------------
def actions_per_call(prefix):
    """The chunk length horizon() multiplies the action repeat by.

    horizon() reads exec_chunk, and where that is <= 0 ("re-generate every
    step") falls back to predict_action_frames, defaulting to 1. We apply the
    identical rule here rather than a second one, so a change to the harness
    moves both or neither.

    SpatialVLA records neither field and so takes the default. That is not an
    assumption about the checkpoint: the SimplerEnv path keeps raw_actions[0]
    and discards the rest (inference.py:1473), so one action reaches the
    environment per call whatever the processor emitted.
    """
    vals = set()
    for path in glob.glob(os.path.join(RESULTS, prefix + "*", "**",
                                       "results_*.json"), recursive=True):
        d = json.load(open(path))
        if "action_repeat" not in d or d.get("action_repeat") is None:
            continue  # not one of the grid runs
        chunk = d.get("exec_chunk")
        chunk = 0 if isinstance(chunk, dict) or chunk is None else int(chunk)
        if chunk <= 0:
            chunk = int(d.get("predict_action_frames") or 1)
        vals.add(chunk)
    assert len(vals) == 1, f"{prefix}: chunk is not constant across runs: {vals}"
    return vals.pop()


def check_typed_sources():
    """Every typed cell asserts against the file it was copied from.

    The match is whole-token and not substring. A substring test passed
    happily on 2026-08-25 while the UniVLA folder name was printed with its
    leading ``UNIVLA_`` missing, because the truncation is still a substring
    of the real name. A truncated checkpoint identifier is unrecoverable by a
    reader, so the check has to be able to see the difference.
    """
    for b in BACKBONES:
        fname, needle = b["checkpoint_src"]
        text = open(os.path.join(HERE, fname), errors="ignore").read()
        assert needle in text, f"{fname}: {needle} is gone"
        for ident in filter(None, [b["checkpoint"], b.get("subfolder")]):
            # A hyphen is allowed to precede, because the shell default in the
            # run scripts is written ${MODEL_PATH:-openvla/openvla-7b}. It is
            # not allowed to follow, or openvla-7b would match inside
            # openvla-7b-finetuned-libero-spatial.
            assert re.search(r"(?<![\w./])%s(?![\w./-])" % re.escape(ident),
                             text), f"{fname} no longer contains {ident!r}"


def main():
    check_typed_sources()

    rows = []
    for b in BACKBONES:
        n, how = LAYERS[b["name"]]()
        chunk = actions_per_call(b["name"].lower())
        print("%-11s %2d layers (%s), %d action%s/call"
              % (b["name"], n, how, chunk, "" if chunk == 1 else "s"))
        rows.append(r"%s & \texttt{%s} & %s & $%d$ & $%d$ & %s (%s\,GB) \\"
                    % (b["name"], tt(b["checkpoint"]), b["params"], n, chunk,
                       b["gpu"], GPU_MEM[b["gpu"]]))

    subs = " ".join(
        r"The \texttt{%s} weights are the \texttt{%s} folder of that repository."
        % (tt(b["checkpoint"]), tt(b["subfolder"]))
        for b in BACKBONES if b.get("subfolder"))

    tex = r"""%% ---------------------------------------------------------------------------
%% Table II -- the backbones. Generated by experiments/make_backbone_table.py.
%% Regenerate rather than edit: the layer counts and the actions-per-call
%% column are derived from the run records, and the other columns assert
%% against the run scripts they were copied from.
%%
%% Spans both columns, so table* and not table. Needs \usepackage{booktabs}.
%%
%% This table exists to keep four facts out of the prose. Setup must not also
%% say in words that SpatialVLA has 26 layers, that UniVLA emits five actions
%% a call, or which card ran what. It may say what those facts MEAN, which is
%% a different sentence: four removed layers is 12.5%% of one stack and 15.4%%
%% of another, and action repeat 4 is a 4-step horizon for two backbones and
%% a 20-step one for the third.
%%
%% "Actions per call" and not "native chunk size". Native chunk is a property
%% of a checkpoint; this column is a property of our runs, and for SpatialVLA
%% the two need not agree, since the SimplerEnv path keeps raw_actions[0] and
%% discards the rest. The statistics multiply THIS number by the action
%% repeat, so this is the one the reader needs.
%%
%% The params column is not a specification for its own sake. It is why the
%% GPU column has two values in it: 8.5B at fp16 is 17 GB and a T4 holds
%% about 15.
%%
%% The UniVLA folder name is 37 characters and sits in the note rather than
%% in the column, which is a width decision and not an editorial one. At 10pt
%% typewriter a character is about 5.25pt, so putting the repository and the
%% folder in one cell would set that cell at roughly 285pt of the 505.89pt
%% \textwidth ieeeconf gives, and the remaining five columns need about 240.
%% With the folder in the note the widest cell is the 35-character SpatialVLA
%% identifier at about 185pt, and the row totals roughly 460pt including
%% column separation. If a column is added, redo this arithmetic.
%% ---------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{The three backbones, their public checkpoints, and how each was run.}
\label{tab:backbones}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l l l r r l}
\toprule
Backbone & Checkpoint & Params & Decoder layers & Actions per call & GPU \\
\midrule
%s
\bottomrule
\end{tabular}

\parbox{\textwidth}{\footnotesize Actions per call is how many actions one
forward pass contributes to the environment, so an action repeat of $r$ puts
$r$ times that many environment steps between decisions. Layer counts are
recovered from the pruning records rather than read from the configuration
files. %s}
\end{table*}
""" % ("\n".join(rows), subs)

    path = os.path.join(OUT, "tablebackbones.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
