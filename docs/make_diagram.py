import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12.5, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 140)
ax.axis("off")

C_CTRL = "#2d6cdf"
C_CTRL_BG = "#e8f0fe"
C_UNI = "#0f9d58"
C_UNI_BG = "#e6f4ea"
C_SPA = "#db4437"
C_SPA_BG = "#fce8e6"
C_FROZEN = "#5f6368"
C_FROZEN_BG = "#eceff1"
C_NEUTRAL = "#444444"

def box(x, y, w, h, text, fc, ec, fs=11, weight="normal", tc="#202124", style="round,pad=0.3"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.8,
                       edgecolor=ec, facecolor=fc, mutation_scale=10)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, wrap=True)

def arrow(x1, y1, x2, y2, color=C_NEUTRAL, lw=2.0, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
                        linewidth=lw, color=color, linestyle=ls,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# ── Title ──
ax.text(50, 137, "BiVLA  —  Shared Task-Phase Controller", ha="center",
        fontsize=17, fontweight="bold", color="#202124")
ax.text(50, 133.5, "training-free test-time control of two frozen VLA backbones",
        ha="center", fontsize=11, color="#5f6368", style="italic")

# ── Instruction ──
box(33, 124, 34, 6, 'Instruction\n"put the carrot on the plate"',
    "#fff8e1", "#f4b400", fs=11, weight="bold")
arrow(50, 124, 50, 118.5, C_CTRL, lw=2.4)

# ── Shared Controller ──
box(8, 92, 84, 26, "", C_CTRL_BG, C_CTRL, style="round,pad=0.4")
ax.text(50, 115, "SHARED CONTROLLER", ha="center", fontsize=13,
        fontweight="bold", color=C_CTRL)
ax.text(50, 112.2, "shared_unified_policy.py  ·  independent module",
        ha="center", fontsize=9.5, color=C_CTRL, style="italic")

box(11, 104.5, 37, 5.5, "(1) extract_source_dest_nouns()\nsource / target", "white", C_CTRL, fs=9.5)
box(52, 104.5, 37, 5.5, "(2) infer_task_archetype()\nstack / thin_tool / container / planar", "white", C_CTRL, fs=9)
box(11, 97.5, 37, 5.5, "(3) shared_task_policy_profile()\nfocus on/off · weights · delay · gates", "white", C_CTRL, fs=9)
box(52, 97.5, 37, 5.5, "(4) phase_compact_focus_gate()\nper-step on/off  (hysteresis)", "white", C_CTRL, fs=9.5)

# ── Split arrows (same profile to both) ──
arrow(30, 92, 26, 84, C_CTRL, lw=2.4)
arrow(70, 92, 74, 84, C_CTRL, lw=2.4)
ax.text(50, 88.5, "same profile applied to each backbone", ha="center",
        fontsize=9.5, color=C_CTRL, style="italic")

# ── UniVLA column ──
ux = 7; uw = 40
box(ux, 80, uw, 4.5, "UniVLA path   ·   adaptive_sparse_vla/", C_UNI_BG, C_UNI, fs=10.5, weight="bold", tc=C_UNI)
box(ux+2, 72, uw-4, 5, "univla_gate_*  +  compact_focus_gate", "white", C_UNI, fs=9.5)
box(ux+2, 64, uw-4, 5, "episode routing\n_layout_patch_gate()", "white", C_UNI, fs=9.5)
box(ux+2, 55.5, uw-4, 5.5, "AutoGaze sparse frontend\n(background blur) + layer pruning", "white", C_UNI, fs=9.5)
box(ux+2, 47, uw-4, 5, "frozen  UniVLA  (Emu3)", C_FROZEN_BG, C_FROZEN, fs=10.5, weight="bold", tc=C_FROZEN)
arrow(ux+uw/2, 80, ux+uw/2, 77.2, C_UNI)
arrow(ux+uw/2, 72, ux+uw/2, 69.2, C_UNI)
arrow(ux+uw/2, 64, ux+uw/2, 61.2, C_UNI)
arrow(ux+uw/2, 55.5, ux+uw/2, 52.2, C_UNI)
arrow(ux+uw/2, 47, ux+uw/2, 43, C_NEUTRAL)
box(ux+8, 38.5, uw-16, 4.5, "action", "#f1f3f4", C_NEUTRAL, fs=10)

# ── SpatialVLA column ──
sx = 53; sw = 40
box(sx, 80, sw, 4.5, "SpatialVLA path   ·   latent_saccade/", C_SPA_BG, C_SPA, fs=10.5, weight="bold", tc=C_SPA)
box(sx+2, 72, sw-4, 5, "_apply_task_policy_profile()\n(overwrite inference weights)", "white", C_SPA, fs=9.5)
box(sx+2, 64, sw-4, 5, "spatial_focus_enabled gate", "white", C_SPA, fs=9.5)
box(sx+2, 55.5, sw-4, 5.5, "latent foveation hook\n(reweight visual tokens)", "white", C_SPA, fs=9.5)
box(sx+2, 47, sw-4, 5, "frozen  SpatialVLA  (PaliGemma2)", C_FROZEN_BG, C_FROZEN, fs=10, weight="bold", tc=C_FROZEN)
arrow(sx+sw/2, 80, sx+sw/2, 77.2, C_SPA)
arrow(sx+sw/2, 72, sx+sw/2, 69.2, C_SPA)
arrow(sx+sw/2, 64, sx+sw/2, 61.2, C_SPA)
arrow(sx+sw/2, 55.5, sx+sw/2, 52.2, C_SPA)
arrow(sx+sw/2, 47, sx+sw/2, 43, C_NEUTRAL)
box(sx+8, 38.5, sw-16, 4.5, "action", "#f1f3f4", C_NEUTRAL, fs=10)

# ── Footnote ──
box(10, 28.5, 80, 7, "Shared decision logic, separate mechanisms.\n"
    "Same controller -> UniVLA blurs background + routes;  SpatialVLA reweights tokens.\n"
    "Both backbones stay frozen and run independently (no cross-talk).",
    "#f8f9fa", "#9aa0a6", fs=9.5, tc="#3c4043")

# legend
ax.add_patch(FancyBboxPatch((10, 22.5), 3, 2.2, boxstyle="round,pad=0.1", fc=C_CTRL_BG, ec=C_CTRL))
ax.text(14, 23.6, "controller", fontsize=8.5, va="center", color="#3c4043")
ax.add_patch(FancyBboxPatch((30, 22.5), 3, 2.2, boxstyle="round,pad=0.1", fc=C_UNI_BG, ec=C_UNI))
ax.text(34, 23.6, "UniVLA", fontsize=8.5, va="center", color="#3c4043")
ax.add_patch(FancyBboxPatch((48, 22.5), 3, 2.2, boxstyle="round,pad=0.1", fc=C_SPA_BG, ec=C_SPA))
ax.text(52, 23.6, "SpatialVLA", fontsize=8.5, va="center", color="#3c4043")
ax.add_patch(FancyBboxPatch((68, 22.5), 3, 2.2, boxstyle="round,pad=0.1", fc=C_FROZEN_BG, ec=C_FROZEN))
ax.text(72, 23.6, "frozen model", fontsize=8.5, va="center", color="#3c4043")

plt.tight_layout()
import os
os.makedirs("/home/user/BiVLA/docs", exist_ok=True)
fig.savefig("/home/user/BiVLA/docs/architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved docs/architecture.png")
