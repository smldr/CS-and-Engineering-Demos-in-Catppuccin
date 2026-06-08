import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    overlay0, surface0, surface1, base, text_col,
)

# ── Geometry constants ────────────────────────────────────────────────────────
LIMIT  = 2.6
_theta = np.linspace(0, 2 * np.pi, 200)
CIRCLE = np.stack([np.cos(_theta), np.sin(_theta)])
GRIDS  = np.linspace(-2.5, 2.5, 11)
TLINE  = np.linspace(-2.5, 2.5, 80)
I2     = np.array([1.0, 0.0])
J2     = np.array([0.0, 1.0])

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9))
fig.suptitle("Matrix Transpose  vs  Inverse — Geometric Comparison (2D)",
             fontsize=13, fontweight="bold", color=text_col)
fig.patch.set_facecolor(base)

ax_A  = fig.add_axes([0.02, 0.12, 0.23, 0.80])
ax_AT = fig.add_axes([0.27, 0.12, 0.23, 0.80])
ax_Ai = fig.add_axes([0.52, 0.12, 0.23, 0.80])

# ── Axis styling ──────────────────────────────────────────────────────────────
def style_ax(ax, title, title_col, subtitle=""):
    ax.set_facecolor(base)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.tick_params(colors=overlay0, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(overlay0)
    ax.axhline(0, color=overlay0, lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color=overlay0, lw=0.6, ls="--", alpha=0.4)
    ax.grid(True, color=overlay0, lw=0.3, alpha=0.2)
    ax.set_xlabel("x", color=overlay0, fontsize=8)
    ax.set_ylabel("y", color=overlay0, fontsize=8)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, color=title_col, fontsize=10, fontweight="bold", pad=6)

# ── Draw a transformed view ───────────────────────────────────────────────────
def draw_transform(ax, M, title, title_col, subtitle=""):
    ax.cla()
    style_ax(ax, title, title_col, subtitle)

    # Transformed coordinate grid — teal for horizontal lines, peach for vertical
    for v in GRIDS:
        ax.plot(*(M @ np.vstack([TLINE, np.full_like(TLINE, v)])),
                color=teal,  lw=0.4, alpha=0.22)
        ax.plot(*(M @ np.vstack([np.full_like(TLINE, v), TLINE])),
                color=peach, lw=0.4, alpha=0.22)

    # Ghost unit circle (original space reference)
    ax.plot(*CIRCLE, color=overlay0, lw=0.8, alpha=0.30, ls=":")

    # Transformed unit circle
    ax.plot(*(M @ CIRCLE), color=mauve, lw=2.0, alpha=0.85)

    # Ghost basis vectors
    ax.quiver(0, 0, *I2, color=teal,  alpha=0.20, angles="xy",
              scale_units="xy", scale=1, width=0.007, headwidth=4)
    ax.quiver(0, 0, *J2, color=peach, alpha=0.20, angles="xy",
              scale_units="xy", scale=1, width=0.007, headwidth=4)

    # Transformed basis vectors
    ax.quiver(0, 0, *(M @ I2), color=teal, angles="xy",
              scale_units="xy", scale=1, width=0.010, headwidth=4, label="î′")
    ax.quiver(0, 0, *(M @ J2), color=peach, angles="xy",
              scale_units="xy", scale=1, width=0.010, headwidth=4, label="ĵ′")

    ax.text(-LIMIT + 0.12, -LIMIT + 0.28,
            f"[{M[0,0]:+.2f}  {M[0,1]:+.2f}]\n[{M[1,0]:+.2f}  {M[1,1]:+.2f}]",
            fontsize=8, color=text_col, fontfamily="monospace", va="bottom")

    ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
              facecolor=surface0, labelcolor=text_col)

# ── Control panel ─────────────────────────────────────────────────────────────
LEFT  = 0.785
SW    = 0.088
SH    = 0.034
HGAP  = 0.012

fig.text(LEFT + SW + HGAP / 2, 0.90, "Matrix A",
         ha="center", color=mauve, fontsize=10, fontweight="bold")

sliders = {}
for r, c, y, init in [(0, 0, 0.82, 1.0), (0, 1, 0.82, 0.0),
                       (1, 0, 0.75, 0.0), (1, 1, 0.75, 1.0)]:
    x = LEFT + c * (SW + HGAP)
    fig.text(x + SW / 2, y + SH + 0.005, f"A[{r},{c}]",
             ha="center", fontsize=7.5, color=text_col)
    sax = fig.add_axes([x, y, SW, SH])
    sax.set_facecolor(surface0)
    sl = Slider(sax, "", valmin=-3.0, valmax=3.0, valinit=init,
                color=(teal if r == c else peach), track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(8)
    sliders[(r, c)] = sl

panel_cx = LEFT + SW + HGAP / 2

det_text  = fig.text(panel_cx, 0.71, "",
                     ha="center", color=sky, fontsize=9, fontweight="bold")
info_text = fig.text(panel_cx, 0.66, "",
                     ha="center", va="top", color=text_col,
                     fontsize=7.5, fontfamily="monospace")
ortho_text = fig.text(panel_cx, 0.36, "",
                      ha="center", color=green, fontsize=9, fontweight="bold")

# ── Presets ───────────────────────────────────────────────────────────────────
fig.text(panel_cx, 0.33, "Presets", ha="center", color=mauve,
         fontsize=8.5, fontweight="bold")
preset_ax = fig.add_axes([LEFT, 0.10, 2 * SW + HGAP, 0.21])
preset_ax.set_facecolor(surface0)
presets = RadioButtons(preset_ax,
    ["Identity", "Rotation 45°", "Shear X", "Stretch 2×3", "Custom"], active=4)
for lbl in presets.labels:
    lbl.set_color(text_col)
    lbl.set_fontsize(7.5)
presets.activecolor = teal

PRESET_MATS = {
    "Identity":     np.eye(2),
    "Rotation 45°": np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                               [np.sin(np.pi / 4),  np.cos(np.pi / 4)]]),
    "Shear X":      np.array([[1.0, 1.0], [0.0, 1.0]]),
    "Stretch 2×3":  np.array([[2.0, 0.0], [0.0, 3.0]]),
}

def on_preset(label):
    if label in PRESET_MATS:
        M = PRESET_MATS[label]
        for (r, c), sl in sliders.items():
            sl.set_val(M[r, c])

presets.on_clicked(on_preset)

# ── Update ────────────────────────────────────────────────────────────────────
def update(_):
    A   = np.array([[sliders[(r, c)].val for c in range(2)] for r in range(2)])
    AT  = A.T
    det = np.linalg.det(A)

    draw_transform(ax_A,  A,  "A   — original transform", teal,
                   "columns = where  î, ĵ  land")
    draw_transform(ax_AT, AT, "Aᵀ  — transpose", peach,
                   "swap rows ↔ columns")

    if abs(det) > 1e-10:
        Ainv = np.linalg.inv(A)
        draw_transform(ax_Ai, Ainv, "A⁻¹ — inverse", lavender,
                       "A · A⁻¹ = I   (undoes A)")

        diff = np.max(np.abs(AT - Ainv))
        if diff < 0.015:
            ortho_text.set_text("Aᵀ = A⁻¹  ✓\northogonal matrix!")
            ortho_text.set_color(green)
        else:
            ortho_text.set_text(f"Aᵀ  ≠  A⁻¹\nmax|Aᵀ − A⁻¹| = {diff:.3f}")
            ortho_text.set_color(flamingo)

        info_text.set_text(
            f"A:   [{A[0,0]:+.2f}  {A[0,1]:+.2f}]\n"
            f"     [{A[1,0]:+.2f}  {A[1,1]:+.2f}]\n\n"
            f"Aᵀ:  [{AT[0,0]:+.2f}  {AT[0,1]:+.2f}]\n"
            f"     [{AT[1,0]:+.2f}  {AT[1,1]:+.2f}]\n\n"
            f"A⁻¹: [{Ainv[0,0]:+.2f}  {Ainv[0,1]:+.2f}]\n"
            f"     [{Ainv[1,0]:+.2f}  {Ainv[1,1]:+.2f}]"
        )
    else:
        ax_Ai.cla()
        style_ax(ax_Ai, "A⁻¹ — inverse", flamingo,
                 "det(A) = 0 — not invertible")
        ax_Ai.text(0, 0, "Singular matrix\nnot invertible",
                   ha="center", va="center", fontsize=13,
                   color=flamingo, fontweight="bold")

        ortho_text.set_text("Singular — no inverse exists")
        ortho_text.set_color(flamingo)
        info_text.set_text(
            f"A:   [{A[0,0]:+.2f}  {A[0,1]:+.2f}]\n"
            f"     [{A[1,0]:+.2f}  {A[1,1]:+.2f}]\n\n"
            f"Aᵀ:  [{AT[0,0]:+.2f}  {AT[0,1]:+.2f}]\n"
            f"     [{AT[1,0]:+.2f}  {AT[1,1]:+.2f}]\n\n"
            f"A⁻¹: undefined\n     (det = 0)"
        )

    det_col = sky if abs(det) > 0.01 else flamingo
    det_text.set_text(f"det(A) = {det:.3f}")
    det_text.set_color(det_col)
    fig.canvas.draw_idle()

for sl in sliders.values():
    sl.on_changed(update)

update(None)
plt.show()
