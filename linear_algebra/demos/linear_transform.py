import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    overlay0, surface0, surface1, base, text_col,
    PANE,
)

# ── Basis vectors ─────────────────────────────────────────────────────────────
i_hat    = np.array([1, 0, 0])
j_hat    = np.array([0, 1, 0])
k_hat    = np.array([0, 0, 1])
i_hat_2d = np.array([1, 0])
j_hat_2d = np.array([0, 1])

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Linear Transformation — Live Matrix Editor", fontsize=13, fontweight="bold")
fig.patch.set_facecolor(base)

ax3d = fig.add_axes([0.02, 0.08, 0.50, 0.85], projection="3d")
ax2d = fig.add_axes([0.02, 0.08, 0.50, 0.85])
ax2d.set_visible(False)

# ── Helpers ───────────────────────────────────────────────────────────────────
LIMIT = 2.2

def style_3d(ax):
    ax.set_facecolor(base)
    ax.xaxis.set_pane_color(PANE)
    ax.yaxis.set_pane_color(PANE)
    ax.zaxis.set_pane_color(PANE)
    ax.xaxis.line.set_color(overlay0)
    ax.yaxis.line.set_color(overlay0)
    ax.zaxis.line.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_zlim(-LIMIT, LIMIT)
    ax.set_xlabel("x", color=overlay0, labelpad=6)
    ax.set_ylabel("y", color=overlay0, labelpad=6)
    ax.set_zlabel("z", color=overlay0, labelpad=6)
    for d in [(1,0,0),(0,1,0),(0,0,1)]:
        v = np.array(d, float) * LIMIT
        ax.plot(*zip(-v, v), color=overlay0, lw=0.6, ls="--", alpha=0.4)

def style_2d(ax):
    ax.set_facecolor(base)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color=overlay0)
    ax.set_ylabel("y", color=overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(overlay0)
    ax.axhline(0, color=overlay0, lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color=overlay0, lw=0.6, ls="--", alpha=0.4)
    ax.grid(True, color=overlay0, lw=0.3, alpha=0.2)

def draw_vectors_3d(ax, A):
    ax.cla()
    style_3d(ax)
    configs = [
        (i_hat, teal,  lavender, "î",  "î′"),
        (j_hat, peach, flamingo, "ĵ",  "ĵ′"),
        (k_hat, green, sky,      "k̂", "k̂′"),
    ]
    for basis, orig_col, trans_col, orig_lbl, trans_lbl in configs:
        transformed = A @ basis
        ax.quiver(0,0,0, *basis,       color=orig_col,  alpha=0.25,
                  arrow_length_ratio=0.15, linewidth=1.5)
        ax.quiver(0,0,0, *transformed, color=trans_col, alpha=1.0,
                  arrow_length_ratio=0.15, linewidth=2.2,
                  label=f"{orig_lbl}→{trans_lbl}")
    det       = np.linalg.det(A)
    det_color = sky if abs(det) > 0.01 else flamingo
    ax.set_title(f"det(A) = {det:.3f}", fontsize=10, color=det_color, pad=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)

def draw_vectors_2d(ax, A2):
    ax.cla()
    style_2d(ax)
    configs = [
        (i_hat_2d, teal,  lavender, "î",  "î′"),
        (j_hat_2d, peach, flamingo, "ĵ",  "ĵ′"),
    ]
    for basis, orig_col, trans_col, orig_lbl, trans_lbl in configs:
        transformed = A2 @ basis
        ax.quiver(0, 0, basis[0], basis[1],
                  color=orig_col, alpha=0.25,
                  angles="xy", scale_units="xy", scale=1,
                  width=0.007, headwidth=4)
        ax.quiver(0, 0, transformed[0], transformed[1],
                  color=trans_col, alpha=1.0,
                  angles="xy", scale_units="xy", scale=1,
                  width=0.007, headwidth=4,
                  label=f"{orig_lbl}→{trans_lbl}")
    det       = np.linalg.det(A2)
    det_color = sky if abs(det) > 0.01 else flamingo
    ax.set_title(f"det(A) = {det:.3f}", fontsize=10, color=det_color, pad=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3,
              facecolor=surface0, labelcolor=text_col)

# ── Slider layout constants ───────────────────────────────────────────────────
LEFT     = 0.58
SLIDER_W = 0.12
SLIDER_H = 0.030
H_GAP    = 0.02
ROW_Y    = [0.78, 0.50, 0.22]

sliders      = {}
slider_axes  = {}
slider_texts = {}
col_headers  = []
row_labels   = []

for c in range(3):
    cx = LEFT + c * (SLIDER_W + H_GAP) + SLIDER_W / 2
    t = fig.text(cx, 0.91, f"col {c}", ha="center", va="bottom",
                 fontsize=8, color=mauve, fontweight="bold")
    col_headers.append(t)

for r in range(3):
    ry_centre = ROW_Y[r] + SLIDER_H / 2
    t = fig.text(LEFT - 0.015, ry_centre, f"row {r}",
                 ha="right", va="center",
                 fontsize=8, color=mauve, fontweight="bold")
    row_labels.append(t)

    for c in range(3):
        sx = LEFT + c * (SLIDER_W + H_GAP)
        sy = ROW_Y[r]

        lbl = fig.text(sx + SLIDER_W / 2, sy + SLIDER_H + 0.012,
                       f"A[{r},{c}]", ha="center", va="bottom",
                       fontsize=7.5, color=text_col)

        sax = fig.add_axes([sx, sy, SLIDER_W, SLIDER_H])
        sax.set_facecolor(surface0)
        init_val = 1.0 if r == c else 0.0
        sl = Slider(
            ax=sax, label="",
            valmin=-3.0, valmax=3.0, valinit=init_val,
            color=teal if r == c else peach,
            track_color=surface1,
        )
        sl.valtext.set_color(text_col)
        sl.valtext.set_fontsize(8)
        sliders[(r, c)]      = sl
        slider_axes[(r, c)]  = sax
        slider_texts[(r, c)] = [lbl]

# ── Mode toggle ───────────────────────────────────────────────────────────────
radio_ax = fig.add_axes([0.645, 0.93, 0.25, 0.055])
radio_ax.set_facecolor(surface0)
radio = RadioButtons(radio_ax, ["3D", "2D"], active=0, activecolor=teal)
for lbl in radio.labels:
    lbl.set_color(text_col)
    lbl.set_fontsize(9)

current_mode = ["3D"]

def apply_mode_visibility(mode):
    is_3d = (mode == "3D")
    ax3d.set_visible(is_3d)
    ax2d.set_visible(not is_3d)
    col_headers[2].set_visible(is_3d)
    row_labels[2].set_visible(is_3d)
    for r in range(3):
        for c in range(3):
            visible = is_3d or (r < 2 and c < 2)
            slider_axes[(r, c)].set_visible(visible)
            for t in slider_texts[(r, c)]:
                t.set_visible(visible)

def on_mode_change(label):
    current_mode[0] = label
    apply_mode_visibility(label)
    update(None)

radio.on_clicked(on_mode_change)

# ── Info text ─────────────────────────────────────────────────────────────────
panel_cx = LEFT + (3 * SLIDER_W + 2 * H_GAP) / 2

det_text = fig.text(panel_cx, 0.11, "det(A) = 1.000",
                    ha="center", va="center",
                    fontsize=11, color=sky, fontweight="bold")

mat_text = fig.text(panel_cx, 0.055, "",
                    ha="center", va="center",
                    fontsize=8, color=text_col, fontfamily="monospace")

# ── Update ────────────────────────────────────────────────────────────────────
def update(_):
    mode = current_mode[0]
    if mode == "3D":
        A   = np.array([[sliders[(r, c)].val for c in range(3)] for r in range(3)])
        draw_vectors_3d(ax3d, A)
        det     = np.linalg.det(A)
        mat_str = "\n".join(
            "[ " + "  ".join(f"{A[r,c]:+.2f}" for c in range(3)) + " ]"
            for r in range(3)
        )
    else:
        A2  = np.array([[sliders[(r, c)].val for c in range(2)] for r in range(2)])
        draw_vectors_2d(ax2d, A2)
        det     = np.linalg.det(A2)
        mat_str = "\n".join(
            "[ " + "  ".join(f"{A2[r,c]:+.2f}" for c in range(2)) + " ]"
            for r in range(2)
        )

    det_color = sky if abs(det) > 0.01 else flamingo
    det_text.set_text(f"det(A) = {det:.3f}")
    det_text.set_color(det_color)
    mat_text.set_text(mat_str)
    fig.canvas.draw_idle()

for sl in sliders.values():
    sl.on_changed(update)

update(None)
plt.show()
