import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    overlay0, surface0, surface1, base, text_col,
    PANE,
)

# ── State ──────────────────────────────────────────────────────────────────────
mode_3d = False
plot_ax = None

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.suptitle("Cosine Similarity — Interactive Visualizer", fontsize=13, fontweight="bold")
fig.patch.set_facecolor(base)

# ── Layout constants ───────────────────────────────────────────────────────────
LEFT     = 0.57
SLIDER_W = 0.37
SLIDER_H = 0.028
LIMIT    = 2.5

# ── Style helpers ──────────────────────────────────────────────────────────────
def style_2d(ax):
    ax.set_facecolor(base)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    ax.axhline(0, color=overlay0, lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color=overlay0, lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("x", color=overlay0)
    ax.set_ylabel("y", color=overlay0)
    ax.set_title("2D View  (z ignored)", fontsize=9, color=overlay0, pad=5)

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
    ax.set_title("3D View", fontsize=9, color=overlay0, pad=8)
    for d in [(1,0,0),(0,1,0),(0,0,1)]:
        v = np.array(d, float) * LIMIT
        ax.plot(*zip(-v, v), color=overlay0, lw=0.6, ls="--", alpha=0.4)

# ── Draw: 2D ──────────────────────────────────────────────────────────────────
def draw_angle_arc(ax, a2, b2, cos_sim, angle_deg):
    na = np.linalg.norm(a2)
    nb = np.linalg.norm(b2)
    if na < 1e-9 or nb < 1e-9:
        return
    arc_r   = 0.40
    ang_a   = np.degrees(np.arctan2(a2[1]/na, a2[0]/na))
    ang_b   = np.degrees(np.arctan2(b2[1]/nb, b2[0]/nb))
    start, end = sorted([ang_a, ang_b])
    if end - start > 180:
        start, end = end, start + 360
    arc_col = sky if cos_sim > 0.1 else flamingo if cos_sim < -0.1 else mauve
    arc = patches.Arc((0, 0), 2*arc_r, 2*arc_r,
                      theta1=start, theta2=end,
                      color=arc_col, lw=1.8, alpha=0.85)
    ax.add_patch(arc)
    mid = np.radians((start + end) / 2)
    ax.text(arc_r * 1.65 * np.cos(mid), arc_r * 1.65 * np.sin(mid),
            f"{angle_deg:.1f}°",
            color=arc_col, fontsize=9, ha="center", va="center", fontweight="bold")

def draw_2d(ax, A, B):
    ax.cla()
    style_2d(ax)
    a2, b2 = A[:2], B[:2]
    na, nb = np.linalg.norm(a2), np.linalg.norm(b2)

    if na > 1e-9 and nb > 1e-9:
        cos_sim = float(np.clip(np.dot(a2, b2) / (na * nb), -1, 1))
        angle   = np.degrees(np.arccos(cos_sim))
        draw_angle_arc(ax, a2, b2, cos_sim, angle)

        nb_hat = b2 / nb
        proj   = np.dot(a2, nb_hat) * nb_hat
        ax.annotate("", xy=tuple(proj), xytext=tuple(a2),
                    arrowprops=dict(arrowstyle="-", color=green,
                                   lw=1.2, linestyle="dashed", alpha=0.6))
        ax.plot(*proj, "o", color=green, ms=5, alpha=0.85, zorder=5)
        ax.text(-LIMIT + 0.1, -LIMIT + 0.15,
                "dashed line = projection of A onto B",
                color=green, fontsize=7, alpha=0.75)

    arrowkw = dict(mutation_scale=20, lw=2.2, zorder=6)
    ax.annotate("", xy=tuple(a2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=teal,   **arrowkw))
    ax.annotate("", xy=tuple(b2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=peach,  **arrowkw))

    off = 0.10
    ax.text(a2[0] + off, a2[1] + off, "A", color=teal,  fontsize=13, fontweight="bold", zorder=7)
    ax.text(b2[0] + off, b2[1] + off, "B", color=peach, fontsize=13, fontweight="bold", zorder=7)

# ── Draw: 3D ──────────────────────────────────────────────────────────────────
def draw_3d(ax, A, B):
    ax.cla()
    style_3d(ax)
    na, nb = np.linalg.norm(A), np.linalg.norm(B)

    ax.quiver(0, 0, 0, *A, color=teal,  lw=2.2, arrow_length_ratio=0.12, label="A")
    ax.quiver(0, 0, 0, *B, color=peach, lw=2.2, arrow_length_ratio=0.12, label="B")

    if na > 1e-9 and nb > 1e-9:
        nb_hat = B / nb
        proj   = np.dot(A, nb_hat) * nb_hat
        ax.quiver(0, 0, 0, *proj, color=green, lw=1.5, arrow_length_ratio=0.12,
                  alpha=0.75, label="proj(A→B)")
        ax.plot([A[0], proj[0]], [A[1], proj[1]], [A[2], proj[2]],
                color=green, lw=1.0, ls="dashed", alpha=0.5)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.3)

# ── Axes factory ───────────────────────────────────────────────────────────────
def make_plot_ax():
    global plot_ax
    if plot_ax is not None:
        fig.delaxes(plot_ax)
    if mode_3d:
        plot_ax = fig.add_axes([0.03, 0.07, 0.49, 0.86], projection="3d")
    else:
        plot_ax = fig.add_axes([0.03, 0.07, 0.49, 0.86])

# ── Sliders ────────────────────────────────────────────────────────────────────
def add_slider(ry, init, color):
    sax = fig.add_axes([LEFT, ry, SLIDER_W, SLIDER_H])
    sax.set_facecolor(surface0)
    sl  = Slider(sax, "", -3.0, 3.0, valinit=init, color=color, track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(8)
    return sl

fig.text(LEFT + SLIDER_W/2, 0.905, "Vector  A",
         ha="center", fontsize=10, color=teal, fontweight="bold")

rows_a  = [0.845, 0.775, 0.705]
inits_a = [1.0, 0.5, 0.0]
for lbl, ry in zip(["x", "y", "z"], rows_a):
    fig.text(LEFT - 0.010, ry + SLIDER_H/2, lbl,
             ha="right", va="center", fontsize=9, color=teal)

sliders = {
    "a0": add_slider(rows_a[0], inits_a[0], teal),
    "a1": add_slider(rows_a[1], inits_a[1], teal),
    "a2": add_slider(rows_a[2], inits_a[2], teal),
}

fig.text(LEFT + SLIDER_W/2, 0.645, "Vector  B",
         ha="center", fontsize=10, color=peach, fontweight="bold")

rows_b  = [0.585, 0.515, 0.445]
inits_b = [0.5, 1.0, 0.0]
for lbl, ry in zip(["x", "y", "z"], rows_b):
    fig.text(LEFT - 0.010, ry + SLIDER_H/2, lbl,
             ha="right", va="center", fontsize=9, color=peach)

sliders.update({
    "b0": add_slider(rows_b[0], inits_b[0], peach),
    "b1": add_slider(rows_b[1], inits_b[1], peach),
    "b2": add_slider(rows_b[2], inits_b[2], peach),
})

fig.text(LEFT + SLIDER_W/2, rows_b[2] - 0.026,
         "z  is only used in 3D mode",
         ha="center", fontsize=7, color=overlay0, style="italic")

# ── Toggle button ──────────────────────────────────────────────────────────────
btn_ax = fig.add_axes([LEFT + 0.06, 0.340, SLIDER_W - 0.12, 0.042])
btn_ax.set_facecolor(surface0)
toggle_btn = Button(btn_ax, "Switch to 3D  →", color=surface0, hovercolor=surface1)
toggle_btn.label.set_color(mauve)
toggle_btn.label.set_fontsize(9)
toggle_btn.label.set_fontweight("bold")

# ── Formula + info ─────────────────────────────────────────────────────────────
cx = LEFT + SLIDER_W / 2

fig.text(cx, 0.290, "cos θ  =  (A · B) / (|A| · |B|)",
         ha="center", fontsize=9, color=overlay0, fontfamily="monospace")

info_interp = fig.text(cx, 0.238, "—",
                       ha="center", fontsize=9, color=mauve, style="italic")
info_sim    = fig.text(cx, 0.190, "cos θ = 0.0000",
                       ha="center", fontsize=12, color=sky, fontweight="bold")
info_ang    = fig.text(cx, 0.148, "θ = 90.00°",
                       ha="center", fontsize=10, color=lavender)
info_dot    = fig.text(cx, 0.106, "A · B = 0.0000",
                       ha="center", fontsize=9, color=overlay0)
info_mags   = fig.text(cx, 0.064, "|A| = 1.118   |B| = 1.118",
                       ha="center", fontsize=8, color=overlay0)

# ── Interpretation ─────────────────────────────────────────────────────────────
def interpret(cos_sim):
    if abs(cos_sim) > 0.999:
        return ("Parallel — identical direction", teal) if cos_sim > 0 \
               else ("Anti-parallel — opposite direction", flamingo)
    if abs(cos_sim) < 0.05:
        return "Orthogonal — completely unrelated", mauve
    if cos_sim > 0.70:
        return "Highly similar", teal
    if cos_sim > 0.30:
        return "Moderately similar", sky
    if cos_sim > 0:
        return "Weakly similar", lavender
    if cos_sim > -0.30:
        return "Weakly dissimilar", peach
    return "Highly dissimilar", flamingo

# ── Update ─────────────────────────────────────────────────────────────────────
def update(_):
    A = np.array([sliders["a0"].val, sliders["a1"].val, sliders["a2"].val])
    B = np.array([sliders["b0"].val, sliders["b1"].val, sliders["b2"].val])

    va = A      if mode_3d else A[:2]
    vb = B      if mode_3d else B[:2]
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)

    if na > 1e-9 and nb > 1e-9:
        dot     = float(np.dot(va, vb))
        cos_sim = float(np.clip(dot / (na * nb), -1.0, 1.0))
        angle   = np.degrees(np.arccos(cos_sim))
        label, col = interpret(cos_sim)
        info_sim.set_text(f"cos θ = {cos_sim:+.4f}")
        info_sim.set_color(col)
        info_ang.set_text(f"θ = {angle:.2f}°")
        info_dot.set_text(f"A · B = {dot:.4f}")
        info_mags.set_text(f"|A| = {na:.3f}   |B| = {nb:.3f}")
        info_interp.set_text(label)
        info_interp.set_color(col)
    else:
        info_sim.set_text("cos θ = undefined")
        info_sim.set_color(overlay0)
        info_ang.set_text("θ = undefined")
        info_dot.set_text("A · B = 0")
        info_mags.set_text("")
        info_interp.set_text("(zero vector)")
        info_interp.set_color(overlay0)

    if mode_3d:
        draw_3d(plot_ax, A, B)
    else:
        draw_2d(plot_ax, A, B)

    fig.canvas.draw_idle()

# ── Toggle mode ────────────────────────────────────────────────────────────────
def toggle_mode(_):
    global mode_3d
    mode_3d = not mode_3d
    toggle_btn.label.set_text("← Switch to 2D" if mode_3d else "Switch to 3D  →")
    make_plot_ax()
    update(None)

toggle_btn.on_clicked(toggle_mode)
for sl in sliders.values():
    sl.on_changed(update)

# ── Launch ─────────────────────────────────────────────────────────────────────
make_plot_ax()
update(None)
plt.show()
