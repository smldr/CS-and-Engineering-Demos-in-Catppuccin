import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    overlay0, surface0, surface1, base, text_col, PANE,
)

# ── Grid ──────────────────────────────────────────────────────────────────────
RANGE = 2.5
N = 60
_xs = np.linspace(-RANGE, RANGE, N)
X, Y = np.meshgrid(_xs, _xs)

# ── Function definitions ──────────────────────────────────────────────────────
def _bowl(x, y):
    return x**2 + 0.5 * y**2

def _saddle(x, y):
    return x**2 - y**2

def _banana(x, y):
    return (1 - x)**2 + 5 * (y - x**2)**2

def _gaussian(x, y):
    return -3 * np.exp(-0.5 * (x**2 + y**2)) + 0.3 * (x**2 + y**2)

def _ripple(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.cos(np.pi * r) * np.exp(-0.2 * r**2)

FUNCTIONS = [
    ("Bowl  x²+½y²",    _bowl),
    ("Saddle  x²−y²",   _saddle),
    ("Rosenbrock",       _banana),
    ("Gaussian",         _gaussian),
    ("Ripple  cos(πr)",  _ripple),
]

func_idx    = [0]
order_state = [2]
show_newton = [True]

# ── Numerical gradient & Hessian ──────────────────────────────────────────────
EPS = 1e-5

def _grad(f, x, y):
    return np.array([
        (f(x + EPS, y) - f(x - EPS, y)) / (2 * EPS),
        (f(x, y + EPS) - f(x, y - EPS)) / (2 * EPS),
    ])

def _hess(f, x, y):
    f0  = f(x, y)
    fxx = (f(x + EPS, y) - 2*f0 + f(x - EPS, y)) / EPS**2
    fyy = (f(x, y + EPS) - 2*f0 + f(x, y - EPS)) / EPS**2
    fxy = (f(x+EPS, y+EPS) - f(x+EPS, y-EPS)
           - f(x-EPS, y+EPS) + f(x-EPS, y-EPS)) / (4 * EPS**2)
    return np.array([[fxx, fxy], [fxy, fyy]])

# ── Second-order Taylor surface on the grid ───────────────────────────────────
def taylor_grid(f, x0, y0, order):
    f0 = f(x0, y0)
    dx = X - x0
    dy = Y - y0
    Z  = np.full_like(X, f0, dtype=float)
    if order >= 1:
        g  = _grad(f, x0, y0)
        Z += g[0] * dx + g[1] * dy
    if order >= 2:
        H  = _hess(f, x0, y0)
        Z += 0.5 * (H[0,0]*dx**2 + 2*H[0,1]*dx*dy + H[1,1]*dy**2)
    return Z

# ── Colourmaps & labels ───────────────────────────────────────────────────────
surface_cmap = LinearSegmentedColormap.from_list(
    "mocha_surf", [sky, teal, green, peach, flamingo], N=256
)
APPROX_COLOR = {0: lavender, 1: sky, 2: peach}
ORDER_LABEL  = {
    0: "0th order  —  constant  f(x₀)",
    1: "1st order  —  tangent plane",
    2: "2nd order  —  paraboloid",
}
_ORDER_MAP = {
    "0th  —  constant":      0,
    "1st  —  tangent plane": 1,
    "2nd  —  paraboloid":    2,
}

# ── Figure & axes ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor(base)
fig.suptitle(
    "Second-Order Taylor Expansion  —  "
    "f(x₀+δ) ≈ f(x₀) + ∇fᵀδ + ½δᵀHδ",
    fontsize=13, fontweight="bold", color=text_col,
)

ax3d = fig.add_axes([0.01, 0.09, 0.46, 0.88], projection="3d")
ax2d = fig.add_axes([0.50, 0.47, 0.215, 0.50])

# ── Axis helpers ──────────────────────────────────────────────────────────────
def style_3d(ax):
    ax.set_facecolor(base)
    ax.xaxis.set_pane_color(PANE)
    ax.yaxis.set_pane_color(PANE)
    ax.zaxis.set_pane_color(PANE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    ax.set_xlabel("x₁", color=overlay0, labelpad=8)
    ax.set_ylabel("x₂", color=overlay0, labelpad=8)
    ax.set_zlabel("f", color=overlay0, labelpad=8)

def style_2d(ax):
    ax.set_facecolor(base)
    ax.set_xlim(-RANGE, RANGE)
    ax.set_ylim(-RANGE, RANGE)
    ax.set_aspect("equal")
    ax.tick_params(colors=overlay0, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(overlay0)
    ax.axhline(0, color=overlay0, lw=0.5, ls="--", alpha=0.3)
    ax.axvline(0, color=overlay0, lw=0.5, ls="--", alpha=0.3)
    ax.grid(True, color=overlay0, lw=0.3, alpha=0.2)
    ax.set_xlabel("x₁", color=overlay0, fontsize=8)
    ax.set_ylabel("x₂", color=overlay0, fontsize=8)
    ax.set_title("Contour view  —  base point & Newton step",
                 color=mauve, fontsize=9, fontweight="bold")

# ── Control panel ─────────────────────────────────────────────────────────────
LEFT = 0.745
SW   = 0.235
SH   = 0.030

fig.text(LEFT + SW/2, 0.940, "── Approximation Order ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")
order_ax_w = fig.add_axes([LEFT, 0.858, SW, 0.072])
order_ax_w.set_facecolor(surface0)
order_radio = RadioButtons(
    order_ax_w,
    ["0th  —  constant", "1st  —  tangent plane", "2nd  —  paraboloid"],
    active=2,
)
for lbl in order_radio.labels:
    lbl.set_color(text_col)
    lbl.set_fontsize(8)
order_radio.activecolor = mauve

fig.text(LEFT + SW/2, 0.843, "── Function ──",
         ha="center", fontsize=9, color=lavender, fontweight="bold")
func_ax_w = fig.add_axes([LEFT, 0.750, SW, 0.085])
func_ax_w.set_facecolor(surface0)
func_radio = RadioButtons(func_ax_w, [f[0] for f in FUNCTIONS], active=0)
for lbl in func_radio.labels:
    lbl.set_color(text_col)
    lbl.set_fontsize(7.5)
func_radio.activecolor = lavender

fig.text(LEFT + SW/2, 0.735, "── Base point x₀ ──",
         ha="center", fontsize=9, color=teal, fontweight="bold")

x0_sliders = {}
for label, key, y_pos, color in [
    ("x₀₁", "x0", 0.692, teal),
    ("x₀₂", "y0", 0.648, peach),
]:
    fig.text(LEFT - 0.010, y_pos + SH/2,
             label, ha="right", va="center", fontsize=9.5, color=text_col)
    sax = fig.add_axes([LEFT, y_pos, SW, SH])
    sax.set_facecolor(surface0)
    sl = Slider(sax, "", valmin=-RANGE, valmax=RANGE, valinit=0.5,
                color=color, track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(8)
    x0_sliders[key] = sl

newton_ax_w = fig.add_axes([LEFT, 0.604, SW, 0.033])
newton_ax_w.set_facecolor(surface0)
btn_newton = Button(newton_ax_w, "Newton step : ON",
                    color=surface0, hovercolor=surface1)
btn_newton.label.set_color(mauve)
btn_newton.label.set_fontsize(8.5)
btn_newton.label.set_fontweight("bold")

info_text = fig.text(
    LEFT + SW/2, 0.585, "",
    ha="center", va="top", color=text_col,
    fontsize=7.5, fontfamily="monospace",
)

# ── Redraw ────────────────────────────────────────────────────────────────────
def redraw():
    x0    = x0_sliders["x0"].val
    y0    = x0_sliders["y0"].val
    order = order_state[0]
    f     = FUNCTIONS[func_idx[0]][1]

    Z_surf = f(X, Y)
    z_lo   = float(np.percentile(Z_surf, 2))
    z_hi   = float(np.percentile(Z_surf, 98))
    z_range = max(z_hi - z_lo, 0.01)
    floor   = z_lo - 0.4 * z_range

    f0     = float(f(x0, y0))
    g      = _grad(f, x0, y0)
    H      = _hess(f, x0, y0)
    eigv   = np.linalg.eigvalsh(H)
    det_H  = float(np.linalg.det(H))
    g_norm = float(np.linalg.norm(g))
    a_col  = APPROX_COLOR[order]

    Z_raw = taylor_grid(f, x0, y0, order)
    Z_3d  = np.clip(Z_raw, z_lo - 0.6*z_range, z_hi + 0.6*z_range)

    # ── 3D plot ───────────────────────────────────────────────────────────────
    ax3d.cla()
    style_3d(ax3d)

    ax3d.plot_surface(X, Y, Z_surf, cmap=surface_cmap, alpha=0.65,
                      linewidth=0, antialiased=True, rcount=50, ccount=50)
    ax3d.contourf(X, Y, Z_surf, zdir="z", offset=floor,
                  cmap=surface_cmap, alpha=0.25, levels=16)

    ax3d.plot_surface(X, Y, Z_3d, color=a_col, alpha=0.45,
                      linewidth=0, antialiased=True, rcount=40, ccount=40)

    ax3d.plot([x0, x0], [y0, y0], [floor, f0], color=flamingo, lw=2, alpha=0.85)
    ax3d.scatter([x0], [y0], [f0],    color=flamingo, s=70, zorder=6)
    ax3d.scatter([x0], [y0], [floor], color=flamingo, s=35, marker="+", zorder=6)

    ax3d.set_zlim(floor, z_hi + 0.4*z_range)
    ax3d.text2D(0.02, 0.96, ORDER_LABEL[order],
                transform=ax3d.transAxes,
                color=a_col, fontsize=9, fontweight="bold")

    # ── 2D contour ────────────────────────────────────────────────────────────
    ax2d.cla()
    style_2d(ax2d)

    ax2d.contourf(X, Y, Z_surf, levels=18, cmap=surface_cmap, alpha=0.50)
    ax2d.contour(X, Y, Z_surf, levels=12,
                 colors=[overlay0], linewidths=0.5, alpha=0.45)

    if order > 0:
        try:
            ax2d.contour(X, Y, Z_raw, levels=10,
                         colors=[a_col], linewidths=0.8,
                         linestyles="dashed", alpha=0.65)
        except Exception:
            pass

    ax2d.plot(x0, y0, "o", color=flamingo, markersize=8, zorder=5)

    if g_norm > 1e-10:
        scale = min(0.7 / g_norm, 1.5)
        ax2d.annotate(
            "", xy=(x0 + g[0]*scale, y0 + g[1]*scale), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=teal, lw=2.0),
        )

    newton_drawn = False
    if show_newton[0] and abs(det_H) > 1e-8:
        try:
            ns  = -np.linalg.solve(H, g)
            x_n = x0 + ns[0]
            y_n = y0 + ns[1]
            xc  = float(np.clip(x_n, -RANGE, RANGE))
            yc  = float(np.clip(y_n, -RANGE, RANGE))
            ax2d.annotate(
                "", xy=(xc, yc), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=peach, lw=2.0,
                                linestyle="dashed"),
            )
            if -RANGE <= x_n <= RANGE and -RANGE <= y_n <= RANGE:
                ax2d.plot(x_n, y_n, "*", color=peach, markersize=12, zorder=5)
            newton_drawn = True
        except np.linalg.LinAlgError:
            pass

    legend_elems = [
        Line2D([0], [0], color=flamingo, marker="o", lw=0,
               label=f"x₀  ({x0:.2f}, {y0:.2f})"),
        Line2D([0], [0], color=teal, lw=2, label="∇f  gradient"),
    ]
    if newton_drawn:
        legend_elems.append(
            Line2D([0], [0], color=peach, lw=2, ls="--",
                   label="−H⁻¹∇f  Newton")
        )
    ax2d.legend(handles=legend_elems, loc="upper right", fontsize=6.5,
                framealpha=0.4, facecolor=surface0, labelcolor=text_col)

    # ── Info text ─────────────────────────────────────────────────────────────
    cp_note = ""
    if g_norm < 0.08:
        n_pos = sum(e > 1e-8 for e in eigv)
        n_neg = sum(e < -1e-8 for e in eigv)
        if n_pos == 2:
            cp_note = "  ← minimum"
        elif n_neg == 2:
            cp_note = "  ← maximum"
        elif n_pos == 1 and n_neg == 1:
            cp_note = "  ← saddle"
        else:
            cp_note = "  ← degenerate"

    newton_info = ""
    if abs(det_H) > 1e-8:
        try:
            ns = -np.linalg.solve(H, g)
            newton_info = f"−H⁻¹∇f = ({ns[0]:+.3f}, {ns[1]:+.3f})"
        except np.linalg.LinAlgError:
            pass

    lines = [
        f"f(x₀)  = {f0:+.4f}",
        f"∇f     = ({g[0]:+.3f}, {g[1]:+.3f})",
        f"|∇f|   = {g_norm:.4f}{cp_note}",
        "",
        f"H  = [{H[0,0]:+.3f}  {H[0,1]:+.3f}]",
        f"     [{H[1,0]:+.3f}  {H[1,1]:+.3f}]",
        f"λ  = {eigv[0]:+.3f},  {eigv[1]:+.3f}",
        f"det(H) = {det_H:+.3f}",
    ]
    if newton_info:
        lines += ["", newton_info]
    info_text.set_text("\n".join(lines))

    fig.canvas.draw_idle()

# ── Callbacks ─────────────────────────────────────────────────────────────────
def on_order(label):
    order_state[0] = _ORDER_MAP[label]
    redraw()

def on_func(label):
    func_idx[0] = next(i for i, (n, _) in enumerate(FUNCTIONS) if n == label)
    redraw()

def toggle_newton(_):
    show_newton[0] = not show_newton[0]
    state = "ON" if show_newton[0] else "OFF"
    btn_newton.label.set_text(f"Newton step : {state}")
    btn_newton.label.set_color(mauve if show_newton[0] else overlay0)
    redraw()

order_radio.on_clicked(on_order)
func_radio.on_clicked(on_func)
btn_newton.on_clicked(toggle_newton)

for sl in x0_sliders.values():
    sl.on_changed(lambda _: redraw())

redraw()
plt.show()
