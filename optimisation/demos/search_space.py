import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    red, yellow,
    overlay0, surface0, surface1, base, text_col,
)

# ── Parametric loss surface ───────────────────────────────────────────────────
#
#   L(θ₁,θ₂) = basin·(θ₁²+θ₂²)                 ← global bowl
#             + roughness·sin(freq·θ₁)·cos(freq·θ₂)  ← local optima
#             + tilt_x·θ₁ + tilt_y·θ₂            ← asymmetry / ill-conditioning
#
RANGE = 3.5

def landscape(X, Y, basin, roughness, freq, tilt_x, tilt_y):
    return (basin * (X**2 + Y**2)
            + roughness * np.sin(freq * X) * np.cos(freq * Y)
            + tilt_x * X + tilt_y * Y)

def gradient(x, y, basin, roughness, freq, tilt_x, tilt_y):
    dx = 2*basin*x + roughness*freq*np.cos(freq*x)*np.cos(freq*y) + tilt_x
    dy = 2*basin*y - roughness*freq*np.sin(freq*x)*np.sin(freq*y) + tilt_y
    return np.array([dx, dy])

# ── Gradient descent simulator ─────────────────────────────────────────────────
def run_gd(start_x, start_y, params, lr=0.05, steps=300):
    basin, roughness, freq, tilt_x, tilt_y = params
    path = [(start_x, start_y)]
    x, y = float(start_x), float(start_y)
    for _ in range(steps):
        g = gradient(x, y, basin, roughness, freq, tilt_x, tilt_y)
        if np.linalg.norm(g) < 1e-5:
            break
        x = np.clip(x - lr * g[0], -RANGE, RANGE)
        y = np.clip(y - lr * g[1], -RANGE, RANGE)
        path.append((x, y))
    return np.array(path)

# ── Surface grid ──────────────────────────────────────────────────────────────
N  = 70
xs = np.linspace(-RANGE, RANGE, N)
ys = np.linspace(-RANGE, RANGE, N)
X, Y = np.meshgrid(xs, ys)

# ── Custom colourmap ──────────────────────────────────────────────────────────
loss_cmap = LinearSegmentedColormap.from_list(
    "mocha_loss", [sky, teal, green, peach, flamingo, red], N=256
)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 9))
fig.patch.set_facecolor(base)
fig.suptitle("Search Space Manifold  —  Loss Landscape Explorer",
             fontsize=13, fontweight="bold")

ax3d = fig.add_axes([0.01, 0.09, 0.54, 0.87], projection="3d")

# ── 3D styling ────────────────────────────────────────────────────────────────
def style_3d(ax):
    ax.set_facecolor(base)
    pane = (*mpl.colors.to_rgb(base), 0.92)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    ax.set_xlabel("θ₁  (param 1)", color=overlay0, labelpad=8)
    ax.set_ylabel("θ₂  (param 2)", color=overlay0, labelpad=8)
    ax.set_zlabel("Loss",          color=overlay0, labelpad=8)

# ── Core draw routine ─────────────────────────────────────────────────────────
def redraw(params, agent_xy, show_grad, show_gd):
    basin, roughness, freq, tilt_x, tilt_y = params
    ax = ax3d
    ax.cla()
    style_3d(ax)

    Z   = landscape(X, Y, *params)
    z_lo = Z.min();  z_hi = Z.max()

    ax.plot_surface(X, Y, Z, cmap=loss_cmap, alpha=0.80,
                    linewidth=0, antialiased=True, rcount=55, ccount=55)

    floor = z_lo - 0.45 * (z_hi - z_lo + 0.1)
    ax.contourf(X, Y, Z, zdir="z", offset=floor,
                cmap=loss_cmap, alpha=0.30, levels=18)
    ax.set_zlim(floor, z_hi + 0.4)

    ax_x, ax_y = agent_xy
    ax_z = landscape(ax_x, ax_y, *params)
    ax.scatter([ax_x], [ax_y], [ax_z], color=yellow, s=100,
               zorder=10, depthshade=False, label=f"Agent  L={ax_z:.3f}")

    ax.plot([ax_x, ax_x], [ax_y, ax_y], [floor, ax_z],
            color=yellow, lw=0.8, alpha=0.45, ls=":")

    if show_grad:
        g      = gradient(ax_x, ax_y, *params)
        g_norm = g / (np.linalg.norm(g) + 1e-9)
        scale  = 0.65
        ax.quiver(ax_x, ax_y, ax_z,
                  -g_norm[0]*scale, -g_norm[1]*scale, 0,
                  color=flamingo, linewidth=2.2, arrow_length_ratio=0.28,
                  label="−∇L  (descent dir.)")

    if show_gd:
        path = run_gd(ax_x, ax_y, params)
        pz   = landscape(path[:, 0], path[:, 1], *params)
        ax.plot(path[:, 0], path[:, 1], pz + 0.04,
                color=mauve, lw=1.6, alpha=0.9, label="GD trajectory")
        ax.scatter([path[-1, 0]], [path[-1, 1]], [pz[-1] + 0.04],
                   color=green, s=70, depthshade=False,
                   label=f"Converged  L={pz[-1]:.3f}")

    idx  = np.unravel_index(Z.argmin(), Z.shape)
    gx, gy, gz = X[idx], Y[idx], Z[idx]
    ax.scatter([gx], [gy], [gz], color=sky, s=90, marker="*",
               depthshade=False, label=f"Grid min ({gx:.1f},{gy:.1f})")

    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.25)

    g_mag      = np.linalg.norm(gradient(ax_x, ax_y, *params))
    is_convex  = roughness < 0.15 and basin > 0.3
    is_nearly  = roughness < 0.55 and basin > 0.1
    conv_label = ("Convex ✓"        if is_convex else
                  "Nearly convex ~" if is_nearly else
                  "Non-convex  ✗")
    conv_color = green if is_convex else (peach if is_nearly else flamingo)

    info_loss.set_text(f"Agent loss: {ax_z:+.4f}   |∇L|: {g_mag:.4f}")
    info_conv.set_text(f"Landscape: {conv_label}")
    info_conv.set_color(conv_color)
    fig.canvas.draw_idle()

# ── Slider definitions ────────────────────────────────────────────────────────
LEFT = 0.61
SW   = 0.33
SH   = 0.028

slider_defs = [
    ("Basin  (convexity)",    "basin",    0.00,  2.0,  0.50,  teal    ),
    ("Roughness",             "rough",    0.00,  2.0,  0.80,  peach   ),
    ("Frequency",             "freq",     0.50,  6.0,  2.00,  lavender),
    ("Tilt  θ₁",              "tilt_x",  -1.50,  1.5,  0.00,  flamingo),
    ("Tilt  θ₂",              "tilt_y",  -1.50,  1.5,  0.00,  flamingo),
    ("Agent  θ₁",             "agent_x", -RANGE, RANGE, 2.00, yellow  ),
    ("Agent  θ₂",             "agent_y", -RANGE, RANGE, 2.00, yellow  ),
]

Y_TOP = 0.88;  Y_BOT = 0.18
n_sl  = len(slider_defs)
y_gap = (Y_TOP - Y_BOT) / (n_sl - 1)

sliders = {}
for i, (label, key, vmin, vmax, vinit, color) in enumerate(slider_defs):
    sy = Y_TOP - i * y_gap

    if i == 5:
        fig.add_axes([LEFT, sy + SH + 0.026, SW, 0.001]).set_visible(False)
        fig.text(LEFT + SW/2, sy + SH + 0.030,
                 "── Agent Position ──",
                 ha="center", fontsize=8, color=yellow, fontstyle="italic")

    fig.text(LEFT - 0.010, sy + SH/2, label,
             ha="right", va="center", fontsize=8.5, color=text_col)

    sax = fig.add_axes([LEFT, sy, SW, SH])
    sax.set_facecolor(surface0)
    sl  = Slider(ax=sax, label="", valmin=vmin, valmax=vmax, valinit=vinit,
                 color=color, track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(8)
    sliders[key] = sl

fig.text(LEFT + SW/2, 0.945, "── Surface Parameters ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

# ── Toggle buttons ─────────────────────────────────────────────────────────────
show_gd_state   = [True]
show_grad_state = [True]

bax_gd   = fig.add_axes([LEFT,          0.06, 0.155, 0.042])
bax_grad = fig.add_axes([LEFT + 0.175,  0.06, 0.155, 0.042])
for bax in (bax_gd, bax_grad):
    bax.set_facecolor(surface0)

btn_gd   = Button(bax_gd,   "GD Path : ON",  color=surface0, hovercolor=surface1)
btn_grad = Button(bax_grad, "Gradient : ON", color=surface0, hovercolor=surface1)
for btn in (btn_gd, btn_grad):
    btn.label.set_color(mauve)
    btn.label.set_fontsize(8.5)
    btn.label.set_fontweight("bold")

def toggle_gd(_):
    show_gd_state[0] = not show_gd_state[0]
    state = "ON" if show_gd_state[0] else "OFF"
    btn_gd.label.set_text(f"GD Path : {state}")
    btn_gd.label.set_color(mauve if show_gd_state[0] else overlay0)
    update(None)

def toggle_grad(_):
    show_grad_state[0] = not show_grad_state[0]
    state = "ON" if show_grad_state[0] else "OFF"
    btn_grad.label.set_text(f"Gradient : {state}")
    btn_grad.label.set_color(mauve if show_grad_state[0] else overlay0)
    update(None)

btn_gd.on_clicked(toggle_gd)
btn_grad.on_clicked(toggle_grad)

# ── Info text ─────────────────────────────────────────────────────────────────
cx = LEFT + SW / 2
info_loss = fig.text(cx, 0.032, "", ha="center", va="center",
                     fontsize=8.5, color=text_col, fontfamily="monospace")
info_conv = fig.text(cx, 0.010, "", ha="center", va="center",
                     fontsize=8.5, fontweight="bold", fontfamily="monospace")

# ── Master update callback ────────────────────────────────────────────────────
def update(_):
    params   = (sliders["basin"].val, sliders["rough"].val,
                sliders["freq"].val,  sliders["tilt_x"].val, sliders["tilt_y"].val)
    agent_xy = (sliders["agent_x"].val, sliders["agent_y"].val)
    redraw(params, agent_xy, show_grad_state[0], show_gd_state[0])

for sl in sliders.values():
    sl.on_changed(update)

update(None)
plt.show()
