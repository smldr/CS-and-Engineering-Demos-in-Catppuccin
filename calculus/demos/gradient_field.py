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
    overlay0, surface0, surface1, base, text_col, PANE,
)

# ── Grid constants ────────────────────────────────────────────────────────────
RANGE = 3.0
N = 70
QUIVER_STEP = 3  # denser grid à la 3B1B

xs = np.linspace(-RANGE, RANGE, N)
ys = np.linspace(-RANGE, RANGE, N)
X, Y = np.meshgrid(xs, ys)


# ── Surface functions & analytic gradients ────────────────────────────────────
def _f0(X, Y, a, b, c, freq):
    """Quadratic + sinusoidal: a(x²+y²) + bxy + c·sin(fx)cos(fy)"""
    return a * (X**2 + Y**2) + b * X * Y + c * np.sin(freq * X) * np.cos(freq * Y)

def _g0(X, Y, a, b, c, freq):
    dZdx = 2*a*X + b*Y + c*freq*np.cos(freq*X)*np.cos(freq*Y)
    dZdy = 2*a*Y + b*X - c*freq*np.sin(freq*X)*np.sin(freq*Y)
    return dZdx, dZdy

def _f1(X, Y, a, b, c, freq):
    """Saddle: a·x² − b·y² + c·sin(freq·x·y)"""
    return a * X**2 - b * Y**2 + c * np.sin(freq * X * Y)

def _g1(X, Y, a, b, c, freq):
    dZdx = 2*a*X + c*freq*Y*np.cos(freq*X*Y)
    dZdy = -2*b*Y + c*freq*X*np.cos(freq*X*Y)
    return dZdx, dZdy

def _f2(X, Y, a, b, c, freq):
    """Rosenbrock-like: a·(y − x²)² + b·(1 − x)² scaled by c"""
    return c * (a * (Y - X**2)**2 + b * (1 - X)**2)

def _g2(X, Y, a, b, c, freq):
    dZdx = c * (-4*a*X*(Y - X**2) - 2*b*(1 - X))
    dZdy = c * (2*a*(Y - X**2))
    return dZdx, dZdy

def _f3(X, Y, a, b, c, freq):
    """Ripple: a·cos(freq·r)/( b·r + 1 ) where r = √(x²+y²)"""
    R = np.sqrt(X**2 + Y**2)
    return a * np.cos(freq * R) / (b * R + 1) + c * 0.1 * (X + Y)

def _g3(X, Y, a, b, c, freq):
    R = np.sqrt(X**2 + Y**2) + 1e-10
    denom = (b * R + 1)
    dRdx = X / R
    dRdy = Y / R
    # d/dx [a cos(fr)/(bR+1)] = a[-f sin(fr)(bR+1) - cos(fr)·b] / (bR+1)² · dR/dx
    num = -a * freq * np.sin(freq*R) * denom - a * np.cos(freq*R) * b
    dfdR = num / denom**2
    dZdx = dfdR * dRdx + c * 0.1
    dZdy = dfdR * dRdy + c * 0.1
    return dZdx, dZdy

def _f4(X, Y, a, b, c, freq):
    """Gaussian wells: −a·exp(−(x−1)²−y²) − b·exp(−(x+1)²−y²) + c·(x²+y²)/freq"""
    return (-a * np.exp(-(X-1)**2 - Y**2)
            - b * np.exp(-(X+1)**2 - Y**2)
            + c * (X**2 + Y**2) / freq)

def _g4(X, Y, a, b, c, freq):
    e1 = np.exp(-(X-1)**2 - Y**2)
    e2 = np.exp(-(X+1)**2 - Y**2)
    dZdx = 2*a*(X-1)*e1 + 2*b*(X+1)*e2 + 2*c*X/freq
    dZdy = 2*a*Y*e1 + 2*b*Y*e2 + 2*c*Y/freq
    return dZdx, dZdy

FUNCTIONS = [
    ("Quadratic + Sin",  _f0, _g0),
    ("Saddle + Sin",     _f1, _g1),
    ("Rosenbrock-like",  _f2, _g2),
    ("Radial Ripple",    _f3, _g3),
    ("Gaussian Wells",   _f4, _g4),
]
func_state = [0]

# Per-function slider labels: [a_label, b_label, c_label, freq_label]
_SLIDER_LABELS = [
    ["a  (quadratic)", "b  (cross-term)", "c  (sin amp)",   "freq  (osc)"],
    ["a  (x² coeff)",  "b  (y² coeff)",   "c  (sin amp)",   "freq  (xy osc)"],
    ["a  (valley)",    "b  (offset)",     "c  (scale)",     "freq  (unused)"],
    ["a  (amplitude)", "b  (decay)",      "c  (tilt)",      "freq  (ripple)"],
    ["a  (well₁ depth)","b  (well₂ depth)","c  (bowl)",     "freq  (bowl width)"],
]


def surface_fn(X, Y, a, b, c, freq):
    return FUNCTIONS[func_state[0]][1](X, Y, a, b, c, freq)


def gradient_fn(X, Y, a, b, c, freq):
    return FUNCTIONS[func_state[0]][2](X, Y, a, b, c, freq)


def quiver_grid(X, Y, dZdx, dZdy, step, normalise):
    """Subsample gradient arrays; return unit vectors + magnitude for 3B1B style."""
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    Uq = dZdx[::step, ::step].copy()
    Vq = dZdy[::step, ::step].copy()

    mag = np.sqrt(Uq**2 + Vq**2)
    if normalise:
        safe_mag = np.where(mag < 1e-10, 1.0, mag)
        Uq = Uq / safe_mag
        Vq = Vq / safe_mag
    else:
        # Clamp to max magnitude so arrows never exceed cell spacing
        max_mag = mag.max() + 1e-10
        Uq = Uq / max_mag
        Vq = Vq / max_mag

    return Xq, Yq, Uq, Vq, mag


# ── Custom colourmaps ─────────────────────────────────────────────────────────
surface_cmap = LinearSegmentedColormap.from_list(
    "mocha_gradient", [sky, teal, green, peach, flamingo, red], N=256
)
# 3B1B-style: low magnitude → cool (teal/sky), high magnitude → warm (peach/flamingo)
field_cmap = LinearSegmentedColormap.from_list(
    "mocha_field", [teal, sky, green, yellow, peach, flamingo], N=256
)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 9))
fig.patch.set_facecolor(base)
fig.suptitle("Gradient Vector Field  —  ∇f Visualiser",
             fontsize=13, fontweight="bold")

ax3d = fig.add_axes([0.01, 0.09, 0.54, 0.87], projection="3d")


# ── 3D styling ────────────────────────────────────────────────────────────────
def style_3d(ax):
    ax.set_facecolor(base)
    ax.xaxis.set_pane_color(PANE)
    ax.yaxis.set_pane_color(PANE)
    ax.zaxis.set_pane_color(PANE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    ax.set_xlabel("x", color=overlay0, labelpad=8)
    ax.set_ylabel("y", color=overlay0, labelpad=8)
    ax.set_zlabel("f(x, y)", color=overlay0, labelpad=8)


# ── Slider panel ──────────────────────────────────────────────────────────────
LEFT = 0.61
SW = 0.33
SH = 0.028

slider_defs = [
    ("a  (quadratic)", "a",    0.1, 3.0, 1.0,  teal),
    ("b  (cross-term)","b",   -2.0, 2.0, 0.0,  peach),
    ("c  (sin amp)",   "c",    0.0, 2.0, 0.5,  lavender),
    ("freq  (osc)",    "freq", 0.5, 6.0, 2.0,  flamingo),
]

Y_TOP = 0.82
Y_BOT = 0.45
n_sl = len(slider_defs)
y_gap = (Y_TOP - Y_BOT) / max(n_sl - 1, 1)

sliders = {}
slider_labels = []
for i, (label, key, vmin, vmax, vinit, color) in enumerate(slider_defs):
    sy = Y_TOP - i * y_gap
    lbl = fig.text(LEFT - 0.010, sy + SH / 2, label,
                   ha="right", va="center", fontsize=8.5, color=text_col)
    slider_labels.append(lbl)
    sax = fig.add_axes([LEFT, sy, SW, SH])
    sax.set_facecolor(surface0)
    sl = Slider(ax=sax, label="", valmin=vmin, valmax=vmax, valinit=vinit,
                color=color, track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(8)
    sliders[key] = sl

fig.text(LEFT + SW / 2, 0.90, "── Surface Parameters ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

# ── Toggle buttons ────────────────────────────────────────────────────────────
show_grad_state = [True]
normalise_state = [True]
# 0 = full ∇f, 1 = ∂f/∂x only, 2 = ∂f/∂y only
component_state = [0]
_COMP_LABELS = ["∇f  (full)", "∂f/∂x only", "∂f/∂y only"]

bax_func = fig.add_axes([LEFT, 0.37, 0.33, 0.042])
bax_grad = fig.add_axes([LEFT, 0.30, 0.155, 0.042])
bax_norm = fig.add_axes([LEFT + 0.175, 0.30, 0.155, 0.042])
bax_comp = fig.add_axes([LEFT, 0.24, 0.33, 0.042])
for bax in (bax_func, bax_grad, bax_norm, bax_comp):
    bax.set_facecolor(surface0)

btn_func = Button(bax_func, f"f(x,y) : {FUNCTIONS[0][0]}", color=surface0, hovercolor=surface1)
btn_grad = Button(bax_grad, "Gradient : ON", color=surface0, hovercolor=surface1)
btn_norm = Button(bax_norm, "Normalise : ON", color=surface0, hovercolor=surface1)
btn_comp = Button(bax_comp, f"View : {_COMP_LABELS[0]}", color=surface0, hovercolor=surface1)
for btn in (btn_func, btn_grad, btn_norm, btn_comp):
    btn.label.set_color(mauve)
    btn.label.set_fontsize(8.5)
    btn.label.set_fontweight("bold")


def toggle_grad(_):
    show_grad_state[0] = not show_grad_state[0]
    state = "ON" if show_grad_state[0] else "OFF"
    btn_grad.label.set_text(f"Gradient : {state}")
    btn_grad.label.set_color(mauve if show_grad_state[0] else overlay0)
    update(None)


def toggle_norm(_):
    normalise_state[0] = not normalise_state[0]
    state = "ON" if normalise_state[0] else "OFF"
    btn_norm.label.set_text(f"Normalise : {state}")
    btn_norm.label.set_color(mauve if normalise_state[0] else overlay0)
    update(None)


def toggle_comp(_):
    component_state[0] = (component_state[0] + 1) % 3
    btn_comp.label.set_text(f"View : {_COMP_LABELS[component_state[0]]}")
    update(None)


def toggle_func(_):
    func_state[0] = (func_state[0] + 1) % len(FUNCTIONS)
    btn_func.label.set_text(f"f(x,y) : {FUNCTIONS[func_state[0]][0]}")
    for lbl, txt in zip(slider_labels, _SLIDER_LABELS[func_state[0]]):
        lbl.set_text(txt)
    update(None)


btn_func.on_clicked(toggle_func)
btn_grad.on_clicked(toggle_grad)
btn_norm.on_clicked(toggle_norm)
btn_comp.on_clicked(toggle_comp)

# ── Info text ─────────────────────────────────────────────────────────────────
cx = LEFT + SW / 2
info_text = fig.text(cx, 0.16, "", ha="center", va="center",
                     fontsize=8.5, color=text_col, fontfamily="monospace")


# ── Redraw routine ────────────────────────────────────────────────────────────
def redraw():
    a = sliders["a"].val
    b = sliders["b"].val
    c = sliders["c"].val
    freq = sliders["freq"].val

    ax3d.cla()
    style_3d(ax3d)

    Z = surface_fn(X, Y, a, b, c, freq)
    z_lo, z_hi = Z.min(), Z.max()
    if z_hi - z_lo < 0.01:
        z_hi = z_lo + 0.01

    ax3d.plot_surface(X, Y, Z, cmap=surface_cmap, alpha=0.78,
                      linewidth=0, antialiased=True, rcount=55, ccount=55)

    floor = z_lo - 0.45 * (z_hi - z_lo + 0.1)
    ax3d.contourf(X, Y, Z, zdir="z", offset=floor,
                  cmap=surface_cmap, alpha=0.30, levels=18)

    if show_grad_state[0]:
        dZdx, dZdy = gradient_fn(X, Y, a, b, c, freq)
        # Apply component filter
        comp = component_state[0]
        if comp == 1:
            dZdy = np.zeros_like(dZdy)
        elif comp == 2:
            dZdx = np.zeros_like(dZdx)
        Xq, Yq, Uq, Vq, mag = quiver_grid(X, Y, dZdx, dZdy, QUIVER_STEP, normalise_state[0])
        Wq = np.zeros_like(Uq)
        floor_arr = np.full_like(Xq, floor)

        # 3B1B style: color each arrow by magnitude, uniform length
        mag_flat = mag.ravel()
        mag_norm = mag_flat / (mag_flat.max() + 1e-10)
        colors = field_cmap(mag_norm)
        # Fade out near-zero magnitude arrows
        colors[:, 3] = np.clip(mag_norm * 2.5, 0.15, 0.95)

        ax3d.quiver(Xq, Yq, floor_arr, Uq, Vq, Wq,
                    color=colors.tolist(), linewidth=1.0,
                    arrow_length_ratio=0.18, length=0.20)

    ax3d.set_zlim(floor, z_hi + 0.4)

    # Gradient stats
    dZdx_full, dZdy_full = gradient_fn(X, Y, a, b, c, freq)
    mags = np.sqrt(dZdx_full**2 + dZdy_full**2)
    info_text.set_text(f"max |∇f| = {mags.max():.3f}   mean |∇f| = {mags.mean():.3f}")

    fig.canvas.draw_idle()


# ── Master update callback ────────────────────────────────────────────────────
def update(_):
    redraw()

for sl in sliders.values():
    sl.on_changed(update)

update(None)
plt.show()
