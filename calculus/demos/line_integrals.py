import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.collections as mc
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    red, overlay0, surface0, surface1, base, text_col, PANE,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_CURVE = 300
N_GRID  = 55
RANGE   = 2.5

# ── Curve definitions  (x, y, dx/dt, dy/dt) ──────────────────────────────────
def _circle(t):
    return 1.8*np.cos(t), 1.8*np.sin(t), -1.8*np.sin(t), 1.8*np.cos(t)

def _ellipse(t):
    return 2.0*np.cos(t), 1.2*np.sin(t), -2.0*np.sin(t), 1.2*np.cos(t)

def _fig8(t):
    x  =  2.0*np.sin(t)
    y  =  1.2*np.sin(t)*np.cos(t)
    dx =  2.0*np.cos(t)
    dy =  1.2*(np.cos(t)**2 - np.sin(t)**2)
    return x, y, dx, dy

def _spiral(t):
    r  = 0.4 + t / (2*np.pi) * 1.8
    dr = 1.8 / (2*np.pi)
    x  = r * np.cos(t)
    y  = r * np.sin(t)
    dx = dr*np.cos(t) - r*np.sin(t)
    dy = dr*np.sin(t) + r*np.cos(t)
    return x, y, dx, dy

CURVES = [
    ("Circle",   _circle,  0.0, 2*np.pi),
    ("Ellipse",  _ellipse, 0.0, 2*np.pi),
    ("Figure-8", _fig8,    0.0, 2*np.pi),
    ("Spiral",   _spiral,  0.0, 2*np.pi),
]

# ── Scalar fields  (name, f, ∇f) ─────────────────────────────────────────────
# Both panels share the same f.  The right panel always shows F = ∇f.

def _f_bowl(x, y):      return 0.4*(x**2 + y**2) + 0.2
def _g_bowl(x, y):      return 0.8*x, 0.8*y

def _f_wave(x, y):      return 1.0 + 0.8*np.sin(1.4*x)*np.cos(1.4*y)
def _g_wave(x, y):      return ( 1.12*np.cos(1.4*x)*np.cos(1.4*y),
                                 -1.12*np.sin(1.4*x)*np.sin(1.4*y) )

def _f_gaussian(x, y):  return 2.2*np.exp(-0.5*(x**2 + y**2)) + 0.1
def _g_gaussian(x, y):
    e = 2.2*np.exp(-0.5*(x**2 + y**2))
    return -x*e, -y*e

def _f_saddle(x, y):    return 1.2 + 0.4*(x**2 - y**2)
def _g_saddle(x, y):    return 0.8*x, -0.8*y

SCALAR_FIELDS = [
    ("Bowl  x²+y²",    _f_bowl,     _g_bowl),
    ("Waves  sin·cos", _f_wave,     _g_wave),
    ("Gaussian peak",  _f_gaussian, _g_gaussian),
    ("Saddle  x²−y²",  _f_saddle,   _g_saddle),
]

# ── State ─────────────────────────────────────────────────────────────────────
curve_state  = [0]
sfield_state = [0]

# ── Grid ──────────────────────────────────────────────────────────────────────
xs = np.linspace(-RANGE, RANGE, N_GRID)
ys = np.linspace(-RANGE, RANGE, N_GRID)
X, Y = np.meshgrid(xs, ys)

# ── Colourmaps ────────────────────────────────────────────────────────────────
surface_cmap = LinearSegmentedColormap.from_list(
    "mocha_surf", [sky, teal, green, peach, flamingo, red], N=256
)
# negative=red (walking downhill), positive=green (walking uphill)
work_cmap = LinearSegmentedColormap.from_list(
    "mocha_work", [red, flamingo, surface1, teal, green], N=256
)

# ── Figure & axes ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 9.5))
fig.patch.set_facecolor(base)
fig.suptitle(
    "Line Integrals  —  Scalar ∫_C f ds  (curtain area)   vs   Vector ∫_C ∇f·dr  (FToLI: = f(end)−f(start))",
    fontsize=12, fontweight="bold", color=text_col,
)

ax3d  = fig.add_axes([0.01, 0.08, 0.44, 0.86], projection="3d")
ax2d  = fig.add_axes([0.47, 0.36, 0.30, 0.57])
axbar = fig.add_axes([0.47, 0.08, 0.30, 0.24])

for ax in (ax2d, axbar):
    ax.set_facecolor(surface0)
    ax.tick_params(colors=overlay0, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(overlay0)


def style_3d(ax):
    ax.set_facecolor(base)
    ax.xaxis.set_pane_color(PANE)
    ax.yaxis.set_pane_color(PANE)
    ax.zaxis.set_pane_color(PANE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=7)
    ax.set_xlabel("x", color=overlay0, labelpad=5)
    ax.set_ylabel("y", color=overlay0, labelpad=5)
    ax.set_zlabel("f", color=overlay0, labelpad=5)


# ── Controls panel ────────────────────────────────────────────────────────────
LEFT, SW = 0.79, 0.19

fig.text(LEFT + SW/2, 0.945, "── Controls ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

bax_curve  = fig.add_axes([LEFT, 0.885, SW, 0.042])
bax_sfield = fig.add_axes([LEFT, 0.825, SW, 0.042])
for bax in (bax_curve, bax_sfield):
    bax.set_facecolor(surface0)

btn_curve  = Button(bax_curve,  f"Curve : {CURVES[0][0]}",        color=surface0, hovercolor=surface1)
btn_sfield = Button(bax_sfield, f"f(x,y): {SCALAR_FIELDS[0][0]}", color=surface0, hovercolor=surface1)

for btn in (btn_curve, btn_sfield):
    btn.label.set_color(mauve)
    btn.label.set_fontsize(7.5)
    btn.label.set_fontweight("bold")

# Integral value readouts
info_scalar = fig.text(LEFT + SW/2, 0.765, "", ha="center", va="top",
                        fontsize=8.5, color=teal,  fontfamily="monospace")
info_vector = fig.text(LEFT + SW/2, 0.720, "", ha="center", va="top",
                        fontsize=8.5, color=peach, fontfamily="monospace")
info_exact  = fig.text(LEFT + SW/2, 0.678, "", ha="center", va="top",
                        fontsize=8.5, color=green, fontfamily="monospace")

# Formula reference block
fig.text(
    LEFT + SW/2, 0.635,
    "Scalar:\n"
    "  ∫_C f ds\n"
    "  = ∫ f(r(t)) |r'(t)| dt\n"
    "  → curtain area under C\n"
    "\n"
    "Vector  (F = ∇f):\n"
    "  ∫_C ∇f·dr\n"
    "  = ∫ ∇f(r(t))·r'(t) dt\n"
    "  = f(end) − f(start)\n"
    "  → Fundamental Theorem",
    ha="center", va="top", fontsize=7.5, color=text_col,
    fontfamily="monospace", linespacing=1.55,
)

# Path colour legend
fig.text(
    LEFT + SW/2, 0.30,
    "Path colour (right panel):\n"
    "  green  walking uphill\n"
    "         (∇f pushes with you)\n"
    "  red    walking downhill\n"
    "         (∇f pushes against you)",
    ha="center", va="top", fontsize=7.5, color=text_col,
    fontfamily="monospace", linespacing=1.5,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_curve():
    _, fn, t0, t1 = CURVES[curve_state[0]]
    t = np.linspace(t0, t1, N_CURVE)
    x, y, dx, dy = fn(t)
    return t, x, y, dx, dy


def draw_scalar(t, x, y, dx, dy):
    ax3d.cla()
    style_3d(ax3d)
    ax3d.set_title("Scalar Line Integral  ∫_C f ds", color=mauve, fontsize=9, pad=2)

    f_fn = SCALAR_FIELDS[sfield_state[0]][1]
    Z    = f_fn(X, Y)
    z_lo, z_hi = Z.min(), Z.max()
    if z_hi - z_lo < 0.01:
        z_hi = z_lo + 0.01

    ax3d.plot_surface(X, Y, Z, cmap=surface_cmap, alpha=0.60,
                      linewidth=0, antialiased=True, rcount=40, ccount=40)

    floor = z_lo - 0.30*(z_hi - z_lo)
    ax3d.contourf(X, Y, Z, zdir="z", offset=floor,
                  cmap=surface_cmap, alpha=0.22, levels=10)

    fz = f_fn(x, y)

    # Curve projected onto the floor
    ax3d.plot(x, y, np.full_like(x, floor), color=lavender, linewidth=1.0, alpha=0.5)

    # Curtain: quad strip from floor to surface
    verts = [
        [[x[i],   y[i],   floor],
         [x[i+1], y[i+1], floor],
         [x[i+1], y[i+1], fz[i+1]],
         [x[i],   y[i],   fz[i]]]
        for i in range(len(x) - 1)
    ]
    fz_mid   = 0.5*(fz[:-1] + fz[1:])
    fz_norm  = (fz_mid - z_lo) / (z_hi - z_lo + 1e-10)
    face_col = surface_cmap(fz_norm)
    face_col[:, 3] = 0.72
    ax3d.add_collection3d(Poly3DCollection(verts, facecolors=face_col, linewidths=0))

    # Top edge of curtain (curve lifted to the surface)
    ax3d.plot(x, y, fz, color=peach, linewidth=2.0)

    # Vertical stitching lines to make the curtain tangible
    for i in range(0, len(x), 30):
        ax3d.plot([x[i], x[i]], [y[i], y[i]], [floor, fz[i]],
                  color=peach, linewidth=0.5, alpha=0.4)

    ax3d.set_zlim(floor, z_hi + 0.2)

    # Numerical value: ∫ f(r(t)) |r'(t)| dt  (Riemann sum)
    ds  = np.sqrt(dx**2 + dy**2)
    dt  = (t[-1] - t[0]) / (N_CURVE - 1)
    val = float(np.sum(fz * ds) * dt)
    info_scalar.set_text(f"∫_C f ds  ≈ {val:.4f}")


def draw_vector(t, x, y, dx, dy):
    ax2d.cla()
    ax2d.set_facecolor(surface0)
    ax2d.set_xlim(-RANGE, RANGE)
    ax2d.set_ylim(-RANGE, RANGE)
    ax2d.set_aspect("equal")
    ax2d.set_title("∫_C ∇f·dr   [F = ∇f, same f as left]", color=mauve, fontsize=9, pad=2)
    ax2d.tick_params(colors=overlay0, labelsize=7)
    for sp in ax2d.spines.values():
        sp.set_color(overlay0)

    f_fn   = SCALAR_FIELDS[sfield_state[0]][1]
    grad_fn = SCALAR_FIELDS[sfield_state[0]][2]

    # Background quiver grid showing ∇f
    step = 5
    Xq, Yq   = X[::step, ::step], Y[::step, ::step]
    Gxq, Gyq = grad_fn(Xq, Yq)
    mag      = np.sqrt(Gxq**2 + Gyq**2)
    max_mag  = mag.max() + 1e-10
    mag_n    = (mag / max_mag).ravel()
    qcmap    = LinearSegmentedColormap.from_list("qc", [teal, sky, peach], N=256)
    qcol     = qcmap(mag_n)
    qcol[:, 3] = np.clip(mag_n * 1.2 + 0.3, 0.3, 0.85)
    ax2d.quiver(Xq, Yq, Gxq/max_mag, Gyq/max_mag,
                color=qcol, scale=20, width=0.003, headwidth=4)

    # ∇f(r(t))·r'(t) — the integrand of the vector line integral
    Gx, Gy = grad_fn(x, y)
    dot    = Gx*dx + Gy*dy

    max_dot  = np.abs(dot).max() + 1e-10
    dot_norm = 0.5*(dot/max_dot + 1)           # map to [0,1] for work_cmap

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    dot_seg  = 0.5*(dot_norm[:-1] + dot_norm[1:])

    lc = mc.LineCollection(segments, cmap=work_cmap,
                            norm=Normalize(0, 1), linewidth=2.5, zorder=5)
    lc.set_array(dot_seg)
    ax2d.add_collection(lc)

    # Travel-direction arrows along the path
    for i in np.linspace(0, len(t)-2, 8, dtype=int):
        ax2d.annotate("", xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                      arrowprops=dict(arrowstyle="-|>", color=lavender,
                                     lw=0.9, mutation_scale=9))

    ax2d.plot(x[0],  y[0],  "o", color=green,   markersize=6, zorder=6, label="start")
    ax2d.plot(x[-1], y[-1], "s", color=flamingo, markersize=6, zorder=6, label="end")
    ax2d.legend(fontsize=7, facecolor=surface1, labelcolor=text_col,
                loc="lower right", framealpha=0.7)

    # ── Accumulated-work plot ──────────────────────────────────────────────────
    axbar.cla()
    axbar.set_facecolor(surface0)
    axbar.tick_params(colors=overlay0, labelsize=7)
    for sp in axbar.spines.values():
        sp.set_color(overlay0)
    axbar.set_title("Accumulated Work  W(t) = ∫₀ᵗ ∇f·r′ dτ", color=mauve, fontsize=8, pad=2)
    axbar.set_xlabel("t / T", fontsize=7, color=overlay0)
    axbar.set_ylabel("W(t)", fontsize=7, color=overlay0)
    axbar.axhline(0, color=overlay0, linewidth=0.6, linestyle="--")

    dt_val   = (t[-1] - t[0]) / (N_CURVE - 1)
    work_cum = np.cumsum(dot * dt_val)
    t_norm   = (t - t[0]) / (t[-1] - t[0])

    axbar.fill_between(t_norm, 0, work_cum,
                       where=(work_cum >= 0), color=teal, alpha=0.35, interpolate=True)
    axbar.fill_between(t_norm, 0, work_cum,
                       where=(work_cum < 0),  color=red,  alpha=0.35, interpolate=True)
    axbar.plot(t_norm, work_cum, color=peach, linewidth=1.5, label="computed")

    # Exact value via Fundamental Theorem of Line Integrals: f(end) − f(start)
    f_start = float(f_fn(x[0],  y[0]))
    f_end   = float(f_fn(x[-1], y[-1]))
    exact   = f_end - f_start

    axbar.axhline(exact, color=green, linewidth=1.2, linestyle="--", alpha=0.85,
                  label=f"f(end)−f(start) = {exact:.4f}")
    axbar.legend(fontsize=7, facecolor=surface1, labelcolor=text_col,
                 loc="upper left", framealpha=0.7)

    computed = float(work_cum[-1])
    info_vector.set_text(f"∫_C ∇f·dr ≈ {computed:.4f}")
    info_exact.set_text( f"f(end)−f(start) = {exact:.4f}")


# ── Master update ─────────────────────────────────────────────────────────────
def update(_):
    t, x, y, dx, dy = compute_curve()
    draw_scalar(t, x, y, dx, dy)
    draw_vector(t, x, y, dx, dy)
    fig.canvas.draw_idle()


def toggle_curve(_):
    curve_state[0] = (curve_state[0] + 1) % len(CURVES)
    btn_curve.label.set_text(f"Curve : {CURVES[curve_state[0]][0]}")
    update(None)


def toggle_sfield(_):
    sfield_state[0] = (sfield_state[0] + 1) % len(SCALAR_FIELDS)
    btn_sfield.label.set_text(f"f(x,y): {SCALAR_FIELDS[sfield_state[0]][0]}")
    update(None)


btn_curve.on_clicked(toggle_curve)
btn_sfield.on_clicked(toggle_sfield)

update(None)
plt.show()
