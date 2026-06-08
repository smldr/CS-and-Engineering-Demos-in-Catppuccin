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

# ── Contour grid ──────────────────────────────────────────────────────────────
LIMIT  = 2.8
NX     = 260
_xs    = np.linspace(-LIMIT, LIMIT, NX)
Xs, Ys = np.meshgrid(_xs, _xs)
XY     = np.stack([Xs.ravel(), Ys.ravel()])   # 2 × NX²

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9))
fig.suptitle("Change of Basis  A⁻¹MA  vs  Quadratic Form  AᵀMA",
             fontsize=13, fontweight="bold", color=text_col)
fig.patch.set_facecolor(base)

ax_M  = fig.add_axes([0.02, 0.12, 0.23, 0.80])
ax_CB = fig.add_axes([0.27, 0.12, 0.23, 0.80])
ax_QF = fig.add_axes([0.52, 0.12, 0.23, 0.80])

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
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 color=title_col, fontsize=10, fontweight="bold", pad=6)

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_contours(ax, Q_sym):
    """Level curves of x^T Q_sym x for a symmetric Q_sym."""
    Z = (XY * (Q_sym @ XY)).sum(axis=0).reshape(NX, NX)
    for c in [0.5, 1.0, 2.0, 4.0]:
        try:
            ax.contour(Xs, Ys, Z, levels=[c], colors=[mauve],
                       linewidths=[1.7 if c == 1.0 else 1.0], alpha=0.85)
        except Exception:
            pass
        try:
            ax.contour(Xs, Ys, Z, levels=[-c], colors=[flamingo],
                       linewidths=[1.7 if c == 1.0 else 1.0],
                       linestyles=["dashed"], alpha=0.75)
        except Exception:
            pass
    try:
        ax.contour(Xs, Ys, Z, levels=[0], colors=[overlay0],
                   linewidths=[0.8], linestyles=[":"], alpha=0.5)
    except Exception:
        pass


def draw_eigvecs(ax, eigvals, eigvecs):
    """Principal axis arrows scaled by 1/√|λ|.

    Same arrow length ↔ same eigenvalue — the key visual signal.
    """
    for i, (lam, col) in enumerate(zip(eigvals, [teal, peach])):
        v = eigvecs[:, i]
        v = v / (np.linalg.norm(v) + 1e-12)
        scale = min(1.0 / (np.sqrt(abs(lam)) + 1e-12), LIMIT * 0.85)
        ax.quiver(0, 0,  v[0] * scale,  v[1] * scale, color=col,
                  angles="xy", scale_units="xy", scale=1,
                  width=0.011, headwidth=4, alpha=0.95,
                  label=f"λ={lam:+.2f}")
        ax.quiver(0, 0, -v[0] * scale, -v[1] * scale, color=col,
                  angles="xy", scale_units="xy", scale=1,
                  width=0.011, headwidth=4, alpha=0.5)


def draw_panel(ax, Q_sym, Q_full, title, title_col, subtitle, eigvals, eigvecs):
    ax.cla()
    style_ax(ax, title, title_col, subtitle)
    draw_contours(ax, Q_sym)
    draw_eigvecs(ax, eigvals, eigvecs)
    ax.text(-LIMIT + 0.12, -LIMIT + 0.28,
            f"[{Q_full[0,0]:+.2f}  {Q_full[0,1]:+.2f}]\n"
            f"[{Q_full[1,0]:+.2f}  {Q_full[1,1]:+.2f}]",
            fontsize=8, color=text_col, fontfamily="monospace", va="bottom")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
              facecolor=surface0, labelcolor=text_col)

# ── Control panel ─────────────────────────────────────────────────────────────
LEFT     = 0.785
SW       = 0.088
SH       = 0.032
HGAP     = 0.012
panel_cx = LEFT + SW + HGAP / 2

# ── M sliders (symmetric: 3 independent entries) ──────────────────────────────
fig.text(panel_cx, 0.92, "Symmetric matrix M",
         ha="center", color=mauve, fontsize=9, fontweight="bold")

m_sliders = {}
for (r, c), y, init, lbl in [
    ((0, 0), 0.855, 3.0, "M[0,0]"),
    ((0, 1), 0.855, 0.0, "M[0,1] = M[1,0]"),
    ((1, 1), 0.795, 1.0, "M[1,1]"),
]:
    x = LEFT + c * (SW + HGAP)
    fig.text(x + SW / 2, y + SH + 0.005, lbl,
             ha="center", fontsize=6.8, color=text_col)
    sax = fig.add_axes([x, y, SW, SH])
    sax.set_facecolor(surface0)
    sl = Slider(sax, "", valmin=-4.0, valmax=4.0, valinit=init,
                color=(teal if r == c else peach), track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(7.5)
    m_sliders[(r, c)] = sl

# ── A sliders ─────────────────────────────────────────────────────────────────
fig.text(panel_cx, 0.74, "Transform A",
         ha="center", color=lavender, fontsize=9, fontweight="bold")

a_sliders = {}
for r, c, y, init in [(0, 0, 0.68, 1.0), (0, 1, 0.68, 0.0),
                       (1, 0, 0.62, 0.0), (1, 1, 0.62, 1.0)]:
    x = LEFT + c * (SW + HGAP)
    fig.text(x + SW / 2, y + SH + 0.005, f"A[{r},{c}]",
             ha="center", fontsize=7.5, color=text_col)
    sax = fig.add_axes([x, y, SW, SH])
    sax.set_facecolor(surface0)
    sl = Slider(sax, "", valmin=-3.0, valmax=3.0, valinit=init,
                color=(teal if r == c else peach), track_color=surface1)
    sl.valtext.set_color(text_col)
    sl.valtext.set_fontsize(7.5)
    a_sliders[(r, c)] = sl

# ── Info + note text ──────────────────────────────────────────────────────────
info_text = fig.text(panel_cx, 0.57, "",
                     ha="center", va="top", color=text_col,
                     fontsize=7.5, fontfamily="monospace")
note_text = fig.text(panel_cx, 0.35, "",
                     ha="center", color=green,
                     fontsize=8.5, fontweight="bold")

# ── M presets ─────────────────────────────────────────────────────────────────
fig.text(panel_cx, 0.295, "M presets", ha="center", color=mauve,
         fontsize=8, fontweight="bold")
m_preset_ax = fig.add_axes([LEFT, 0.20, 2 * SW + HGAP, 0.085])
m_preset_ax.set_facecolor(surface0)
m_presets = RadioButtons(m_preset_ax,
    ["Circle", "Ellipse", "Tilted", "Saddle"], active=1)
for lbl in m_presets.labels:
    lbl.set_color(text_col)
    lbl.set_fontsize(7.5)
m_presets.activecolor = mauve

M_PRESETS = {
    "Circle":  np.eye(2),
    "Ellipse": np.array([[3.0, 0.0], [0.0, 1.0]]),
    "Tilted":  np.array([[2.0, -1.0], [-1.0, 2.0]]),
    "Saddle":  np.array([[2.0, 0.0], [0.0, -1.0]]),
}

def on_m_preset(label):
    P = M_PRESETS[label]
    m_sliders[(0, 0)].set_val(P[0, 0])
    m_sliders[(0, 1)].set_val(P[0, 1])
    m_sliders[(1, 1)].set_val(P[1, 1])

m_presets.on_clicked(on_m_preset)

# ── A presets ─────────────────────────────────────────────────────────────────
fig.text(panel_cx, 0.185, "A presets", ha="center", color=lavender,
         fontsize=8, fontweight="bold")
a_preset_ax = fig.add_axes([LEFT, 0.10, 2 * SW + HGAP, 0.078])
a_preset_ax.set_facecolor(surface0)
a_presets = RadioButtons(a_preset_ax,
    ["Identity", "Rotation", "Shear", "Stretch"], active=0)
for lbl in a_presets.labels:
    lbl.set_color(text_col)
    lbl.set_fontsize(7.5)
a_presets.activecolor = lavender

A_PRESETS = {
    "Identity": np.eye(2),
    "Rotation": np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                           [np.sin(np.pi / 4),  np.cos(np.pi / 4)]]),
    "Shear":    np.array([[1.0, 1.0], [0.0, 1.0]]),
    "Stretch":  np.array([[2.0, 0.0], [0.0, 0.5]]),
}

def on_a_preset(label):
    P = A_PRESETS[label]
    for (r, c), sl in a_sliders.items():
        sl.set_val(P[r, c])

a_presets.on_clicked(on_a_preset)

# ── Update ────────────────────────────────────────────────────────────────────
def update(_):
    M01 = m_sliders[(0, 1)].val
    M = np.array([[m_sliders[(0, 0)].val, M01],
                  [M01,                   m_sliders[(1, 1)].val]])
    A = np.array([[a_sliders[(r, c)].val for c in range(2)] for r in range(2)])

    det_A  = np.linalg.det(A)
    AT_M_A = A.T @ M @ A           # always symmetric when M is symmetric

    # ── M eigendecomposition ──────────────────────────────────────────────────
    eig_M_vals, eig_M_vecs = np.linalg.eigh(M)

    # ── AᵀMA eigendecomposition ───────────────────────────────────────────────
    eig_QF_vals, eig_QF_vecs = np.linalg.eigh(AT_M_A)

    # ── Draw M ────────────────────────────────────────────────────────────────
    draw_panel(ax_M, M, M, "M  — base quadratic form", teal,
               f"xᵀMx = c   │   λ = {eig_M_vals[0]:+.2f}, {eig_M_vals[1]:+.2f}",
               eig_M_vals, eig_M_vecs)

    # ── Draw AᵀMA ─────────────────────────────────────────────────────────────
    draw_panel(ax_QF, AT_M_A, AT_M_A,
               "AᵀMA  — quadratic form transform", peach,
               f"x = Ay substituted   │   λ = {eig_QF_vals[0]:+.2f}, {eig_QF_vals[1]:+.2f}",
               eig_QF_vals, eig_QF_vecs)

    # ── Draw A⁻¹MA ───────────────────────────────────────────────────────────
    if abs(det_A) > 1e-10:
        Ainv = np.linalg.inv(A)
        CB   = Ainv @ M @ A          # not symmetric unless A is orthogonal

        # Eigenvectors of A⁻¹MA: A⁻¹ applied to M's eigenvectors
        CB_eigvecs = Ainv @ eig_M_vecs
        CB_eigvecs = CB_eigvecs / (np.linalg.norm(CB_eigvecs, axis=0) + 1e-12)

        # Level curves use symmetrised form so contours are well-defined
        CB_sym = (CB + CB.T) / 2

        draw_panel(ax_CB, CB_sym, CB,
                   "A⁻¹MA  — change of basis", lavender,
                   f"M in A's coordinates   │   λ = {eig_M_vals[0]:+.2f}, {eig_M_vals[1]:+.2f}  (= M)",
                   eig_M_vals, CB_eigvecs)
    else:
        ax_CB.cla()
        style_ax(ax_CB, "A⁻¹MA  — change of basis", flamingo,
                 "det(A) = 0 — A is singular")
        ax_CB.text(0, 0, "A is singular\nCannot form A⁻¹MA",
                   ha="center", va="center", fontsize=13,
                   color=flamingo, fontweight="bold")

    # ── Info panel ────────────────────────────────────────────────────────────
    det_M  = np.linalg.det(M)
    det_QF = np.linalg.det(AT_M_A)

    info_lines = [
        f"M:      λ = {eig_M_vals[0]:+.3f},  {eig_M_vals[1]:+.3f}",
        f"        tr = {np.trace(M):.3f},  det = {det_M:.3f}",
        "",
    ]

    if abs(det_A) > 1e-10:
        CB = np.linalg.inv(A) @ M @ A
        info_lines += [
            f"A⁻¹MA:  λ = {eig_M_vals[0]:+.3f},  {eig_M_vals[1]:+.3f}  ✓",
            f"        tr = {np.trace(CB):.3f},  det = {np.linalg.det(CB):.3f}",
            "",
        ]
    else:
        info_lines += ["A⁻¹MA:  undefined (det A = 0)", ""]

    info_lines += [
        f"AᵀMA:   λ = {eig_QF_vals[0]:+.3f},  {eig_QF_vals[1]:+.3f}",
        f"        tr = {np.trace(AT_M_A):.3f},  det = {det_QF:.3f}",
        f"        det = det(A)²·det(M) = {det_A**2 * det_M:.3f}",
    ]
    info_text.set_text("\n".join(info_lines))

    # ── Note ──────────────────────────────────────────────────────────────────
    if abs(det_A) > 1e-10:
        is_orthogonal = np.max(np.abs(A @ A.T - np.eye(2))) < 0.015
        if is_orthogonal:
            note_text.set_text("A is orthogonal  (AᵀA = I)\nA⁻¹MA = AᵀMA  ✓")
            note_text.set_color(green)
        else:
            eig_diff = np.max(np.abs(np.sort(eig_QF_vals) - np.sort(eig_M_vals)))
            note_text.set_text(
                f"A⁻¹MA:  eigenvalues preserved  ✓\n"
                f"AᵀMA:   eigenvalues scaled by A\n"
                f"        max Δλ = {eig_diff:.3f}"
            )
            note_text.set_color(flamingo)
    else:
        note_text.set_text("A singular — only AᵀMA defined")
        note_text.set_color(flamingo)

    fig.canvas.draw_idle()


for sl in list(m_sliders.values()) + list(a_sliders.values()):
    sl.on_changed(update)

on_m_preset("Ellipse")   # triggers initial draw via on_changed
plt.show()
