import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform

from shared.theme import (
    teal, peach, green, lavender, flamingo, sky, mauve,
    red, yellow, sapphire, maroon,
    overlay0, surface0, surface1, base, text_col,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING DATA
#  Axes are deliberately semantic so the geometry *means* something:
#    x  ≈  animacy       (0 = abstract/thing → 1 = living)
#    y  ≈  size/power    (0 = small/weak → 1 = large/powerful)
#    z  ≈  gender signal (0 = masculine → 1 = feminine)
#
#  The king–queen / man–woman analogy is a perfect parallelogram in this space.
# ═══════════════════════════════════════════════════════════════════════════════

CLUSTERS = {
    "Royalty": {
        "color": yellow,
        "words": {
            "king":     np.array([0.70,  0.90,  0.10]),
            "queen":    np.array([0.70,  0.90,  0.90]),
            "prince":   np.array([0.68,  0.75,  0.12]),
            "princess": np.array([0.68,  0.75,  0.88]),
            "knight":   np.array([0.65,  0.80,  0.15]),
            "duke":     np.array([0.66,  0.72,  0.14]),
            "throne":   np.array([0.20,  0.85,  0.50]),
            "emperor":  np.array([0.68,  0.96,  0.10]),
            "empress":  np.array([0.68,  0.96,  0.90]),
            "lord":     np.array([0.65,  0.68,  0.12]),
            "lady":     np.array([0.65,  0.68,  0.88]),
            "baron":    np.array([0.66,  0.62,  0.14]),
            "crown":    np.array([0.12,  0.88,  0.50]),
            "noble":    np.array([0.42,  0.66,  0.50]),
        },
    },
    "People": {
        "color": peach,
        "words": {
            "man":     np.array([0.80,  0.45,  0.10]),
            "woman":   np.array([0.80,  0.45,  0.90]),
            "boy":     np.array([0.80,  0.20,  0.12]),
            "girl":    np.array([0.80,  0.20,  0.88]),
            "adult":   np.array([0.78,  0.50,  0.50]),
            "child":   np.array([0.78,  0.15,  0.50]),
            "person":  np.array([0.82,  0.38,  0.50]),
            "father":  np.array([0.81,  0.55,  0.10]),
            "mother":  np.array([0.81,  0.55,  0.90]),
            "elder":   np.array([0.79,  0.60,  0.50]),
            "youth":   np.array([0.80,  0.18,  0.50]),
            "baby":    np.array([0.82,  0.05,  0.50]),
            "citizen": np.array([0.78,  0.42,  0.50]),
            "sibling": np.array([0.80,  0.28,  0.50]),
        },
    },
    "Animals": {
        "color": green,
        "words": {
            "cat":     np.array([0.95,  0.15,  0.50]),
            "dog":     np.array([0.95,  0.20,  0.45]),
            "lion":    np.array([0.95,  0.75,  0.25]),
            "lioness": np.array([0.95,  0.75,  0.75]),
            "horse":   np.array([0.95,  0.55,  0.40]),
            "wolf":    np.array([0.95,  0.60,  0.30]),
            "bird":    np.array([0.95,  0.10,  0.50]),
            "fish":    np.array([0.95,  0.08,  0.52]),
            "tiger":   np.array([0.95,  0.72,  0.28]),
            "bear":    np.array([0.95,  0.68,  0.45]),
            "rabbit":  np.array([0.95,  0.05,  0.55]),
            "deer":    np.array([0.95,  0.32,  0.55]),
            "eagle":   np.array([0.95,  0.42,  0.48]),
            "whale":   np.array([0.95,  0.88,  0.50]),
            "snake":   np.array([0.95,  0.16,  0.50]),
            "fox":     np.array([0.95,  0.24,  0.44]),
        },
    },
    "Technology": {
        "color": sky,
        "words": {
            "computer":  np.array([0.05, 0.55,  0.50]),
            "phone":     np.array([0.05, 0.35,  0.50]),
            "server":    np.array([0.05, 0.65,  0.52]),
            "network":   np.array([0.08, 0.58,  0.50]),
            "code":      np.array([0.05, 0.48,  0.48]),
            "algorithm": np.array([0.06, 0.52,  0.51]),
            "model":     np.array([0.07, 0.60,  0.50]),
            "data":      np.array([0.06, 0.50,  0.49]),
            "robot":     np.array([0.18, 0.55,  0.50]),
            "satellite": np.array([0.03, 0.72,  0.50]),
            "chip":      np.array([0.03, 0.32,  0.50]),
            "database":  np.array([0.04, 0.62,  0.49]),
            "cloud":     np.array([0.04, 0.68,  0.51]),
            "cable":     np.array([0.02, 0.28,  0.50]),
            "screen":    np.array([0.03, 0.40,  0.50]),
            "battery":   np.array([0.03, 0.30,  0.52]),
        },
    },
    "Food": {
        "color": flamingo,
        "words": {
            "apple":  np.array([0.90,  0.08,  0.52]),
            "bread":  np.array([0.30,  0.10,  0.50]),
            "meat":   np.array([0.75,  0.25,  0.50]),
            "rice":   np.array([0.25,  0.08,  0.50]),
            "soup":   np.array([0.35,  0.12,  0.51]),
            "cheese": np.array([0.20,  0.12,  0.50]),
            "cake":   np.array([0.18,  0.10,  0.55]),
            "pasta":  np.array([0.28,  0.09,  0.50]),
            "milk":   np.array([0.78,  0.15,  0.55]),
            "egg":    np.array([0.85,  0.08,  0.50]),
            "berry":  np.array([0.88,  0.07,  0.52]),
            "beer":   np.array([0.12,  0.15,  0.40]),
            "wine":   np.array([0.12,  0.18,  0.62]),
            "coffee": np.array([0.10,  0.20,  0.50]),
            "sugar":  np.array([0.08,  0.06,  0.50]),
            "grain":  np.array([0.22,  0.08,  0.50]),
        },
    },
    "Places": {
        "color": lavender,
        "words": {
            "city":     np.array([0.10,  0.70,  0.50]),
            "village":  np.array([0.12,  0.30,  0.50]),
            "ocean":    np.array([0.05,  0.80,  0.50]),
            "forest":   np.array([0.85,  0.40,  0.50]),
            "desert":   np.array([0.15,  0.35,  0.50]),
            "mountain": np.array([0.10,  0.75,  0.50]),
            "river":    np.array([0.20,  0.45,  0.50]),
            "island":   np.array([0.08,  0.58,  0.50]),
            "cave":     np.array([0.08,  0.22,  0.50]),
            "valley":   np.array([0.10,  0.38,  0.50]),
            "farm":     np.array([0.35,  0.28,  0.50]),
            "town":     np.array([0.10,  0.50,  0.50]),
            "lake":     np.array([0.15,  0.48,  0.50]),
            "park":     np.array([0.50,  0.32,  0.50]),
        },
    },
    "Emotion": {
        "color": mauve,
        "words": {
            "joy":      np.array([0.50,  0.65,  0.55]),
            "fear":     np.array([0.50,  0.30,  0.50]),
            "anger":    np.array([0.55,  0.50,  0.20]),
            "sadness":  np.array([0.50,  0.20,  0.52]),
            "love":     np.array([0.60,  0.55,  0.80]),
            "surprise": np.array([0.52,  0.60,  0.50]),
            "disgust":  np.array([0.50,  0.35,  0.30]),
            "hope":     np.array([0.50,  0.70,  0.55]),
            "guilt":    np.array([0.50,  0.25,  0.50]),
            "pride":    np.array([0.52,  0.72,  0.45]),
            "shame":    np.array([0.50,  0.22,  0.52]),
            "envy":     np.array([0.52,  0.40,  0.38]),
            "grief":    np.array([0.50,  0.15,  0.52]),
            "calm":     np.array([0.50,  0.55,  0.50]),
        },
    },
}

# ── Flatten for convenience ────────────────────────────────────────────────────
ALL_WORDS   = {}
CLUSTER_VEC = {}
for cname, cdata in CLUSTERS.items():
    CLUSTER_VEC[cname] = []
    for word, vec in cdata["words"].items():
        ALL_WORDS[word]        = (vec.copy(), cname)
        CLUSTER_VEC[cname].append(vec.copy())

def centroid(cname):
    return np.mean(CLUSTER_VEC[cname], axis=0)

# ── Gender / analogy vectors ──────────────────────────────────────────────────
GENDER_VEC  = np.array([0.0, 0.0, 0.80])
ROYALTY_VEC = ALL_WORDS["king"][0] - ALL_WORDS["man"][0]

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor(base)
fig.suptitle("Token / Embedding Space Explorer  —  Semantic Geometry",
             fontsize=13, fontweight="bold")

ax = fig.add_axes([0.00, 0.06, 0.56, 0.91], projection="3d")

# ── 3D style ──────────────────────────────────────────────────────────────────
def style_3d(ax):
    ax.set_facecolor(base)
    pane = (*mpl.colors.to_rgb(base), 0.92)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(overlay0)
    ax.tick_params(colors=overlay0, labelsize=6)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1);  ax.set_zlim(0, 1)
    ax.set_xlabel("← abstract  ·  animacy  ·  living →",
                  color=overlay0, labelpad=10, fontsize=7)
    ax.set_ylabel("← weak / small  ·  power  ·  large →",
                  color=overlay0, labelpad=10, fontsize=7)
    ax.set_zlabel("← masculine  ·  gender  ·  feminine →",
                  color=overlay0, labelpad=10, fontsize=7)

# ── Arrow helper ──────────────────────────────────────────────────────────────
def arrow3d(ax, start, end, color, lw=1.8, alpha=0.9, label=None):
    d = end - start
    ax.quiver(*start, *d, color=color, linewidth=lw, alpha=alpha,
              arrow_length_ratio=0.22, label=label)

# ── State flags ───────────────────────────────────────────────────────────────
SHOW = {
    "labels":    True,
    "centroids": True,
    "analogy":   True,
    "similarity":True,
    "convex":    False,
}

sliders   = {}
pair_state = {"a": "king", "b": "queen"}

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW
# ═══════════════════════════════════════════════════════════════════════════════
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def redraw():
    ax.cla()
    style_3d(ax)

    spread = sliders["spread"].val if sliders else 0.04
    noise  = sliders["noise"].val  if sliders else 0.00
    rng    = np.random.default_rng(42)

    drawn_vecs = {}

    for cname, cdata in CLUSTERS.items():
        color = cdata["color"]
        xs, ys, zs, labels = [], [], [], []
        for word, base_vec in cdata["words"].items():
            jitter = rng.normal(0, spread, 3) + rng.normal(0, noise, 3)
            vec    = np.clip(base_vec + jitter, 0, 1)
            drawn_vecs[word] = vec
            xs.append(vec[0]); ys.append(vec[1]); zs.append(vec[2])
            labels.append(word)

        ax.scatter(xs, ys, zs, color=color, s=52, depthshade=True,
                   alpha=0.88, zorder=5, label=cname)

        if SHOW["labels"]:
            for word, x, y, z in zip(labels, xs, ys, zs):
                ax.text(x, y, z + 0.025, word, fontsize=6.5, color=color,
                        ha="center", va="bottom", alpha=0.92)

        if SHOW["centroids"]:
            cx = np.mean(xs); cy = np.mean(ys); cz = np.mean(zs)
            ax.scatter([cx], [cy], [cz], color=color, s=130,
                       marker="D", alpha=0.35, depthshade=False)
            ax.text(cx, cy, cz + 0.05, f"[{cname}]",
                    fontsize=7, color=color, fontweight="bold",
                    ha="center", alpha=0.75)

        if SHOW["convex"]:
            cx_v = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
            for x, y, z in zip(xs, ys, zs):
                ax.plot([cx_v[0], x], [cx_v[1], y], [cx_v[2], z],
                        color=color, lw=0.5, alpha=0.18)

    if SHOW["analogy"]:
        k = drawn_vecs["king"];   q = drawn_vecs["queen"]
        m = drawn_vecs["man"];    w = drawn_vecs["woman"]

        arrow3d(ax, m, w,  color=peach,  lw=2.2, label="man → woman  (+gender)")
        arrow3d(ax, k, q,  color=yellow, lw=2.2, label="king → queen  (+gender)")

        for start, end in [(m, k), (w, q)]:
            d = end - start
            ax.quiver(*start, *d, color=overlay0, linewidth=0.9,
                      alpha=0.40, arrow_length_ratio=0.15, linestyle="dashed")

        mid = (m + w + k + q) / 4
        ax.text(*mid, "analogy\nparallelogram",
                fontsize=6.5, color=overlay0, ha="center", alpha=0.70,
                fontstyle="italic")

    if SHOW["similarity"]:
        wa, wb = pair_state["a"], pair_state["b"]
        if wa in drawn_vecs and wb in drawn_vecs:
            va = drawn_vecs[wa]; vb = drawn_vecs[wb]
            ax.plot([va[0], vb[0]], [va[1], vb[1]], [va[2], vb[2]],
                    color=teal, lw=1.8, alpha=0.80, ls="--")
            sim = cosine_sim(ALL_WORDS[wa][0], ALL_WORDS[wb][0])
            mid = (va + vb) / 2
            ax.text(*mid, f"cos={sim:.3f}",
                    fontsize=7, color=teal, ha="center", fontweight="bold")

    ax.legend(loc="upper left", fontsize=7, framealpha=0.22,
              markerscale=1.4, ncol=2)
    _update_info_panel()
    fig.canvas.draw_idle()

# ═══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL  —  controls
# ═══════════════════════════════════════════════════════════════════════════════
LEFT = 0.60
SW   = 0.35
SH   = 0.028

fig.text(LEFT + SW/2, 0.945, "── Embedding Parameters ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

slider_defs = [
    ("Cluster spread",  "spread", 0.00, 0.18, 0.04, teal),
    ("Random noise",    "noise",  0.00, 0.12, 0.00, flamingo),
]
SL_TOP = 0.89
for i, (label, key, vmin, vmax, vinit, col) in enumerate(slider_defs):
    sy = SL_TOP - i * 0.072
    fig.text(LEFT - 0.008, sy + SH/2, label,
             ha="right", va="center", fontsize=8.5, color=text_col)
    sax = fig.add_axes([LEFT, sy, SW, SH])
    sax.set_facecolor(surface0)
    sl  = Slider(ax=sax, label="", valmin=vmin, valmax=vmax, valinit=vinit,
                 color=col, track_color=surface1)
    sl.valtext.set_color(text_col);  sl.valtext.set_fontsize(8)
    sliders[key] = sl

fig.text(LEFT + SW/2, 0.725, "── Toggle Layers ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

check_ax = fig.add_axes([LEFT + 0.02, 0.555, SW - 0.02, 0.155])
check_ax.set_facecolor(surface0)
check_labels = ["Word labels", "Cluster centres", "Analogy arrows",
                "Similarity line", "Cluster mesh"]
check_init   = [SHOW["labels"], SHOW["centroids"], SHOW["analogy"],
                SHOW["similarity"], SHOW["convex"]]
chk = CheckButtons(check_ax, check_labels, check_init)
n = len(check_labels)
chk.set_frame_props({"facecolors": [surface1] * n, "edgecolors": [overlay0] * n})
chk.set_label_props({"color": [text_col] * n, "fontsize": [8.5] * n})
chk.set_check_props({"color": [mauve] * n, "linewidth": [2] * n})

def on_check(label):
    mapping = {
        "Word labels":      "labels",
        "Cluster centres":  "centroids",
        "Analogy arrows":   "analogy",
        "Similarity line":  "similarity",
        "Cluster mesh":     "convex",
    }
    key = mapping[label]
    SHOW[key] = not SHOW[key]
    redraw()

chk.on_clicked(on_check)

fig.text(LEFT + SW/2, 0.528, "── Cosine Similarity Pair ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

WORD_LIST = sorted(ALL_WORDS.keys())
pair_idx  = {"a": WORD_LIST.index("king"), "b": WORD_LIST.index("queen")}

for slot, label, y_pos in [("a", "Word A", 0.470), ("b", "Word B", 0.395)]:
    fig.text(LEFT + 0.01, y_pos + SH/2 + 0.012, label,
             ha="left", va="center", fontsize=8, color=text_col)
    sax = fig.add_axes([LEFT, y_pos, SW, SH])
    sax.set_facecolor(surface0)
    sl  = Slider(ax=sax, label="", valmin=0, valmax=len(WORD_LIST)-1,
                 valinit=pair_idx[slot], valstep=1,
                 color=teal if slot == "a" else peach,
                 track_color=surface1)
    sl.valtext.set_color(text_col); sl.valtext.set_fontsize(0)
    wlbl = fig.text(LEFT + SW/2, y_pos + SH + 0.010,
                    WORD_LIST[pair_idx[slot]],
                    ha="center", va="bottom",
                    fontsize=9, color=teal if slot == "a" else peach,
                    fontweight="bold")
    sliders[f"pair_{slot}"]     = sl
    sliders[f"pair_{slot}_lbl"] = wlbl

def on_pair_change(_):
    for slot in ("a", "b"):
        idx  = int(round(sliders[f"pair_{slot}"].val))
        word = WORD_LIST[idx]
        pair_state[slot] = word
        col = teal if slot == "a" else peach
        sliders[f"pair_{slot}_lbl"].set_text(word)
        sliders[f"pair_{slot}_lbl"].set_color(col)
    redraw()

sliders["pair_a"].on_changed(on_pair_change)
sliders["pair_b"].on_changed(on_pair_change)

fig.text(LEFT + SW/2, 0.340, "── Stats ──",
         ha="center", fontsize=9, color=mauve, fontweight="bold")

info_sim   = fig.text(LEFT + SW/2, 0.295, "",
                      ha="center", fontsize=8.5, color=teal,
                      fontfamily="monospace")
info_clust = fig.text(LEFT + SW/2, 0.258, "",
                      ha="center", fontsize=8.5, color=text_col,
                      fontfamily="monospace")
info_analogy = fig.text(LEFT + SW/2, 0.210, "",
                        ha="center", fontsize=8, color=yellow,
                        fontfamily="monospace")
info_note  = fig.text(LEFT + SW/2, 0.160, "",
                      ha="center", fontsize=7.5, color=overlay0,
                      fontstyle="italic", wrap=True)

NOTE = (
    "Axes are a simplification — real embedding\n"
    "spaces have 768–4096 dimensions.  Clusters\n"
    "emerge from co-occurrence statistics across\n"
    "billions of tokens, not manual placement."
)
fig.text(LEFT + SW/2, 0.065, NOTE,
         ha="center", va="center", fontsize=7.5,
         color=overlay0, fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.5",
                   facecolor=surface0, edgecolor=overlay0,
                   alpha=0.55, linewidth=0.8))

def _update_info_panel():
    wa = pair_state["a"];  wb = pair_state["b"]
    va = ALL_WORDS[wa][0]; vb = ALL_WORDS[wb][0]
    sim = cosine_sim(va, vb)
    ca  = ALL_WORDS[wa][1]; cb  = ALL_WORDS[wb][1]
    same = "same cluster ✓" if ca == cb else f"{ca} vs {cb}"
    col  = green if ca == cb else peach
    info_sim.set_color(teal)
    info_sim.set_text(f'cos("{wa}", "{wb}") = {sim:.4f}')
    info_clust.set_text(f"{same}")
    info_clust.set_color(col)

    target = ALL_WORDS["king"][0] - ALL_WORDS["man"][0] + ALL_WORDS["woman"][0]
    best_word, best_sim = "", -2
    for word, (vec, _) in ALL_WORDS.items():
        if word in ("king", "man", "woman"):
            continue
        s = cosine_sim(target, vec)
        if s > best_sim:
            best_sim  = s
            best_word = word
    info_analogy.set_text(
        f"king − man + woman  ≈  {best_word}  (cos={best_sim:.3f})"
    )
    info_note.set_text("")

def on_slider(_):
    redraw()

sliders["spread"].on_changed(on_slider)
sliders["noise"].on_changed(on_slider)

redraw()
plt.show()
