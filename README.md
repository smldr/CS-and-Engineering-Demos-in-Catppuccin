<div align="center">

<h1>CS & Engineering Demos</h1>

<p><em>Interactive visualisations for PhD-level maths and engineering — all in Catppuccin Mocha.</em></p>

[![Catppuccin Mocha](https://img.shields.io/badge/Catppuccin-Mocha-cba6f7?style=flat-square&labelColor=1e1e2e&color=cba6f7)](https://github.com/catppuccin/catppuccin)
[![Python](https://img.shields.io/badge/Python-3.10+-89b4fa?style=flat-square&labelColor=1e1e2e)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-interactive-a6e3a1?style=flat-square&labelColor=1e1e2e)](https://matplotlib.org/)

</div>

---

Interactive Python demos built around my PhD studies and for students — nothing too fancy, but every plot is themed in [Catppuccin Mocha](https://github.com/catppuccin/catppuccin). Because Catppuccin all the things.

Each demo is a standalone script: run it, drag sliders, and watch the maths move.

---

## Structure

```
.
├── shared/
│   └── theme.py                  ← Catppuccin Mocha palette for matplotlib
│
├── linear_algebra/
│   └── demos/
│       ├── linear_transform.py   ← live 2D/3D matrix editor with basis vectors
│       └── cosine_similarity.py  ← angle, dot product & projection visualiser
│
├── optimisation/
│   └── demos/
│       └── search_space.py       ← loss landscape explorer with GD trajectory
│
├── machine_learning/
│   └── demos/
│       └── embeddings.py         ← 3D embedding space & semantic analogies
│
├── calculus/                     ← coming soon
├── differential_equations/       ← coming soon
├── optimal_control/              ← coming soon
└── reinforcement_learning/       ← coming soon
```

---

## Demos

### Linear Algebra

#### `linear_transform.py` — Live Matrix Editor

Drag nine sliders to build any 3×3 matrix and watch the standard basis vectors transform in real time. Toggle between 3D and 2D mode, and track the determinant to see when the transformation collapses the space.

#### `cosine_similarity.py` — Cosine Similarity Visualiser

Place two vectors with sliders and see the angle between them, their dot product, projection, and an interpretability label (orthogonal / highly similar / anti-parallel, etc.). Supports 2D and 3D.

---

### Optimisation

#### `search_space.py` — Loss Landscape Explorer

Parametric loss surface `L(θ₁, θ₂)` with adjustable basin depth, roughness, frequency, and tilt. Drop an agent anywhere on the surface, watch the negative gradient arrow update live, and trace a full gradient descent trajectory to convergence. The landscape reports convexity in real time.

---

### Machine Learning

#### `embeddings.py` — Embedding Space Explorer

A hand-crafted 3D semantic space with 98 word vectors across 7 clusters (Royalty, People, Animals, Technology, Food, Places, Emotion). Toggle cluster centroids, word labels, the king−man+woman analogy parallelogram, and a live cosine-similarity pair selector.

---

## Quick Start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run any demo directly
python linear_algebra/demos/linear_transform.py
python linear_algebra/demos/cosine_similarity.py
python optimisation/demos/search_space.py
python machine_learning/demos/embeddings.py
```

Requires Python 3.10+. A virtual environment is recommended.

---

## The Palette

All demos share `shared/theme.py`, which loads Catppuccin Mocha via the [`catppuccin`](https://pypi.org/project/catppuccin/) package and exposes named colour constants:

| Colour | Used for |
|--------|----------|
| `teal` | primary vector / axis A |
| `peach` | secondary vector / axis B |
| `sky` | positive values, projections |
| `flamingo` | negative values, warnings |
| `mauve` | UI labels, trajectories |
| `green` | convergence, projections |
| `yellow` | agent position, royalty cluster |
| `overlay0` | grid lines, minor labels |
| `base` | background |

---

<div align="center">
<sub>Built with Catppuccin Mocha · matplotlib · numpy</sub>
</div>
