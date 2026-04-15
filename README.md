# BeamCalc — Beam Analysis Tool

> A professional structural engineering web app for computing and visualizing shear force, bending moment, and beam deflection — built with Python and Streamlit.

**Live Demo:** [beamcalcpy.streamlit.app](https://beamcalcpy.streamlit.app/)

---

## What is BeamCalc?

BeamCalc is an interactive beam analysis tool designed for mechanical and civil engineering students and professionals. It automates the core structural calculations that would otherwise require manual statics work — shear force, bending moment, and deflection — and presents them through clean, interactive visualizations.

You can configure any beam with custom loads and instantly see the SFD, BMD, and deflection curve update in real time.

---

## Features

- **Beam Schematic Visualizer** — draws the beam with support symbols, point load arrows, and UDL hatching
- **Shear Force Diagram (SFD)** — computed numerically using NumPy across 1000 points
- **Bending Moment Diagram (BMD)** — integrated from shear force distribution
- **Deflection Curve** — solved symbolically using SymPy with boundary conditions applied
- **Interactive Plotly Charts** — hover over any point to see exact values (position, force, moment, deflection)
- **Animated Result Cards** — max shear, max moment, max deflection, reaction forces with count-up animation
- **PDF Report Export** — one-click download of a full report including beam schematic, loads table, and all diagrams
- **Support Types** — Simply Supported and Cantilever
- **Multiple Loads** — add any number of Point Loads and UDLs, remove individually
- **Apple-inspired UI** — clean light theme, frosted glass sidebar, smooth CSS animations

---

## Tech Stack

| Library | Purpose |
|---|---|
| `Streamlit` | Web UI framework |
| `NumPy` | Numerical SFD/BMD computation |
| `SymPy` | Symbolic deflection solving |
| `Matplotlib` | Beam schematic + PDF chart rendering |
| `Plotly` | Interactive SFD/BMD/deflection charts |
| `fpdf2` | PDF report generation |

---

## How It Works

### 1. Support Reactions
Static equilibrium equations are solved to find vertical reactions at supports:
- **Simply Supported:** Sum of forces and moments → R_A and R_B
- **Cantilever:** Fixed at x=0, R_A = total applied load

### 2. Shear Force Diagram
NumPy evaluates shear force at 1000 evenly spaced points along the beam by summing all forces to the left of each point.

### 3. Bending Moment Diagram
The moment at each point is obtained by numerically integrating the shear force using cumulative summation (`np.cumsum`).

### 4. Deflection
SymPy builds a symbolic expression for M(x) using Macaulay's method (Piecewise functions), then double-integrates to get deflection:

```
EI · d²y/dx² = M(x)
```

Integration constants C1 and C2 are solved using boundary conditions:
- Simply Supported: y(0) = 0, y(L) = 0
- Cantilever: y(0) = 0, y'(0) = 0

---

## Sign Convention

| Quantity | Positive Direction |
|---|---|
| Loads | Downward |
| Shear Force | Left face upward |
| Bending Moment | Sagging (concave up) |
| Deflection | Downward |

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/rujopujo/beamcalc.git
cd beamcalc
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## Project Structure

```
beamcalc/
├── app.py              # Full application (UI + all computation logic)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

### Functions in `app.py`

| Function | Description |
|---|---|
| `validate_inputs()` | Checks all inputs before running analysis |
| `compute_reactions()` | Calculates support reactions via equilibrium |
| `compute_sfd()` | Returns shear force array using NumPy |
| `compute_bmd()` | Returns bending moment array via integration |
| `compute_deflection()` | Returns deflection array using SymPy |
| `draw_beam_visualizer()` | Renders beam schematic with Matplotlib |
| `plot_results_plotly()` | Builds 3 interactive Plotly subplots |
| `generate_pdf()` | Exports full PDF report using fpdf2 |

---

## Usage Guide

1. **Set beam length** and **support type** in the sidebar
2. **Enter material properties** — Young's Modulus E (GPa) and Moment of Inertia I (cm⁴)
3. **Add loads** — Point Loads (position + magnitude) and/or UDLs (intensity + range)
4. Click **Analyze Beam**
5. View the beam schematic, result cards, and interactive diagrams
6. Click **Download PDF Report** to export results

---

## Example Input

| Parameter | Value |
|---|---|
| Beam Length | 5 m |
| Support Type | Simply Supported |
| E | 200 GPa (Steel) |
| I | 8000 cm⁴ |
| Point Load | 10 kN at 2.5 m |
| UDL | 5 kN/m over full span |

---

## Contributors

| Name | Role |
|---|---|
| **Ruhaan Joshi** | Contributor |
| **Rudra Jain** | Contributor |
| **Aarush Nalavade** | Contributor |
| **Varun Singh** | Contributor |

---

## References

Open-access papers related to educational beam analysis, Python tooling, and symbolic computation used in solvers like BeamCalc:

1. **Carella, A. R.** (2019). *BeamBending: a teaching aid for 1-D shear force and bending moment diagrams.* *Journal of Open Source Education*, 2(19), 65. [https://doi.org/10.21105/jose.00065](https://doi.org/10.21105/jose.00065) — [PDF](https://jose.theoj.org/papers/10.21105/jose.00065.pdf)

2. **Bonanno, J.** (2021). *IndeterminateBeam: A Python package for solving 1D indeterminate beams.* *Journal of Open Source Education*, 4(40), 111. [https://doi.org/10.21105/jose.00111](https://doi.org/10.21105/jose.00111) — [PDF](https://jose.theoj.org/papers/10.21105/jose.00111.pdf)

3. **Meurer, A., et al.** (2017). *SymPy: symbolic computing in Python.* *PeerJ Computer Science*, 3, e103. [https://doi.org/10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103) — [Article (incl. PDF)](https://peerj.com/articles/cs-103/)

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

```
MIT License

Copyright (c) 2025 Ruhaan Joshi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
