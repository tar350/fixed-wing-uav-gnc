# Fixed-Wing UAV GNC Simulator

Python-based fixed-wing UAV Guidance, Navigation, and Control project.

## Current Version: v0.2 Longitudinal Trim Solver

This version includes everything from v0.1 plus a formal trim solver:

- Longitudinal aircraft dynamics
- Aerodynamic coefficient model
- Cascaded altitude → pitch → elevator control
- Airspeed → throttle control
- Basic wind/gust hook
- Simulation runner
- Plot generation
- Digital-thread documentation structure
- **Longitudinal trim solver for steady level flight**
- **Trim JSON result logging**
- **Trim unit tests**

## Project Goal

Build a portfolio-grade GNC simulator that evolves from a basic longitudinal autopilot into a complete fixed-wing UAV GNC stack with:

- Trim analysis
- Linearization
- PID and LQR control
- Waypoint guidance
- Sensor simulation
- EKF navigation
- Wind disturbance rejection
- Monte Carlo validation
- 2D/3D trajectory animation

## Recommended Environment

Python 3.10+.

## Setup in VS Code

From the project root, meaning the folder containing `requirements.txt` and `pyproject.toml`:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

Run the trim solver:

```bash
python scripts/run_trim_demo.py
```

Run the longitudinal simulation:

```bash
python scripts/run_longitudinal_demo.py
```

Run tests:

```bash
pytest
```

## Folder Logic

```text
fixed_wing_uav_gnc/
├── configs/                 # Aircraft, trim, and simulation input files
├── digital_thread/          # Requirements, assumptions, verification logs
├── docs/                    # Theory notes and report drafts
├── notebooks/               # Exploration notebooks
├── outputs/                 # Generated plots, logs, simulation CSV files
├── scripts/                 # User-facing runnable scripts
├── src/gnc_sim/             # Main Python package
└── tests/                   # Unit tests
```

## Engineering Rule

Do not let random scripts become the project. Every major feature should live inside `src/gnc_sim/` as a function or class and be called by a script in `scripts/`.

## Current Architecture

```text
configs/uav_longitudinal.json
        ↓
scripts/run_trim_demo.py
        ↓
gnc_sim.models.uav_longitudinal
gnc_sim.trim.longitudinal_trim
        ↓
outputs/logs/trim_result_v0_2.json
```

```text
configs/uav_longitudinal.json
        ↓
scripts/run_longitudinal_demo.py
        ↓
gnc_sim.models.uav_longitudinal
gnc_sim.controllers.autopilot_longitudinal
gnc_sim.simulation.integrator
gnc_sim.utils.plotting
        ↓
outputs/plots/
outputs/logs/
```

## v0.2 Trim Method

The trim solver finds angle of attack, elevator deflection, and throttle for a requested airspeed, altitude, and flight-path angle.

For level flight:

```text
gamma = 0
theta = alpha
u_dot ≈ 0
w_dot ≈ 0
q_dot ≈ 0
h_dot ≈ 0
```

The solution is saved to:

```text
outputs/logs/trim_result_v0_2.json
```

## Suggested Next Version

v0.3 should add numerical linearization about the trim point:

```text
x_dot = f(x, u)
A = ∂f/∂x |trim
B = ∂f/∂u |trim
```

That will enable formal stability analysis and LQR controller design.
