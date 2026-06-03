# Requirements

## Functional Requirements

| ID | Requirement | Verification Method | Status |
|---|---|---|---|
| GNC-FR-001 | Simulate fixed-wing UAV longitudinal dynamics | Simulation test | Started |
| GNC-FR-002 | Track commanded altitude using elevator control | Plot and error metric | Started |
| GNC-FR-003 | Track commanded airspeed using throttle control | Plot and error metric | Started |
| GNC-FR-004 | Include configurable aircraft parameters | Config inspection | Started |
| GNC-FR-005 | Include configurable controller gains | Config inspection | Started |
| GNC-FR-006 | Generate plots for state and control response | Plot review | Started |
| GNC-FR-007 | Save simulation outputs for traceability | CSV/log review | Started |
| GNC-FR-008 | Solve longitudinal trim for a commanded airspeed and altitude | Trim unit test and trim demo | Implemented |
| GNC-FR-009 | Save trim result as a traceable output artifact | JSON log review | Implemented |

## Non-Functional Requirements

| ID | Requirement | Verification Method | Status |
|---|---|---|---|
| GNC-NFR-001 | Code shall be modular and function-based | Code review | Started |
| GNC-NFR-002 | Simulation shall be runnable from VS Code terminal | Execution test | Started |
| GNC-NFR-003 | Project shall maintain a digital thread | Folder/document review | Started |
| GNC-NFR-004 | Core functions shall be unit-testable | Pytest | Started |
| GNC-NFR-005 | Versioned outputs shall be stored under `outputs/` | Artifact inspection | Started |
## v0.7 Interactive Dashboard Requirements

| ID | Requirement | Verification |
|---|---|---|
| GNC-REQ-010 | The software shall provide an interactive GUI for running nonlinear LQR simulations. | Run `streamlit run apps/gnc_dashboard.py` |
| GNC-REQ-011 | The GUI shall allow the user to modify altitude command, airspeed command, initial conditions, simulation time, and wind disturbance settings. | Manual GUI inspection |
| GNC-REQ-012 | The GUI shall display time histories, control inputs, tracking errors, summary metrics, and 2D aircraft motion. | Manual GUI inspection |
| GNC-REQ-013 | The GUI shall allow simulation results to be downloaded as CSV. | Manual GUI inspection |
## v0.7 Interactive Dashboard Test Plan

### Test Case: Dashboard Launch

Command:

```powershell
streamlit run apps\gnc_dashboard.py

## v0.9 EKF Navigation Requirements

| ID | Requirement | Verification |
|---|---|---|
| GNC-REQ-014 | The software shall simulate noisy longitudinal air-data, gyro, attitude, and barometric measurements. | Run `python scripts/run_ekf_navigation_demo.py` |
| GNC-REQ-015 | The software shall estimate longitudinal state `[u, w, q, theta, h]` using an Extended Kalman Filter. | Run `pytest tests/test_ekf_navigation.py` |
| GNC-REQ-016 | The software shall export EKF truth, measurement, estimate, and error histories. | Inspect `outputs/logs/ekf_navigation_results_v0_9.csv` |
| GNC-REQ-017 | The software shall provide an interactive EKF dashboard for sensor-noise and estimation visualization. | Run `streamlit run apps/ekf_navigation_dashboard.py` |

## v1.0 Estimator-in-the-Loop GNC Requirements

| ID | Requirement | Verification |
|---|---|---|
| GNC-REQ-018 | The software shall close the loop between noisy sensors, EKF navigation, LQR control, and nonlinear aircraft dynamics. | Run `python scripts/run_estimator_in_loop_demo.py` |
| GNC-REQ-019 | The LQR controller shall use EKF-estimated states instead of truth states. | Inspect `src/gnc_sim/simulation/estimator_in_loop.py` |
| GNC-REQ-020 | The software shall log truth states, noisy measurements, EKF estimates, controls, tracking errors, and estimation errors. | Inspect `outputs/logs/estimator_in_loop_results_v1_0.csv` |
| GNC-REQ-021 | The software shall generate plots and animation for estimator-in-the-loop behavior. | Inspect `outputs/plots` and `outputs/animations` |