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
