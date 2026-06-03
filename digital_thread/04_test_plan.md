# Test Plan

## v0.1 Tests

| Test ID | Description | Expected Result |
|---|---|---|
| TEST-001 | Run simulation with default config | Simulation completes without error |
| TEST-002 | Altitude step from 100 m to 150 m | Altitude response approaches command |
| TEST-003 | Airspeed command from 20 m/s to 22 m/s | Airspeed approaches command |
| TEST-004 | Enable vertical gust | Controller remains bounded |
| TEST-005 | Run unit tests | All tests pass |

## v0.2 Tests

| Test ID | Description | Expected Result |
|---|---|---|
| TEST-006 | Run trim solver for default level-flight case | Trim converges |
| TEST-007 | Check trim residual norm | Residual norm < 1e-6 |
| TEST-008 | Check trim state airspeed | Trim state airspeed equals requested airspeed |
| TEST-009 | Save trim JSON artifact | `outputs/logs/trim_result_v0_2.json` is created |

## Future Tests

- Linearization validation
- LQR closed-loop eigenvalue check
- Waypoint tracking RMS cross-track error
- EKF estimation error convergence
- Monte Carlo uncertainty sweep

## v0.9 EKF Navigation Test Plan

### Test Case: EKF Navigation Demo

Command:

```powershell
python scripts\run_ekf_navigation_demo.py

## v1.0 Estimator-in-the-Loop Test Plan

### Test Case: Estimator-in-the-Loop Demo

Command:

```powershell
python scripts\run_estimator_in_loop_demo.py