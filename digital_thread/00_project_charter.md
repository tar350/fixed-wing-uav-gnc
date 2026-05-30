# Project Charter

## Project Title

Python-Based Fixed-Wing UAV GNC Simulator with EKF Navigation and Wind-Disturbance Rejection

## Purpose

Develop a portfolio-grade aerospace GNC project that demonstrates flight dynamics, control design, navigation estimation, disturbance rejection, simulation validation, and clean software architecture.

## Current Scope: v0.2

Version 0.2 focuses on longitudinal trim:

- Solve steady level-flight trim for a target airspeed and altitude
- Report trim angle of attack, pitch angle, elevator, and throttle
- Save trim results to a machine-readable JSON artifact
- Add unit tests for trim convergence
- Maintain traceability from requirements to verification artifacts

## Completed Previous Scope: v0.1

Version 0.1 established the longitudinal autopilot skeleton:

- Altitude hold
- Airspeed hold
- Pitch dynamics
- Elevator and throttle control
- Basic disturbance modeling
- Plot generation

## Future Scope

- Numerical linearization about trim
- Stability analysis
- LQR control
- Lateral-directional dynamics
- Waypoint guidance
- Sensor models
- EKF
- Monte Carlo validation
- Animation
- Technical report

## Success Criteria

The final project should be suitable for:

- GitHub portfolio
- Resume bullets
- Interview walkthrough
- Technical presentation
