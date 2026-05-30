# Assumptions

## Aircraft Modeling Assumptions

- Aircraft is represented initially using longitudinal dynamics only.
- Aircraft is assumed symmetric.
- Lateral-directional dynamics are not included yet.
- Aerodynamic coefficients are approximate and intended for project development, not aircraft certification.
- Elevator sign convention: positive elevator command produces positive nose-up pitching moment in this model.
- Thrust acts along the body x-axis.
- Earth is treated as flat and non-rotating for this phase.
- Density is constant at sea-level standard value unless changed in config.

## Trim Assumptions

- v0.2 trim solves a steady longitudinal flight condition.
- Trim variables are angle of attack, elevator deflection, and throttle.
- Pitch rate is assumed zero at trim.
- Desired flight-path angle is enforced kinematically using `theta = alpha + gamma`.
- For steady level flight, `gamma = 0`, which gives `theta = alpha`.
- The residual vector minimized by the trim solver is `[u_dot, w_dot, q_dot]`.
- Altitude-rate residual is reported separately for traceability.

## Control Assumptions

- Cascaded PID architecture is used for the first controller version.
- Altitude controller outputs pitch command.
- Pitch controller outputs elevator command.
- Airspeed controller outputs throttle command.
- Actuator dynamics are not modeled yet.

## Wind Assumptions

- Gust model is a simple vertical sinusoidal disturbance hook for early robustness testing.
- Full Dryden/Von Karman turbulence models are future work.
