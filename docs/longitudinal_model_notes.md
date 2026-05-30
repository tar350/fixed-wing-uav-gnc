# Longitudinal Model Notes

This project currently uses a simplified 5-state longitudinal rigid-body model.

State vector:

```text
x = [u, w, q, theta, h]
```

where:

- `u` is body-axis forward velocity.
- `w` is body-axis vertical velocity, positive down.
- `q` is pitch rate.
- `theta` is pitch angle.
- `h` is altitude, positive upward.

Control vector:

```text
u_control = [delta_e, throttle]
```

where:

- `delta_e` is elevator deflection in radians.
- `throttle` is normalized between 0 and 1.

The aerodynamic model uses coefficient buildup:

```text
CL = CL0 + CL_alpha * alpha + CL_q * q_hat + CL_delta_e * delta_e
CD = CD0 + k * CL^2
Cm = Cm0 + Cm_alpha * alpha + Cm_q * q_hat + Cm_delta_e * delta_e
```

The body-axis force conversion uses positive body `z` downward:

```text
X_aero = -D cos(alpha) + L sin(alpha)
Z_aero = -D sin(alpha) - L cos(alpha)
```

The equations of motion are:

```text
u_dot     = X/m - g sin(theta) - q w
w_dot     = Z/m + g cos(theta) + q u
q_dot     = M/I_y
theta_dot = q
h_dot     = u sin(theta) - w cos(theta)
```

## v0.2 Trim Formulation

For a desired airspeed `V`, altitude `h`, and flight-path angle `gamma`, the trim solver uses:

```text
u = V cos(alpha)
w = V sin(alpha)
q = 0
theta = alpha + gamma
```

For steady level flight, `gamma = 0`, so `theta = alpha` and altitude rate is zero.

The solver finds:

```text
alpha_trim
delta_e_trim
throttle_trim
```

such that:

```text
u_dot ≈ 0
w_dot ≈ 0
q_dot ≈ 0
```

The altitude-rate residual is also reported for traceability.

## Known Simplifications

- Density is currently constant and does not vary with altitude.
- Actuator dynamics are not included yet.
- Propulsion is modeled as throttle-scaled thrust along the body x-axis.
- Aerodynamic coefficients are approximate and not intended for certification.
- This version is longitudinal only; lateral-directional dynamics are future work.
