# Interface Control Document

## State Vector

Current longitudinal state vector:

| Index | Symbol | Description | Unit |
|---|---|---|---|
| 0 | u | Body-axis forward velocity | m/s |
| 1 | w | Body-axis vertical velocity, positive down | m/s |
| 2 | q | Pitch rate | rad/s |
| 3 | theta | Pitch angle | rad |
| 4 | h | Altitude, positive up | m |

## Control Vector

| Symbol | Description | Unit/Range |
|---|---|---|
| delta_e | Elevator command. Positive means nose-up in this simplified model. | rad |
| throttle | Throttle command | 0 to 1 |

## Command Inputs

| Symbol | Description | Unit |
|---|---|---|
| h_cmd | Commanded altitude | m |
| va_cmd | Commanded airspeed | m/s |

## Trim Inputs

| Symbol | Description | Unit |
|---|---|---|
| V_trim | Target trim airspeed | m/s |
| h_trim | Target trim altitude | m |
| gamma_trim | Target trim flight-path angle | rad |

## Trim Outputs

| Symbol | Description | Unit |
|---|---|---|
| alpha_trim | Angle of attack at trim | rad |
| theta_trim | Pitch angle at trim | rad |
| delta_e_trim | Elevator deflection at trim | rad |
| throttle_trim | Throttle setting at trim | nondimensional |
| residual_norm | Norm of `[u_dot, w_dot, q_dot]` residual | mixed |
