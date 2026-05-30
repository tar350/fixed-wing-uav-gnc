"""Reusable PID controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDGains:
    """PID gain container."""

    kp: float
    ki: float
    kd: float
    integral_limit: float | None = None


class PIDController:
    """Basic PID controller with anti-windup integral limiting.

    This class is intentionally small and testable.
    """

    def __init__(
        self,
        gains: PIDGains,
        output_limits: tuple[float, float] | None = None,
        initial_integral: float = 0.0,
    ) -> None:
        self.gains = gains
        self.output_limits = output_limits
        self.integral = initial_integral
        self.previous_error: float | None = None

    def reset(self) -> None:
        """Reset controller memory."""
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> float:
        """Update PID controller.

        Parameters
        ----------
        error:
            Command minus measured value.
        dt:
            Time step in seconds.

        Returns
        -------
        float
            Saturated controller output.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.integral += error * dt

        if self.gains.integral_limit is not None:
            limit = abs(self.gains.integral_limit)
            self.integral = max(-limit, min(limit, self.integral))

        if self.previous_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.previous_error) / dt

        raw_output = (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )

        self.previous_error = error

        return self._saturate(raw_output)

    def _saturate(self, value: float) -> float:
        if self.output_limits is None:
            return value

        lower, upper = self.output_limits
        if lower > upper:
            raise ValueError("Lower output limit cannot exceed upper output limit.")

        return max(lower, min(upper, value))
