from gnc_sim.controllers.pid import PIDController, PIDGains


def test_pid_zero_error_returns_zero():
    pid = PIDController(PIDGains(kp=1.0, ki=0.0, kd=0.0))
    assert pid.update(error=0.0, dt=0.1) == 0.0


def test_pid_output_saturation():
    pid = PIDController(
        PIDGains(kp=10.0, ki=0.0, kd=0.0),
        output_limits=(-1.0, 1.0),
    )
    assert pid.update(error=10.0, dt=0.1) == 1.0
    assert pid.update(error=-10.0, dt=0.1) == -1.0
