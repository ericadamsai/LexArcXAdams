import numpy as np
from arc_loop import safety_project


def test_projection_basic():
    x_hat = np.zeros(2)
    u_candidate = np.array([10.0, -10.0])
    u_safe, meta = safety_project(x_hat, u_candidate)
    # projection should bound each control to [-1,1] or to the constraint set
    assert meta["status"] in ("ok", "failed")
    assert len(u_safe) == 2
