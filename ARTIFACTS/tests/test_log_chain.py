import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from arc_loop import SignedLog, LogRecord


def test_log_sign_and_verify():
    priv = Ed25519PrivateKey.generate()
    slog = SignedLog(priv)
    rec = LogRecord(
        t=0,
        timestamp=time.time(),
        prev_hash=None,
        o_t_hash="abc",
        b_mean=[0.0, 0.0],
        b_cov_hash="def",
        u_candidate=[0.0, 0.0],
        u_safe=[0.0, 0.0],
        solver_meta={}
    )
    entry = slog.append(rec)
    assert SignedLog.verify_entry(entry)
