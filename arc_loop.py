import os
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import concurrent.futures

import numpy as np
import cvxpy as cp
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_private_key_from_env() -> Ed25519PrivateKey:
    """Load Ed25519 private key from environment (hex) or from file path.
    If not provided, generate an ephemeral key and log a clear prototype warning.
    Environment variables supported:
      - ARC_PRIVATE_KEY_HEX: hex-encoded private key bytes
      - ARC_PRIVATE_KEY_PATH: path to a raw private key file (bytes)
    """
    hexenv = os.getenv("ARC_PRIVATE_KEY_HEX")
    pathenv = os.getenv("ARC_PRIVATE_KEY_PATH")
    if hexenv:
        try:
            raw = bytes.fromhex(hexenv)
            return Ed25519PrivateKey.from_private_bytes(raw)
        except Exception:
            print("[WARN] ARC_PRIVATE_KEY_HEX provided but invalid hex; generating ephemeral key.")
    if pathenv and os.path.exists(pathenv):
        try:
            with open(pathenv, "rb") as f:
                raw = f.read()
            return Ed25519PrivateKey.from_private_bytes(raw)
        except Exception:
            print(f"[WARN] ARC_PRIVATE_KEY_PATH={pathenv} unreadable; generating ephemeral key.")
    # fallback: ephemeral key (prototype only)
    print("[PROTOTYPE WARNING] No persistent signing key found. Using ephemeral in-memory key. Replace with HSM/KMS for production.")
    return Ed25519PrivateKey.generate()


@dataclass
class LogRecord:
    t: int
    timestamp: float
    prev_hash: Optional[str]
    o_t_hash: str
    b_mean: list
    b_cov_hash: str
    u_candidate: list
    u_safe: list
    solver_meta: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class SignedLog:
    def __init__(self, privkey: Ed25519PrivateKey):
        self.priv = privkey
        self.pub = privkey.public_key()
        self.prev_hash: Optional[str] = None
        self.records = []

    def append(self, record: LogRecord) -> dict:
        payload = record.to_json().encode("utf-8")
        entry_hash = sha256_hex(payload + (self.prev_hash or b""))
        sig = self.priv.sign(entry_hash.encode("utf-8"))
        entry = {
            "record": json.loads(payload.decode("utf-8")),
            "entry_hash": entry_hash,
            "signature": sig.hex(),
            "pubkey": self.pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex(),
        }
        self.prev_hash = entry_hash
        self.records.append(entry)
        return entry

    @staticmethod
    def verify_entry(entry: dict) -> bool:
        entry_hash = entry["entry_hash"]
        sig = bytes.fromhex(entry["signature"])
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(entry["pubkey"]))
        try:
            pub.verify(sig, entry_hash.encode("utf-8"))
            return True
        except Exception:
            return False


class ToyBelief:
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov

    def update(self, u: np.ndarray, z: np.ndarray):
        # Toy linear-Gaussian update: x_{t+1} = A x + B u + w; observe z = C x + v
        A = np.eye(len(self.mean))
        B = np.eye(len(self.mean)) * 0.1
        C = np.eye(len(self.mean))
        Q = np.eye(len(self.mean)) * 0.01
        R = np.eye(len(self.mean)) * 0.05

        x_pred = A @ self.mean + B @ u
        P_pred = A @ self.cov @ A.T + Q

        # simple KF update
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        y = z - C @ x_pred
        self.mean = x_pred + K @ y
        self.cov = (np.eye(len(self.mean)) - K @ C) @ P_pred

    def snapshot_hash(self) -> str:
        m = self.mean.tobytes() + self.cov.tobytes()
        return sha256_hex(m)


def safety_project(x_hat: np.ndarray, u_candidate: np.ndarray) -> Tuple[np.ndarray, dict]:
    # Projection onto simple linear constraints: G u <= h
    n = len(u_candidate)
    u = cp.Variable(n)
    # Example: limit each control to [-1, 1] and a linear constraint sum(u) <= 0.8
    G = np.vstack([np.eye(n), -np.eye(n), np.ones((1, n))])
    h = np.hstack([np.ones(n), np.ones(n), 0.8 * np.ones(1)])
    cost = cp.sum_squares(u - u_candidate)
    prob = cp.Problem(cp.Minimize(cost), [G @ u <= h])
    prob.solve(solver=cp.OSQP, warm_start=True)
    if u.value is None:
        return u_candidate, {"status": "failed"}
    return u.value, {"status": "ok", "proj_resid": float(prob.value)}


def minimize_lagrangian_stub(b: ToyBelief, a0: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, dict]:
    # Minimal proximal-regularized QP: minimize ||u - u0||^2 + alpha*||u||^2
    n = len(u0)
    u = cp.Variable(n)
    alpha = 1e-2
    cost = cp.sum_squares(u - u0) + alpha * cp.sum_squares(u)
    # Dummy safety linear constraint (small)
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.hstack([2.0 * np.ones(n), 2.0 * np.ones(n)])
    prob = cp.Problem(cp.Minimize(cost), [G @ u <= h])
    prob.solve(solver=cp.OSQP, warm_start=True)
    if u.value is None:
        return u0, {"status": "failed"}
    return u.value, {"status": "ok", "cost": float(prob.value)}


def pi_safe(b: ToyBelief) -> np.ndarray:
    """Simple safe fallback policy: return zero control clipped to safe bounds.
    Replace with certified safe policy in production.
    """
    return np.zeros_like(b.mean)


def run_minimize_with_timeout(b, a0, u0, timeout_s=0.5):
    """Run minimize_lagrangian_stub with a timeout; return fallback if it times out or fails."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(minimize_lagrangian_stub, b, a0, u0)
        try:
            res = future.result(timeout=timeout_s)
            return res
        except concurrent.futures.TimeoutError:
            # timeout: use safe fallback
            print(f"[WARN] Solver timeout after {timeout_s}s — using fallback policy")
            return pi_safe(b), {"status": "timeout_fallback"}
        except Exception as e:
            print(f"[ERROR] Solver exception: {e}; using fallback")
            return pi_safe(b), {"status": "exception_fallback"}


def main(steps: int = 5):
    # load signing key (prefer persistent key via env/path). Ephemeral key only for prototype.
    priv = load_private_key_from_env()
    slog = SignedLog(priv)

    # initial belief
    b = ToyBelief(mean=np.zeros(2), cov=np.eye(2) * 0.1)

    u_prev = np.zeros(2)
    a0 = np.zeros(2)

    for t in range(steps):
        # 1) observe
        o_t = (np.random.randn(2) * 0.05 + b.mean).tolist()
        o_t_hash = sha256_hex(json.dumps(o_t).encode("utf-8"))

        # 2) belief update (simulate measurement z = x + noise)
        z = np.array(o_t)
        b.update(u_prev, z)

        # 3) state estimate
        x_hat = b.mean.copy()

        # 4-5) propose
        a0 = -0.1 * b.mean
        u0 = a0.copy()

        # 6) minimize L (stub) — run with timeout and fallback
        timeout_s = float(os.getenv("ARC_SOLVER_TIMEOUT", "0.5"))
        u_candidate, solver_meta = run_minimize_with_timeout(b, a0, u0, timeout_s=timeout_s)

        # 7) safety projector
        u_safe, proj_meta = safety_project(x_hat, u_candidate)

        # 8) execute (simulated)
        # 9) measure costs (toy)
        r_t = -np.sum(u_safe**2)
        c_t = np.maximum(0.0, np.sum(u_safe) - 0.8)

        # Build log
        rec = LogRecord(
            t=t,
            timestamp=time.time(),
            prev_hash=slog.prev_hash,
            o_t_hash=o_t_hash,
            b_mean=b.mean.tolist(),
            b_cov_hash=b.snapshot_hash(),
            u_candidate=u_candidate.tolist(),
            u_safe=u_safe.tolist(),
            solver_meta={**solver_meta, **proj_meta, "r_t": float(r_t), "c_t": float(c_t)},
        )
        entry = slog.append(rec)
        print(f"Step {t}: u_safe={u_safe}, entry_hash={entry['entry_hash'][:8]}")

        # advance
        u_prev = u_safe


if __name__ == "__main__":
    main(steps=6)
