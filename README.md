# ApexLex ARC — Minimal Prototype

This repository contains a minimal Python prototype demonstrating the ApexLex ARC 12-line control loop with:

- Gaussian belief update (toy dynamics)
- A simple Lagrangian minimizer stub using a proximal QP
- A safety projector implemented as a projection QP (cvxpy + OSQP)
- Append-only signed log chain using Ed25519

Files
- `arc_loop.py` — main prototype (run to simulate a few loop steps)
- `requirements.txt` — Python dependencies
- `tests/test_projection.py` — unit test for projector
- `tests/test_log_chain.py` — unit test for log chaining and signature verification

Run (recommended into a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python arc_loop.py
```

Run tests:

```powershell
pip install pytest
pytest -q
```

Notes
- This is a minimal educational prototype and not production-grade. It demonstrates the structure and auditable logging pattern; replace solver, belief model, and key storage with secure production components for deployment.
