# ApexLex ARC — Minimal Prototype  
#### Powered by AxiomHive | Alexis M. Adams, Founder & Architect

This repository is part of **AxiomHive**, a real, founder-led deterministic AI R&D program created by Alexis M. Adams. AxiomHive focuses on building deterministic, auditable AI architectures for critical domains, advancing zero-drift substrate control (C=0), cryptographically verifiable cognition, and formal-invariant execution.

---

## About AxiomHive
AxiomHive’s mission is to provide deterministic AI ecosystems where every output is formally proven, cryptographically logged, and consistent with a single source of truth (SSOT). We reject probabilistic “best-effort” answers in favor of proof-carrying, consistent computation—especially for high-stakes applications in security, finance, infrastructure, and defense.

Learn more at [axiomhive.org](https://axiomhive.org).

---

## About the Architect
**Alexis M. Adams** is the sole originator and architect of the AxiomHive deterministic AI framework. Their public footprint spans GitHub (AXI0MH1VE), X/Twitter (@devdollzai), and long-form writings on formal AI guarantees.

---

## Thesis: Deterministic AI  
Mainstream AI relies on stochastic models. AxiomHive’s work prioritizes **deterministic substrates** and “C=0” (zero consistency error), where all outputs are cryptographically provable, machine-verifiable, and formally auditable.

---

## This Prototype  
This repository contains a minimal Python demonstration of the ApexLex ARC 12-line control loop, featuring:

- Gaussian belief update (toy dynamics)
- A simple Lagrangian minimizer stub using a proximal QP
- A safety projector implemented as a projection QP (cvxpy + OSQP)
- Append-only signed log chain using Ed25519

### Files
- `arc_loop.py` — main prototype
- `requirements.txt` — dependencies
- `tests/test_projection.py` — projector test
- `tests/test_log_chain.py` — log/signature test

### Run (recommended in virtualenv)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python arc_loop.py
```

### Run tests
```powershell
pip install pytest
pytest -q
```

### Notes
This prototype is educational and R&D-stage, consistent with AxiomHive’s philosophy of verifiable cognition and deterministic control. Replace stubbed components with secure production-grade alternatives for real-world deployment.

---