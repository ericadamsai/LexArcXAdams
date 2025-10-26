# DEPLOYMENT.md (Now)

This document lists exact steps to set up, run, observe, and roll back the ApexLex ARC prototype. It explicitly marks prototype-grade components and gives actionable migration notes.

Prerequisites (local developer machine, Windows PowerShell)
- Python 3.11+
- PowerShell (Windows) or bash (Linux/Mac)
- Optional: access to an HSM/KMS for production keys

Quick-start (development)

```powershell
# create venv and activate
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install --upgrade pip
pip install -r ARTIFACTS/src/requirements.txt

# run prototype loop
python ARTIFACTS/src/arc_loop.py

# run tests
$env:PYTHONPATH='ARTIFACTS/src'; pytest -q ARTIFACTS/tests
```

Prototype-grade components (explicit containment)
- In-memory Ed25519 private key (file: `ARTIFACTS/src/arc_loop.py`) — non-production. MIGRATION: use HSM/KMS, store only public keys in repo.
- Minimized solver stub `minimize_lagrangian_stub` — replace with SQP/MPC implementation (see `PRINCIPLES.md`).

Production hardening checklist
- Replace in-memory private keys with HSM-backed signing or KMS (AWS KMS, Azure Key Vault). Use envelope encryption for logs.
- Migrate solver to an optimized runtime (C++ or Rust) or call to a vetted solver service; profile CPU and memory for target control loop rate.
- Implement secure logging pipeline: signed logs persisted to immutable storage, replicated, and periodically anchored to an external notarization service.
- Add role-based access control for audit data and rotate keys regularly.

Rollback and emergency procedures
- If tests fail on deploy, revert commit and restart CI. Maintain blue/green deployment only when integrating with services.
- If signer key is suspected compromised, rotate key in KMS, re-sign subsequent roots, and record a revocation event in the log.

Cost/latency notes
- Prototype dependencies are CPU-light for small models; production MPC or high-dimensional belief filtering will increase CPU/GPU needs. Profile in a staging environment.

Version: v0.1
