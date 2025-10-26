# PRINCIPLES.md (How)

This document distills actionable lessons from the ApexLex ARC prototype into compact Observations → Principles → Implications entries.

1) Observation: Systems fail silently when safety is an afterthought.
   Principle: Safety must be embedded in system structure and enforced by deterministic checks (projectors, barriers).
   Implication: All decision paths end with an explicit safety projection and an auditable decision record. Fallback policies must exist and be tested.

2) Observation: Alignment objectives can dominate optimization and cause numerical issues.
   Principle: Treat lexicographic priorities with a staged solver or proximal regularization rather than extreme scalarization.
   Implication: Implement a two-stage solve (alignment feasibility → optimality) and retain warm-started duals for fast convergence.

3) Observation: Auditability is only useful if logs are tamper-evident and verifiable.
   Principle: Use cryptographic primitives (hash-chains, Merkle roots, signatures) and store minimal cleartext with hashed payloads for PII.
   Implication: Every runtime decision writes a signed log entry that includes solver metadata and justification; periodic root anchoring is required for long-term integrity.

4) Observation: Prototypes often leak unsafe defaults (in-memory keys, no rotation).
   Principle: Explicitly mark prototype-grade components and require migration plans to production-grade equivalents (HSM/KMS, encrypted storage).
   Implication: Deployment docs must list exact modules needing hardening with migration steps and risks.

5) Observation: Verifiability requires machine-checkable artifacts.
   Principle: Provide JSON Schemas, CI checks, and integrity attestation that cryptographically binds docs to artifacts.
   Implication: Include `VALIDATION/` with schemas and a CI workflow that runs tests and schema checks on every push.

Version: v0.1
