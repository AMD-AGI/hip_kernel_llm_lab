# Security Policy

This repository is part of the **AMD-AGI** GitHub organization and is
classified as **internal**. It contains HIP kernel LLM training workflows for
continued pretraining, supervised fine-tuning, reinforcement learning, and
kernel data generation. This document explains what is in scope, how to report
a vulnerability, and what to expect from the maintainers.

> Important: Please do not open public GitHub issues, public discussions, or
> pull requests for suspected security vulnerabilities. Use the private
> channels described below.

## Supported Versions

This repository tracks `main`. We support security fixes only on the latest
commit of `main`. Older tags, branches, and forks are not maintained.

| Version / branch | Receives security fixes |
| ---------------- | ----------------------- |
| `main` (latest)  | Yes                     |
| Tagged releases  | Best effort             |
| Forks            | No                      |

## What Is In Scope

Reports are welcome for issues such as:

- Remote code execution, command injection, or arbitrary file writes reachable
  through training launchers, dataset conversion scripts, RL reward code, HIP
  evaluation services, or generator tools.
- Unsafe handling of training datasets, checkpoints, model weights, config
  files, evaluation-server inputs, or generated HIP code.
- Insecure deserialization of checkpoints, parquet files, cached artifacts, or
  model outputs from untrusted sources.
- Credential leakage involving Hugging Face tokens, W&B keys, model-provider
  credentials, cloud credentials, or internal service URLs.
- Supply-chain issues in training requirements, container setup, package
  metadata, or shell scripts maintained in this repository.

## What Is Out Of Scope

- Vulnerabilities in upstream LLaMA-Factory, veRL, PyTorch, ROCm, Hugging Face,
  W&B, or other third-party dependencies. Please report those upstream.
- Denial of service from intentionally expensive training jobs, large datasets,
  or adversarial GPU workloads.
- Model quality regressions, reward instability, or failed training runs
  without a security impact.
- Issues requiring physical access to the user's machine or administrator
  access on the host.
- Findings from automated scanners without a working proof of concept.

## How To Report A Vulnerability

Because this is an internal AMD-AGI repository, use one of the following
private channels:

1. **AMD Security (recommended).** Report through AMD's product security
   channel: <https://www.amd.com/en/resources/product-security.html>. Internal
   AMD employees should follow the IT security incident-reporting process via
   the AMD intranet.
2. **Email or contact the repository maintainers** listed in
   [`.github/CODEOWNERS`](./.github/CODEOWNERS).

Please include, where possible:

- A clear description of the issue and its impact.
- The affected file(s), commit SHA, command line, and environment details.
- Step-by-step reproduction instructions and a minimal proof of concept.
- Any logs, stack traces, configs, or artifacts that help us reproduce. Redact
  secrets before sharing.
- Your name or handle for credit, or a note that you wish to stay anonymous.

## Our Commitments

| Stage                       | Target                                  |
| --------------------------- | --------------------------------------- |
| Acknowledge receipt         | within 3 business days                  |
| Initial assessment / triage | within 10 business days                 |
| Status update cadence       | at least every 14 days until closed     |
| Fix or mitigation           | as quickly as severity allows           |

We will coordinate disclosure with you and will not take adverse action
against people who follow this policy, act in good faith, and avoid privacy
violations, data destruction, or service disruption.

## Handling Secrets

If you discover that a secret has been committed to this repository:

1. Do not post the secret in an issue, pull request, or discussion.
2. Report it through the private channels above.
3. We will rotate or revoke the secret, rewrite history if needed, and add a
   regression check where practical.

## Hardening Guidance

- Treat datasets, model weights, checkpoints, generated HIP code, and cached
  outputs from untrusted sources as untrusted input.
- Restrict RL evaluation services to trusted networks and users.
- Avoid committing Hugging Face, W&B, cloud, or internal service credentials.
- Review training logs, configs, and artifacts before sharing them outside the
  repository team.

Thank you for helping keep `hip_kernel_llm_lab` and AMD's AI infrastructure
safe.
