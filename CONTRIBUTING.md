# Contributing to hip_kernel_llm_lab

Thanks for your interest in `hip_kernel_llm_lab`. This repository contains HIP
kernel LLM training workflows for continued pretraining, supervised fine-tuning,
reinforcement learning, and kernel data generation.

By participating in this project you agree to follow these guidelines and the
project's [LICENSE](./LICENSE).

## Code Of Conduct

Be respectful, constructive, and inclusive. Maintainers may close issues or
pull requests that are disruptive, hostile, or unrelated to the project.

## Ways To Contribute

- Training recipe, launcher, and configuration improvements for `cpt`, `sft`,
  and `rl`.
- HIP kernel data-generation and evaluation workflow improvements.
- Dataset registration, conversion, and documentation fixes.
- Reproducibility, logging, environment setup, and troubleshooting updates.

## Reporting Bugs

Open an issue with:

- A clear title and short description.
- Environment details: OS, Python, PyTorch, ROCm, GPU model, training
  framework, and relevant container or launcher.
- The stage you ran: `cpt`, `sft`, `rl`, or `hip-kernel-generator`.
- Exact command(s), config files, logs, and full error or stack trace.
- Expected versus actual behavior.

## Requesting Features

Open an issue describing the training stage, target dataset or workflow,
proposed config/CLI surface, and whether you are willing to implement it.

## Security Issues

Do not file public issues for vulnerabilities. Follow the private reporting
process in [SECURITY.md](./SECURITY.md).

## Development Setup

Follow the stage-specific README for the area you are changing:

- `cpt/`
- `sft/`
- `rl/`
- `hip-kernel-generator/README.md`

For dataset-backed workflows, download datasets outside Git-tracked paths and
avoid committing local credentials or model weights.

## Validation

Include the smallest useful validation for your change:

- For `cpt` or `sft`, include the launcher/config used and whether the job
  reached startup, first step, or completion.
- For `rl`, include the reward/evaluation-server command and training launcher.
- For `hip-kernel-generator`, include the package install and CLI validation
  command from that subproject.
- For docs-only changes, include the paths you checked.

## Pull Request Workflow

1. Branch from the latest `main`.
2. Keep the change focused on one training stage, generator workflow, or docs
   area.
3. Update stage-specific READMEs when configs, datasets, or commands change.
4. Include validation commands and environment details in the PR.
5. Request review from the owners listed in
   [`.github/CODEOWNERS`](./.github/CODEOWNERS).

## PR Checklist

- [ ] Change is rebased on the latest `main`.
- [ ] No unrelated formatting churn or large generated artifacts.
- [ ] No Hugging Face tokens, W&B keys, cloud credentials, checkpoints, model
      weights, or private datasets are committed.
- [ ] Relevant README, config documentation, and example commands are updated.
- [ ] Validation commands and logs summary are included in the PR.

## Coding Style

- Follow the local style in the stage you are modifying.
- Keep training configs explicit and reproducible.
- Avoid hard-coded local paths, usernames, tokens, or internal service URLs.
- Prefer clear comments for non-obvious distributed-training or ROCm-specific
  behavior.

## Commit Messages And Sign-Off

Use clear, focused commit messages. We prefer Conventional Commits where they
fit, such as `docs(rl): clarify HIP evaluation server setup`.

Every commit should include a Developer Certificate of Origin sign-off:

```text
Signed-off-by: Your Name <you@example.com>
```

You can add it with `git commit -s`.

## Reviews And Code Owners

Repository ownership is defined in
[`.github/CODEOWNERS`](./.github/CODEOWNERS). At least one relevant owner should
review changes before merge.

## License Of Contributions

By submitting a contribution you agree that your work is licensed under the
[Apache License 2.0](./LICENSE), and you confirm that you have the right to
submit it.
