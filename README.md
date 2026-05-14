[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat)](LICENSE)

# hip-kernel-llm-lab

> A HIP kernel LLM training framework covering continued pretraining, supervised fine-tuning, and reinforcement learning.

`hip-kernel-llm-lab` is a repository for training and evolving LLMs toward HIP
kernel generation, understanding, optimization, and related engineering tasks.
The project is organized around three training stages:

1. Continued pretraining (`cpt`)
2. Supervised fine-tuning (`sft`)
3. Reinforcement learning (`rl`)

At the moment, the repository mainly provides the top-level structure and
ownership definition for these stages. Stage-specific code has not been checked
in yet and will be added incrementally.

## Repository Layout

```text
hip_kernel_llm_lab/
├── cpt/        # continued pretraining assets and code
├── sft/        # supervised fine-tuning assets and code
├── rl/         # reinforcement learning assets and code
├── examples/   # example data, recipes, and usage samples
├── .github/    # repository metadata such as CODEOWNERS
├── LICENSE
└── README.md
```

## Training Scope

### `cpt`

The continued pretraining stage is intended for domain adaptation on HIP
kernel-related corpora, including source code, optimization patterns, compiler
knowledge, runtime usage, and associated engineering context.

### `sft`

The supervised fine-tuning stage is intended for instruction-following behavior
on HIP kernel tasks such as code generation, code transformation, debugging,
optimization suggestions, and reasoning over kernel implementations.

### `rl`

The reinforcement learning stage is intended for further improving model
behavior using task-specific rewards, such as correctness, compilability,
performance-oriented preferences, or other training objectives relevant to HIP
kernel workflows.

## Planned Workflow

The intended high-level training flow is:

1. Use continued pretraining to adapt the base model to HIP kernel and
   GPU-system-specific knowledge.
2. Use supervised fine-tuning to teach task-oriented instruction behavior.
3. Use reinforcement learning to optimize for downstream quality signals such as
   correctness, preference alignment, and performance-related objectives.

## Current Status

This repository is currently in an early layout stage.

- The root structure for `cpt`, `sft`, `rl`, and `examples` is already in place.
- Stage-specific implementations, configs, scripts, and detailed usage docs are
  not in the repository yet.
- As code is added, each stage directory can grow its own dedicated README and
  operational documentation.

## Code Ownership

Current repository ownership is defined in [`.github/CODEOWNERS`](.github/CODEOWNERS).

Temporary owners for the repository are:

- `@AMD-AGI/AI-Algorithm`
- `@liuji` `@zepingli` `@chushi` `@zihao` `@puyuan`

Because the code for each training stage has not been added yet, ownership is
currently maintained at the repository level. More granular ownership can be
introduced after `cpt`, `sft`, and `rl` receive their stage-specific code.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
