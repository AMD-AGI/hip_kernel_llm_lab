# torch_modu2func_kit

`torch_modu2func_kit` is a production-oriented refactor of the original
`torch_modu2func` workflow.

It converts every PyTorch module file under an input tree such as
`kernelbench_torch_modu` into a verified functional implementation under an
output tree such as `kernelbench_torch_func`.

## Highlights

- Keeps the `models/` provider interface unchanged.
- Preserves the original directory structure and file names on successful
  conversion.
- Retries each sample up to a configurable attempt limit.
- Injects failed attempt history into the next prompt, including the previous
  candidate code and the captured failure message or traceback.
- Verifies correctness by comparing original and generated outputs with
  configurable `rtol` and `atol`.
- Writes structured JSON records for successful and failed samples.
- Persists prompt and candidate artifacts for every attempt to simplify
  debugging and prompt iteration.

## How It Works

For each `.py` sample under the input tree:

1. Read the original module file.
2. Build a prompt from the system instruction, few-shot examples, and the
   original source code.
3. Call the selected LLM provider through the unchanged `models/` interface.
4. Verify the generated file by importing both the original and generated
   modules, calling `get_init_inputs()`, `get_inputs()`, and `Model.forward()`,
   and comparing outputs with `torch.allclose`.
5. If verification fails, append the failed candidate code and the observed
   failure details to the next attempt's prompt.
6. Stop early on the first successful conversion, otherwise move on after
   `--max-attempts` failures.

## Output Layout

Assuming `--artifacts-dir .artifacts`, the pipeline writes:

- `kernelbench_torch_func/...`: final verified functional files
- `.artifacts/prompts/...`: the exact prompt used for each attempt
- `.artifacts/candidates/...`: raw generated candidates for each attempt
- `.artifacts/conversion_records.json`: all records
- `.artifacts/successful_conversions.json`: successful samples only
- `.artifacts/failed_conversions.json`: failed samples only

Each attempt record stores prompt path, candidate path, status, and captured
feedback such as mismatch details or exception tracebacks.

## Quick Start

Install locally:

```bash
pip install -e .[dev]
```

Run the pipeline:

```bash
torch-modu2func \
  --input-dir ../kernelbench_torch_modu \
  --output-dir ../kernelbench_torch_func \
  --artifacts-dir .artifacts \
  --provider openai \
  --model-id dvue-aoai-001-gpt-5 \
  --api-key YOUR_API_KEY
```

Or use environment variables:

```bash
export TORCH_MODU2FUNC_API_KEY=YOUR_API_KEY
torch-modu2func --input-dir ../kernelbench_torch_modu --output-dir ../kernelbench_torch_func
```

## Important Arguments

- `--max-attempts`: maximum retries per sample, default `5`
- `--rtol`: relative tolerance for output comparison, default `1e-4`
- `--atol`: absolute tolerance for output comparison, default `1e-4`
- `--seed`: deterministic verification seed, default `1234`
- `--temperature`: model sampling temperature
- `--max-tokens`: maximum tokens per generation call
- `--overwrite`: rewrite existing successful outputs

The pipeline also supports prompt history truncation through
`PipelineConfig.history_code_char_limit` and
`PipelineConfig.history_feedback_char_limit` to avoid oversized retry prompts.

## Development

Run unit tests:

```bash
python -m pytest
```
