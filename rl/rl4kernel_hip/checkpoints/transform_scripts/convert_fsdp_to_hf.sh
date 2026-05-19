#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT_PATH=""
CONFIG_DIR=""
OUTPUT_DIR=""
LOAD_WORKERS=8
REPO_ID=""
PATH_IN_REPO=""
TOKEN="${HF_TOKEN:-}"
UPLOAD_ONLY=false

usage() {
    echo "Usage: $0 --ckpt CHECKPOINT_PATH [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --ckpt PATH         global_step_xxx 目录，或直接 actor 目录"
    echo ""
    echo "Optional:"
    echo "  --config_dir PATH   HF config/tokenizer 目录 (默认: <actor>/huggingface)"
    echo "  --output PATH       本地输出目录 (默认: checkpoints/converted_hf_models/<exp>/<step>)"
    echo "  --load_workers N    并行加载 shard 数 (默认: 8)"
    echo "  --repo_id ID        HuggingFace repo id (启用上传时使用)"
    echo "  --path_in_repo PATH HuggingFace 仓库子路径"
    echo "  --token TOKEN       HuggingFace token (默认读 HF_TOKEN 环境变量)"
    echo "  --upload_only       仅上传，不重新转换"
    echo "  -h, --help          显示帮助"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --config_dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --load_workers)
            LOAD_WORKERS="$2"
            shift 2
            ;;
        --repo_id)
            REPO_ID="$2"
            shift 2
            ;;
        --path_in_repo)
            PATH_IN_REPO="$2"
            shift 2
            ;;
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --upload_only)
            UPLOAD_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Error: --ckpt is required."
    usage
    exit 1
fi

if [[ ! -d "$CKPT_PATH" ]]; then
    echo "Error: checkpoint path not found: $CKPT_PATH"
    exit 1
fi

shopt -s nullglob
shard_files=("$CKPT_PATH"/model_world_size_*_rank_0.pt)
shopt -u nullglob

if (( ${#shard_files[@]} > 0 )); then
    ACTOR_DIR="$CKPT_PATH"
elif [[ -d "$CKPT_PATH/actor" ]]; then
    ACTOR_DIR="$CKPT_PATH/actor"
else
    echo "Error: no model shards found under '$CKPT_PATH' or '$CKPT_PATH/actor'."
    exit 1
fi

if [[ -z "$CONFIG_DIR" ]]; then
    CONFIG_DIR="$ACTOR_DIR/huggingface"
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: config dir not found: $CONFIG_DIR"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    step_name="$(basename "$CKPT_PATH")"
    exp_name="$(basename "$(dirname "$CKPT_PATH")")"
    OUTPUT_DIR="$SCRIPT_DIR/../converted_hf_models/$exp_name/$step_name"
fi

mkdir -p "$OUTPUT_DIR"

echo "🚀 启动 FSDP -> HF 转换"
echo "ACTOR_DIR:      $ACTOR_DIR"
echo "CONFIG_DIR:     $CONFIG_DIR"
echo "OUTPUT_DIR:     $OUTPUT_DIR"
echo "LOAD_WORKERS:   $LOAD_WORKERS"
[[ -n "$REPO_ID" ]] && echo "REPO_ID:        $REPO_ID"
[[ -n "$PATH_IN_REPO" ]] && echo "PATH_IN_REPO:   $PATH_IN_REPO"
[[ "$UPLOAD_ONLY" == true ]] && echo "UPLOAD_ONLY:    true"
echo "----------------------------------------"

CMD=(
    python "$SCRIPT_DIR/fast_convert_upload.py"
    --ckpt "$ACTOR_DIR"
    --config_dir "$CONFIG_DIR"
    --output "$OUTPUT_DIR"
    --load_workers "$LOAD_WORKERS"
)

if [[ "$UPLOAD_ONLY" == true ]]; then
    CMD+=(--upload_only)
fi
if [[ -n "$REPO_ID" ]]; then
    CMD+=(--repo_id "$REPO_ID")
fi
if [[ -n "$PATH_IN_REPO" ]]; then
    CMD+=(--path_in_repo "$PATH_IN_REPO")
fi
if [[ -n "$TOKEN" ]]; then
    CMD+=(--token "$TOKEN")
fi

"${CMD[@]}"

echo "✅ 转换完成。输出目录: $OUTPUT_DIR"
