#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONFIG_PATH="$ROOT_DIR/model/pretrain_qwen3_5_200m.json"

python -m model.train --config "$CONFIG_PATH"
