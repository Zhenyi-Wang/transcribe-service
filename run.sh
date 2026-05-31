#!/bin/bash

# 1. 激活 Conda 环境
# 注意：请根据你的系统修改 Conda 路径
source ~/miniconda3/etc/profile.d/conda.sh
# 或者如果使用系统默认 Python，可以注释掉上面两行

# 激活环境（注意：Qwen3-ASR 需要 Python 3.12+）
conda activate funasr

# 2. 进入脚本所在目录
cd "$(dirname "$0")"

# 设置 llama.cpp 共享库路径（libggml.so 依赖带版本号的 .so.0）
export LD_LIBRARY_PATH="$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 3. 启动 Python 服务（默认启用自动重载）
# 使用 --no-reload 禁用自动重载
if [ "$1" == "--no-reload" ]; then
    exec python server.py
else
    exec python -m uvicorn server:app --host 0.0.0.0 --port 31080 --reload
fi