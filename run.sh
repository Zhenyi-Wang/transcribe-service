#!/bin/bash

# 1. 激活 Conda 环境
# 注意：请根据你的系统修改 Conda 路径
source ~/miniconda3/etc/profile.d/conda.sh
# 或者如果使用系统默认 Python，可以注释掉上面两行

# 激活环境
conda activate funasr

# 2. 进入脚本所在目录
cd "$(dirname "$0")"

# 3. 启动 Python 服务
exec python server.py