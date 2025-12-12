# 音频转录服务

基于 FunASR 的音频转录 API 服务，返回模拟 B 站字幕接口格式的数据。

## 功能特性

- 🎯 支持多种音频格式转录
- 🌍 自动语言检测（中文、英文、日文、韩文）
- ⚡ GPU/CPU 自适应，显存不足自动降级
- 🎬 返回 B 站字幕格式的 JSON 数据
- 🔄 自动资源管理，闲置释放模型
- 🔒 可选的 API 访问令牌认证

## 快速开始

### 环境要求

- Python 3.8+
- 可选：CUDA 支持（用于 GPU 加速）
- Conda 环境（推荐）

### 安装依赖

#### 方式1：使用 Conda（推荐）

1. **创建并激活 Conda 环境：**
   ```bash
   conda create -n funasr python=3.11 -y
   conda activate funasr
   ```

2. **安装 PyTorch（根据你的 CUDA 版本选择）：**
   ```bash
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # CUDA 11.7
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

   # CPU 版本
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # AMD GPU (ROCm)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   ```

3. **安装项目依赖：**
   ```bash
   pip install -r requirements.txt
   ```

#### 方式2：使用 pip（直接安装）

```bash
# 1. 安装 PyTorch（选择适合你的版本）
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 安装项目依赖
pip install -r requirements.txt
```

#### 方式3：使用 conda-forge（可选）

```bash
# 安装 PyTorch（通常版本较旧，但简单）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装项目依赖
pip install -r requirements.txt
```

**重要提示**：
- `requirements.txt` 不包含 PyTorch，因为不同用户需要不同的 CUDA 版本
- 请根据你的硬件配置选择合适的 PyTorch 版本
- 如果不确定 CUDA 版本，可运行 `nvidia-smi` 查看

#### Conda 环境管理说明

**Conda 是什么？**
Conda 是一个开源的包管理器和环境管理器，可以轻松安装不同版本的软件包及其依赖关系，并在它们之间切换。

**为什么推荐使用 Conda？**
- 环境隔离：避免不同项目的依赖冲突
- 跨平台：支持 Windows、macOS 和 Linux
- 科学计算优化：专门针对数据科学和机器学习优化
- CUDA 管理：更容易管理不同版本的 CUDA 环境

**常用 Conda 命令：**
```bash
# 创建新环境
conda create -n 环境名 python=版本号

# 激活环境
conda activate 环境名

# 退出环境
conda deactivate

# 删除环境
conda remove -n 环境名 --all

# 查看所有环境
conda env list

# 导出环境配置
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

### 配置服务

1. **复制配置文件：**
   ```bash
   cp config.yaml.example config.yaml
   ```

2. **根据需要修改配置文件：**
   ```yaml
   # 服务器配置
   server:
     idle_timeout: 300
     check_interval: 10

   # 模型配置
   model:
     name: "paraformer-zh"
     vad_model: "fsmn-vad"
     punc_model: "ct-punc"

   # API配置
   api:
     host: "0.0.0.0"
     port: 8000
     token: ""         # API访问令牌，空表示不需要验证
   ```

### 运行服务

```bash
# 方式1：直接运行
python server.py

# 方式2：使用启动脚本（推荐）
bash run.sh
```

服务将在配置的地址和端口启动（默认：`http://0.0.0.0:8000`）。

## API 接口

### POST /transcribe

上传音频文件进行转录。

**请求参数：**
- `file`: 音频文件（multipart/form-data）

**请求头（可选）：**
- `Authorization`: Bearer token（如果配置了token则需要）

**响应格式：**

```json
{
  "font_size": 0.4,
  "font_color": "#FFFFFF",
  "background_alpha": 0.5,
  "background_color": "#9C27B0",
  "Stroke": "none",
  "type": "manual_transcribe",
  "lang": "zh",
  "version": "v1",
  "body": [
    {
      "from": 0.0,
      "to": 3.0,
      "sid": 1,
      "location": 2,
      "content": "转录文本片段",
      "music": 0
    }
  ],
  "device_used": "cpu",
  "status": "success"
}
```

**字段说明：**

- `font_size`: 字体大小
- `font_color`: 字体颜色
- `background_alpha`: 背景透明度
- `background_color`: 背景颜色
- `type`: 字幕类型（manual_transcribe）
- `lang`: 检测到的语言代码（zh, en, ja, ko）
- `version`: 接口版本
- `body`: 字幕内容数组
  - `from`: 开始时间（秒）
  - `to`: 结束时间（秒）
  - `sid`: 字幕序号
  - `location`: 位置（2=底部）
  - `content`: 字幕文本
  - `music`: 是否为音乐（0=否）
- `device_used`: 使用的设备（cpu/cuda）
- `status`: 处理状态（success/error）

## 使用示例

### Python 客户端

```python
import requests

# 上传音频文件
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    headers = {}

    # 如果配置了token，添加Authorization头
    # headers['Authorization'] = 'Bearer your_token_here'

    response = requests.post('http://localhost:8000/transcribe', files=files, headers=headers)

result = response.json()
print(f"检测语言: {result['lang']}")
for subtitle in result['body']:
    print(f"{subtitle['from']:.1f}s - {subtitle['to']:.1f}s: {subtitle['content']}")
```

### 带 Token 认证的 Python 客户端

```python
import requests

# 配置token
token = "your_token_here"
headers = {
    'Authorization': f'Bearer {token}'
}

# 上传音频文件
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/transcribe', files=files, headers=headers)

result = response.json()
print(f"检测语言: {result['lang']}")
for subtitle in result['body']:
    print(f"{subtitle['from']:.1f}s - {subtitle['to']:.1f}s: {subtitle['content']}")
```

### cURL 示例

```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.mp3"
```

### 带 Token 认证的 cURL 示例

```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -H "Authorization: Bearer your_token_here" \
     -F "file=@audio.mp3"
```

## 配置说明

### 配置文件结构

`config.yaml` 文件包含以下配置项：

#### 服务器配置
```yaml
server:
  idle_timeout: 300      # 模型闲置超时时间（秒）
  check_interval: 10     # 模型状态检查间隔（秒）
```

#### 模型配置
```yaml
model:
  name: "paraformer-zh"  # 核心识别模型
  vad_model: "fsmn-vad"  # VAD模型
  punc_model: "ct-punc"  # 标点模型
  disable_update: true   # 禁用模型更新检查
```

#### 处理配置
```yaml
processing:
  batch_size_s: 300                     # 批处理大小（秒）
  max_segment_length: 20                # 字幕最大长度
  duration_per_segment: 3.0             # 每段字幕持续时间
  chinese_ratio_threshold: 0.3          # 中文比例阈值
```

#### 字幕样式配置
```yaml
subtitle:
  font_size: 0.4
  font_color: "#FFFFFF"
  background_alpha: 0.5
  background_color: "#9C27B0"
  stroke: "none"
  type: "manual_transcribe"
  version: "v1"
```

#### API配置
```yaml
api:
  host: "0.0.0.0"  # 监听地址
  port: 8000       # 监听端口
  token: ""        # API访问令牌，空表示不需要验证
```

**API配置说明：**
- `host`: 服务器监听的IP地址，0.0.0.0表示监听所有网络接口
- `port`: 服务器监听的端口号
- `token`: API访问令牌，用于客户端认证
  - 留空（默认）：不需要认证，任何人都可以访问API
  - 设置值：客户端需要在请求头中添加`Authorization: Bearer <token>`才能访问

## 语言支持

- `zh`: 中文
- `en`: 英文
- `ja`: 日文
- `ko`: 韩文

语言检测基于转录文本的字符特征自动判断。

## 项目结构

```
transcribe-service/
├── server.py              # 主服务文件
├── config.py              # 配置管理模块
├── config.yaml.example    # 配置文件模板
├── config.yaml           # 实际配置文件（需要从模板复制，已被git忽略）
├── requirements.txt       # Python依赖
├── run.sh               # 启动脚本
├── .gitignore           # Git忽略文件
├── test/               # 测试目录
│   └── test.mp3       # 测试音频文件
└── README.md           # 项目说明
```

> **注意**：`config.yaml` 文件已在 `.gitignore` 中被忽略，因为包含个人配置信息。使用时需要从 `config.yaml.example` 复制。

## 注意事项

1. **首次运行**：会自动下载模型文件，可能需要较长时间
2. **配置文件**：必须复制 `config.yaml.example` 为 `config.yaml` 才能启动
3. **PyTorch安装**：`requirements.txt` 不包含PyTorch，需要根据CUDA版本手动安装
4. **内存管理**：服务会自动管理内存，闲置超时后释放模型资源
5. **GPU支持**：支持 CUDA 加速，显存不足会自动切换到 CPU
6. **音频格式**：支持常见音频格式，内部使用 FFmpeg 进行音频处理
7. **网络访问**：首次运行需要访问 ModelScope 下载模型
8. **CUDA版本检查**：运行 `nvidia-smi` 查看支持的CUDA版本

## 开发指南

### 环境准备
```bash
# 1. 克隆项目
git clone https://github.com/Zhenyi-Wang/transcribe-service.git
cd transcribe-service

# 2. 设置环境
cp config.yaml.example config.yaml
# 根据需要修改 config.yaml

# 3. 安装依赖
pip install -r requirements.txt
```

### 本地测试
```bash
# 测试API
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@test/test.mp3"
```

## 贡献指南

1. Fork 本项目
2. 创建特性分支：`git checkout -b feature/YourFeature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送分支：`git push origin feature/YourFeature`
5. 提交 Pull Request

## 许可证

MIT License

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 语音识别框架
- [ModelScope](https://modelscope.cn/) - 模型仓库