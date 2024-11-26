FROM ubuntu:22.04

WORKDIR /workspace

# 安装 Python 3.10 和常用工具
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 升级 Python 包管理工具 pip
RUN python3.10 -m pip install --upgrade pip

# 设置默认的执行命令
CMD ["bash"]

