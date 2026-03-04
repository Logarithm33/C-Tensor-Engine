FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 仅安装最核心的编译工具、Python 运行环境和内存检测工具
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    build-essential cmake gdb valgrind \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 NumPy 用于后续的对比实验
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy

WORKDIR /workspace

# 保持容器运行，不执行特定任务，等待 VS Code 接入
CMD ["tail", "-f", "/dev/null"]