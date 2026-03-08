FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    build-essential cmake gdb valgrind \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy

WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]