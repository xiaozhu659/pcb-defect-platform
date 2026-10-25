# PCB 缺陷检测平台镜像(部署到无 GPU 的 Linux 服务器)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    HF_ENDPOINT=https://hf-mirror.com

# 不执行 apt-get:
#   * opencv-python 需要 libGL,国内 Debian 源极不稳定 -> 后面换成 headless 版;
#   * 其余依赖均为纯 Python,无需系统库。
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# 锁文件里的 torch 是 CUDA 版,会带上十几个 nvidia-* 包(合计数 GB)。
# 服务器没有 GPU,因此这里跳过它们,改为安装 CPU 版 PyTorch。
ARG SKIP_PACKAGES="torch torchvision triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12"

RUN set -e; FLAGS=""; \
    for p in $SKIP_PACKAGES; do FLAGS="$FLAGS --no-install-package $p"; done; \
    uv sync --frozen --no-dev --no-install-project $FLAGS \
    && uv pip install --python /usr/local/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY README.md ./

RUN set -e; FLAGS=""; \
    for p in $SKIP_PACKAGES; do FLAGS="$FLAGS --no-install-package $p"; done; \
    uv sync --frozen --no-dev $FLAGS \
    && uv pip install --python /usr/local/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv pip uninstall --python /usr/local/bin/python opencv-python \
    && uv pip install --python /usr/local/bin/python opencv-python-headless

EXPOSE 8000
CMD ["uvicorn", "pcb_defect_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
