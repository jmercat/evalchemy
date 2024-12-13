ARG AWS_REGION

FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    curl \
    wget \
    vim \
    libpq-dev

ENV PATH="/opt/ml/code:${PATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

RUN pip install --upgrade pip

# Explicitly install PyTorch with CUDA 12.1

COPY pyproject.toml /opt/ml/code/pyproject.toml
COPY README.md /opt/ml/code/README.md
COPY eval/chat_benchmarks/alpaca_eval /opt/ml/code/eval/chat_benchmarks/alpaca_eval
COPY eval/chat_benchmarks/MTBench /opt/ml/code/eval/chat_benchmarks/MTBench
RUN cd /opt/ml/code && pip install -e ".[eval]"
RUN cd /opt/ml/code/eval/chat_benchmarks/alpaca_eval && pip install -e .

RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
# RUN pip install torch torchvision
RUN pip uninstall flash-attn -y
RUN pip install flash-attn

# Separate copy of eval directories to help caching
# COPY . /opt/ml/code/
COPY database/ /opt/ml/code/database/
COPY eval/ /opt/ml/code/eval/


COPY secrets.env /opt/ml/code/secrets.env

RUN rm /opt/ml/code/pyproject.toml
ENV SAGEMAKER_PROGRAM=/opt/ml/code/eval/run_eval.py
