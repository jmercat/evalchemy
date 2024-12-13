ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONPATH="/opt/ml/code/eval:${PYTHONPATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

COPY . /opt/ml/code/
WORKDIR /opt/ml/code
RUN pip install -e ".[eval]"
RUN pip install -e eval/chat_benchmarks/alpaca_eval
RUN pip uninstall torch -y
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab

RUN cp /opt/ml/code/eval/eval.py /opt/ml/code/eval.py
