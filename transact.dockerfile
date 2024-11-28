FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


RUN apt-get update -y && \
    apt-get install -y lsb-release openssh-server && \
    apt-get install -y curl tree git bc && \
    apt-get install -y python3.10 python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    apt-get install -y python3.10-venv python3.10-dev && \
    apt-get clean all && rm -rf /var/lib/apt/lists/


RUN pip install --no-cache-dir torch>=2.2.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir transformers>=4.39.2 peft accelerate datasets>=2.18.0 evaluate 

RUN pip install --no-cache-dir deepspeed>=0.13.5

RUN pip install --no-cache-dir flash-attn

RUN pip install --no-cache-dir accuracy sentencepiece sacrebleu sqlitedict omegaconf pycountry rouge_score pytablewriter scikit-learn scipy sentencepiece protobuf

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
