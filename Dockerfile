FROM tensorflow/tensorflow:1.15.2-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

WORKDIR /root/tensorflow

# Copy this version of of the model garden into the image
COPY ./models /root/tensorflow/models

# Compile protobuf configs
RUN (cd /root/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /root/tensorflow/models/research/

RUN cp object_detection/packages/tf1/setup.py ./
ENV PATH="/root/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

# fix bug: No such file or directory (libGL.so.1)
RUN apt-get install -y libgl1-mesa-glx

ENV TF_CPP_MIN_LOG_LEVEL 3
