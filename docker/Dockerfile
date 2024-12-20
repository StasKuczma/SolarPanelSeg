# Change according to the local verion of CUDA
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt update
RUN apt install -y    \
        git wget bash-completion unzip python3-pip vim feh curl

RUN mkdir -p /workspace

RUN pip install segmentation-models-pytorch matplotlib pytorch-lightning roboflow
WORKDIR /workspace/src

