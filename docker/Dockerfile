# Change according to the local verion of CUDA
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

# Install python3
# RUN apt install -y python3 python3-pip
RUN apt-get install -y    \
        git \
        wget \ 
        unzip \
        python3-pip \ 
        vim \
        feh \
        curl

RUN mkdir -p /workspace

RUN pip install segmentation-models-pytorch matplotlib pytorch-lightning roboflow onnx albumentations
WORKDIR /workspace/src

