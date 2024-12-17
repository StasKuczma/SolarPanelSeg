FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

RUN apt update
RUN apt install -y    \
        git wget bash-completion unzip python3-pip 


RUN mkdir -p /workspace
WORKDIR /workspace


RUN git clone https://github.com/PUTvision/LandCoverSeg.git
WORKDIR /workspace/LandCoverSeg

RUN pip3 install -r requirements.txt