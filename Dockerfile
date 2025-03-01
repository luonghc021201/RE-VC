# FROM harbor.infra.zalo.services/public/python:3.8.10
FROM harbor.infra.zalo.services/ailab/cuda11.6-devel-ubuntu20.04

ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y python3-opencv vim net-tools curl
RUN apt install ffmpeg -y 
RUN pip install --upgrade pip setuptools wheel

ENV APP_DIR=DDDM_VCs
WORKDIR $APP_DIR

RUN pip3 install torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip3 install protobuf==3.20.*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install nvitop
RUN pip3 install matplotlib
RUN pip3 install tensorboard
RUN pip3 install transformers

ENV http_proxy 'http://10.60.28.99:81'
ENV https_proxy 'http://10.60.28.99:81'
# ENV http_proxy 'http://10.40.34.14:81'
# ENV https_proxy 'http://10.40.34.14:81'

COPY . .

ENTRYPOINT python train_ms.py -c configs/vits_vc.json -m /data/luonghc/checkpoint_vits_vc

