FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install python3.8 python3-pip nano git wget software-properties-common -y
RUN useradd rd -p "$(openssl passwd -1 SilverLaptop)" -s /bin/bash && mkdir /home/rd && chown rd:rd /home/rd

WORKDIR MGCN/
USER rd
ADD . ./
USER root

RUN chmod 777 -R bin/
RUN mkdir -p /ai-research/notebooks/testing_repos/

RUN pip3 install -r ./requirements.txt
# The following installations didnt work with the previous requirement file.
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

