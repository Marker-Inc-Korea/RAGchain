FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get -y install python3-pip
RUN alias pip=pip3
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt-get install -y curl git wget --fix-missing
RUN mkdir workspace
WORKDIR ./workspace
RUN git clone https://github.com/edai-club/KoPrivateGPT.git
WORKDIR ./KoPrivateGPT
RUN pip3 install -r requirements.txt
RUN pip3 install -r ./webui/requirements.txt

EXPOSE 7860

ENTRYPOINT ["gradio", "app.py"]
