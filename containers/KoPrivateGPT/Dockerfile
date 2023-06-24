FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get -y install python3-pip
RUN alias pip=pip3
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt-get update
RUN apt-get install -y curl git wget vim
RUN mkdir workspace
RUN cd workspace
RUN git clone https://github.com/edai-club/KoPrivateGPT.git
RUN cd KoPrivateGPT
RUN pip3 install -r requirements.txt

EXPOSE 7860

CMD ["gradio", "app.py"]
