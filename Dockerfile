#FROM nvcr.io/nvidia/pytorch:19.09-py3
FROM nvcr.io/nvidia/pytorch:20.03-py3
RUN apt-get update && apt-get install -y fish geeqie
RUN pip --version
RUN cd / && git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd /torch2trt && python setup.py install
#WORKDIR /temp
#COPY requirements.txt /temp
#RUN pip install -r requirements.txt
#RUN pip install kornia==0.2.0 numpy==1.18.1 pytorch_lightning==0.7.6rc1 torchvision==0.6.0 imageio==2.6.1 torch==1.5.0 Pillow==7.1.2
#RUN pip install kornia==0.2.0 numpy==1.18.1 pytorch_lightning==0.8.1 torchvision==0.6.0 imageio==2.6.1 torch==1.5.0 Pillow==7.1.2
RUN conda remove wrapt
RUN pip install msgpack==0.5.6 wrapt==1.10.0
#RUN pip install numpy==1.18.1 pytorch_lightning==0.7.1 torchvision==0.6.0 imageio==2.6.1 Pillow==7.1.2
RUN pip install numpy==1.18.1 pytorch_lightning==0.7.1 torchvision==0.8.2 imageio==2.6.1 Pillow==7.1.2
#RUN ls
#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/TensorRT-6.0.1.5/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
#ADD TensorRT-6.0.1.5 /TensorRT-6.0.1.5
#RUN cd /TensorRT-6.0.1.5/python && pip install tensorrt-6.0.1.5-cp36-none-linux_x86_64.whl
#RUN conda remove wrapt
RUN pip install einops==0.3.0 yacs==0.1.8 kornia==0.4.1 opencv-python==4.3.0.36 tensorflow-gpu==1.15
RUN apt-get update && apt-get install -y ffmpeg gifsicle webp
CMD ["fish"]
