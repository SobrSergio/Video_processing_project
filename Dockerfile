FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app

RUN mkdir -p /app/uploads /app/processed

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app", "--workers", "4"]

#Если хотите использовать CUDA, то вот примерный код. 
# (Для каждого компьютера и системы нужно будет использовать разные методы. Cuda очень проблемная в установке.)
# FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

# # Install Python and dependencies
# RUN apt-get update && apt-get install -y \
#     python3.8 python3-pip ffmpeg libsm6 libxext6 && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Install OpenCV with CUDA support
# RUN pip install opencv-python-headless opencv-contrib-python-headless

# WORKDIR /app

# COPY requirements.txt /app/
# RUN pip install --no-cache-dir -r requirements.txt

# COPY app /app

# RUN mkdir -p /app/uploads /app/processed

# EXPOSE 5000

# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app", "--workers", "4"]
