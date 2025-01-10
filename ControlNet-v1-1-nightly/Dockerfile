ARG PYTORCH="2.5.1"
ARG CUDA="12.1"
ARG CUDNN="9"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
# RUN pip install --no-cache-dir --ignore-installed -r /requirements.txt -U

CMD [ "bash" ]