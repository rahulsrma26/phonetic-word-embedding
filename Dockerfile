FROM continuumio/miniconda3
# frolvlad/alpine-miniconda3

RUN apt-get update
RUN apt-get install build-essential -y

RUN conda --version
ENV CONDA_BIN='/opt/conda'

COPY environment.yml /tmp/conda-tmp/
COPY src/requirements.txt /tmp/conda-tmp/req1.txt
COPY embedding_english/requirements.txt /tmp/conda-tmp/req2.txt
COPY embedding_hindi/requirements.txt /tmp/conda-tmp/req3.txt

RUN conda env update -n base -f /tmp/conda-tmp/environment.yml
RUN conda init
