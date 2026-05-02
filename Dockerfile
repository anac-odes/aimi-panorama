#FROM --platform=linux/amd64 pytorch/pytorch
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS="ignore "

# CHANGE: Added
ENV DEBIAN_FRONTEND=noninteractive 

# CHANGE: Switch Python version from 3.9 to 3.12
RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  wget \
  unzip \
  libopenblas-dev \
  python3.12 \
  python3.12-dev \
  python3-pip \
  nano \
  && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/* 

# CHANGE: Added:
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python python /usr/bin/python3.12 1

# CHANGE: Added:
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

RUN groupadd -r user && useradd -m --no-log-init -r -g user user


RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"

USER user

# CHANGE: Added requirements.text to the path below
COPY --chown=user:user requirements.txt /opt/app/requirements.txt

# You can add any Python dependencies to requirements.txt
RUN python3.12 -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt


# IN BASELINE: clone nnUNet
# ### Clone nnUNet
# # Configure Git, clone the repository without checking out, then checkout the specific commit
# RUN git config --global advice.detachedHead false && \
#     git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ 

# # Install a few dependencies that are not automatically installed
# RUN pip3 install \
#         -e /opt/algorithm/nnunet \
#         graphviz \
#         onnx \
#         SimpleITK && \
#     rm -rf ~/.cache/pip

# COPY --chown=user:user ./src/customTrainerCEcheckpoints.py /opt/algorithm/nnunet/nnunetv2/training/nnUNetTrainer/customTrainerCEcheckpoints.py
# COPY --chown=user:user ./src/nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

# CHANGE: Do not clone nnUNet but copy from the pre-downloaded nnunetv2 directory
# Copy local nnunetv2 package and install it (no internet needed)
COPY --chown=user:user ./packages/nnunetv2 /opt/algorithm/nnunetv2/
RUN pip install \
    --user \
    --no-cache-dir \
    -e /opt/algorithm/nnunetv2 && \
    rm -rf ~/.cache/pip

# Copy local report-guided-annotation package and install it
COPY --chown=user:user ./packages/report-guided-annotation /opt/algorithm/report-guided-annotation/
RUN pip install \
    --user \
    --no-cache-dir \
    -e /opt/algorithm/report-guided-annotation && \
    rm -rf ~/.cache/pip

# Copy model weights into the container
# These are baked in at build time — no internet access at runtime
COPY --chown=user:user ./workspace/nnUNet_results /opt/algorithm/nnUNet_results/


### Define workdir
WORKDIR /opt/app

# COPY --chown=user:user ./src/process.py /opt/app/
# COPY --chown=user:user ./src/data_utils.py /opt/app/
# COPY --chown=user:user ./src/__init__.py /opt/app/

# CHANGE: Change the directories
COPY --chown=user:user ./main.py /opt/app/main.py

### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["python3.12", "/opt/app/main.py", "-i", "/input", "-o", "/output"]