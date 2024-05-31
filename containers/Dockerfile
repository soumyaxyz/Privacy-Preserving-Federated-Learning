ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.02-py3
FROM ${PYTORCH_IMAGE}

ARG NVF_VERSION=2.4
ENV NVF_BRANCH=${NVF_VERSION}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install nvflare==2.4.0

WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare


# # Copy the startup script into the image
# COPY startup.sh /startup.sh

# # Make sure the script is executable
# RUN chmod +x /startup.sh

# # Execute the startup script when the container starts
# CMD ["/startup.sh"]