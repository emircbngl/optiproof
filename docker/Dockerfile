# Rich toolchain image for OptiProof's Docker sandbox.
# Adds pytest (so a target's existing test suite runs in-container) and numpy
# (so `native` delegation candidates work in-container). Build once:
#
#   docker build -t optiproof-python:latest docker/
#
# Then:  optiproof optimize mod.py::fn --sandbox docker --toolchain-image optiproof-python:latest
FROM python:3.13-slim
RUN pip install --no-cache-dir pytest==8.* numpy==2.* hypothesis==6.*
