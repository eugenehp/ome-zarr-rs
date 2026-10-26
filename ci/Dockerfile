# Cross-platform parity check: regenerate every Python fixture on this
# platform and run the Rust test suite against them.
#
#   docker build --platform linux/amd64 -t ome-zarr-rs-parity -f ci/Dockerfile ci
#   ci/run-parity.sh linux/amd64
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang curl ca-certificates git pkg-config libssl-dev zip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-parity.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-parity.txt
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /work
