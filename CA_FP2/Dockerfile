FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN printf 'Acquire::ForceIPv4 "true";\nAcquire::https::Verify-Peer "false";\nAcquire::https::Verify-Host "false";\n' > /etc/apt/apt.conf.d/99bootstrap && \
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    make \
    wget \
    tar \
    python3 \
    python3-numpy \
    ca-certificates \
    bash && \
    rm -f /etc/apt/apt.conf.d/99bootstrap && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN wget --no-check-certificate https://github.com/accellera-official/systemc/archive/refs/tags/2.3.4.tar.gz && \
    tar -xzf 2.3.4.tar.gz && \
    cd systemc-2.3.4 && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/systemc \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_CXX_EXTENSIONS=OFF && \
    make -j"$(nproc)" && \
    make install

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]