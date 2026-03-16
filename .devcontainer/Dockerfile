FROM ubuntu:22.04

# Install build essentials, Python 3.10 toolchain, sudo, and deps for symforce/ceres build
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        build-essential cmake ninja-build libgmp-dev \
        libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev libeigen3-dev \
        libboost-all-dev libtbb-dev libabsl-dev \
        git curl ca-certificates sudo && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Provide SCM version hints (build context lacks .git metadata)
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SYMFORCE=0.0.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SKYMARSHAL=0.0.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Base workspace inside image
WORKDIR /opt/slambench

# Copy symforce submodule and build/install
WORKDIR /opt/slambench/symforce
COPY symforce/ /opt/slambench/symforce/
RUN python3 -m pip install --no-cache-dir -e .[dev] && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" bundle_adjustment_in_the_large_example

# Build ceres-solver with examples (bundle_adjuster)
WORKDIR /opt/slambench/ceres-solver
COPY ceres-solver/ /opt/slambench/ceres-solver/
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DBUILD_TESTING=OFF && \
    make -j"$(nproc)"

# Build GTSAM with examples
WORKDIR /opt/slambench/gtsam
COPY gtsam/ /opt/slambench/gtsam/
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGTSAM_BUILD_EXAMPLES=ON \
             -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_UNSTABLE=OFF \
             -DGTSAM_BUILD_UNSTABLE_EXAMPLES=OFF \
             -DGTSAM_WITH_TBB=ON && \
    make -j"$(nproc)" SFMExample_bal SFMExample_SmartFactor_bal

# Build MSCKF C solver
WORKDIR /opt/slambench/msckf_c
COPY msckf_c/ /opt/slambench/msckf_c/
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" msckf_runner

# Copy scripts/data into the image workspace
WORKDIR /workspace
COPY run_symforce.sh run_ceres.sh run_gtsam.sh run_msckf.sh run_benchmarks.sh README.md ./ 
COPY data/ ./data/
RUN chmod +x /workspace/run_*.sh

# Install OpenBLAS late to preserve cache on heavy build layers.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for devcontainer sessions
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    chown -R $USERNAME:$USER_GID /workspace

# Switch to non-root user by default
USER $USERNAME
ENV HOME=/home/$USERNAME

# Back to workspace
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/workspace/run_benchmarks.sh"]
