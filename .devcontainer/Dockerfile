# CUDA 12.6 + cuDNN on Ubuntu 22.04 — supports RTX 500 Ada (sm_89)
# torch R 0.17.0 requires CUDA 12.6, 12.8, or 12.9 (not 12.1).
# Steps are split so each layer caches independently and parallel builds are reliable.
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Step 1: minimal tools to set up the CRAN apt repo ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Step 2: add the CRAN repo (no heavy downloads) ---
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
    | gpg --dearmor -o /usr/share/keyrings/r-project.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] \
https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    > /etc/apt/sources.list.d/r-project.list

# --- Step 3: install R ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Step 4: system deps for Bioconductor / ggplot / torch ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    locales \
    build-essential \
    pkg-config \
    cmake \
    libgomp1 \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libglpk-dev \
    libmagick++-dev \
    && rm -rf /var/lib/apt/lists/*

# Locale
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# --- Step 5: renv ---
RUN R -q -e "install.packages('renv', repos='https://cloud.r-project.org')"

# --- Step 6: torch R package (compiles C++ bindings, ~6 min) ---
RUN R -q -e "install.packages('torch', repos='https://cloud.r-project.org')"

# --- Step 7: download LibTorch with CUDA 12.1 (~500 MB) ---
# CUDA libs in this base image ensure torch detects CUDA and downloads the GPU build.
# For CPU-only: change type = 'cu121' to type = 'cpu'
RUN R -q -e "options(timeout = 600); torch::install_torch(type = 'cu126')"

# --- Step 8: Quarto ---
ENV QUARTO_VERSION=1.5.57
RUN wget -q "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && dpkg -i "quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && rm -f "quarto-${QUARTO_VERSION}-linux-amd64.deb"

ENV RENV_PATHS_CACHE=/opt/renv/cache

WORKDIR /workspaces/project
