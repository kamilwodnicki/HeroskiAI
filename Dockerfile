FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake g++ clang \
    libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev \
    zlib1g-dev \
    libavformat-dev libswscale-dev \
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-date-time-dev libboost-iostreams-dev \
    qt6-base-dev qt6-base-dev-tools qt6-tools-dev qt6-tools-dev-tools qt6-l10n-tools libqt6svg6-dev \
    libtbb-dev libluajit-5.1-dev liblzma-dev libsqlite3-dev libminizip-dev libsquish-dev libfmt-dev \
    ninja-build ccache \
    git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /vcmi
