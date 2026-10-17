FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Commit of smanolloff/vcmi (branch "mmai") that this Dockerfile is verified against.
# The "mmai" branch moves fast and can break or change at any time, so we pin to a
# known-good commit instead of always pulling the branch head. See README.md for
# how/when to bump this on purpose.
ARG VCMI_COMMIT=75e343f607f8e9b6f6865c83203062425d23a1fb

# Same reasoning for vcmi-gym itself (also pins which exact commit of its own
# "vcmi" submodule gets used, since that's recorded in vcmi-gym's own git history).
ARG VCMIGYM_COMMIT=9cdfe841ba9aa6f72a9508ec0c2eafff97efd99b

RUN apt-get update && apt-get install -y \
    cmake g++ clang \
    libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev \
    zlib1g-dev \
    libavformat-dev libswscale-dev \
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-date-time-dev libboost-iostreams-dev \
    qt6-base-dev qt6-base-dev-tools qt6-tools-dev qt6-tools-dev-tools qt6-l10n-tools libqt6svg6-dev \
    libtbb-dev libluajit-5.1-dev liblzma-dev libsqlite3-dev libminizip-dev libsquish-dev libfmt-dev \
    ninja-build ccache gettext \
    git curl libcurl4-openssl-dev \
    python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz \
    -o /tmp/cmake.tar.gz && \
    tar -xzf /tmp/cmake.tar.gz -C /usr/local --strip-components=1 && \
    rm /tmp/cmake.tar.gz

RUN mkdir -p /opt/onnxruntime && \
    curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz \
    | tar -xz --strip-components=1 -C /opt/onnxruntime && \
    ldconfig /opt/onnxruntime

WORKDIR /vcmi

# The "mmai" branch gets force-pushed/rebased regularly, so a plain branch clone
# can lose a previously-pinned commit entirely. Fetching the exact SHA directly
# works even once it's no longer reachable from any branch tip.
RUN git init src && cd src && \
    git remote add origin https://github.com/smanolloff/vcmi.git && \
    git fetch origin ${VCMI_COMMIT} && \
    git checkout FETCH_HEAD && \
    git submodule update --init --recursive

# Upstream bug: this block references gameHandler->mlplugin without guarding it with
# ENABLE_ML, unlike every other usage in the file, so it fails to compile with the
# (default) ENABLE_ML=OFF. Wrap it the same way the rest of the file does.
RUN sed -i \
    -e '/const bool regenerateLayout = IS_ML/i #ifdef ENABLE_ML' \
    -e '/tightFormationChance > 0);/a #else\n\tconst bool regenerateLayout = false;\n#endif' \
    src/server/battles/BattleProcessor.cpp

RUN mkdir build && cd build && \
    cmake -S ../src -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime -DENABLE_MMAI=OFF -DENABLE_DISCORD=OFF && \
    cmake --build . -j$(nproc)

RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

# Same fetch-by-SHA approach as above, for the same reason.
RUN git init /vcmi-gym && cd /vcmi-gym && \
    git remote add origin https://github.com/smanolloff/vcmi-gym.git && \
    git fetch origin ${VCMIGYM_COMMIT} && \
    git checkout FETCH_HEAD && \
    git submodule update --init --recursive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cd /vcmi-gym && \
    python3.12 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements/requirements-torch-2.12.1+cpu.txt

# Upstream bug: BattleHexArray's iterator (boost::container::vec_iterator) isn't
# recognized as a C++20 contiguous_iterator, so std::span can't be constructed
# from it directly (neither the range ctor nor the iterator-pair ctor accept it).
# Materialize a plain std::vector via toVector() first; its iterator is a real
# contiguous_iterator.
RUN sed -i 's|target = PickClosestLandHex(guard, distances, NearbyMoveHexes(guard, target), skips);|auto nearbyHexesFix = NearbyMoveHexes(guard, target).toVector();\n\t\ttarget = PickClosestLandHex(guard, distances, std::span<const BattleHex>(nearbyHexesFix.begin(), nearbyHexesFix.end()), skips);|' \
    /vcmi-gym/vcmi/AI/MMAI/BAI/scripted/VIPBot.cpp

RUN mkdir /vcmi-gym/vcmi/rel && cd /vcmi-gym/vcmi/rel && \
    cmake -S .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime -DENABLE_ML=ON -DENABLE_DISCORD=OFF && \
    cmake --build . -j$(nproc) --target vcmiserver vcmiclient mlclient

RUN cd /vcmi-gym/vcmi_gym/connectors && \
    /vcmi-gym/.venv/bin/conan profile detect --force && \
    /vcmi-gym/.venv/bin/conan install . --output-folder=conan-generated --build=missing -s build_type=Release && \
    VIRTUAL_ENV=/vcmi-gym/.venv PATH=/vcmi-gym/.venv/bin:$PATH /vcmi-gym/.venv/bin/cmake --preset vcmigym-rel && \
    /vcmi-gym/.venv/bin/cmake --build rel -- -j$(nproc)

WORKDIR /vcmi
