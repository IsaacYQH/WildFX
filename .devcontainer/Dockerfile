# Base with Python and audio dependencies
FROM ubuntu:22.04

# Let the installation process of jackd automatically select yes, preventing the need for user input
RUN echo "jackd2 jackd/tweak_rt_limits boolean true" | debconf-set-selections

# Enable 32-bit architecture and install core dependencies
RUN dpkg --add-architecture i386 && \
    DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    xvfb \
    vim \
    git \
    tmux \
    sudo \
    # Core audio libraries
    libasound2 \
    libasound2-plugins \
    alsa-base \
    alsa-utils \
    # JACK audio support
    jackd2 \
    # GTK3 dependencies
    libgtk-3-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    # GLib dependencies
    libglib2.0-0 \
    # Pango/font rendering
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    # Other libraries requried for VSTs
    libcurl4 \
    libfreetype6 \
    # Fontconfig
    libfontconfig1 \
    # Dependencies for Wine
    gpg-agent \
    # Calf plugins (Optional)
    calf-plugins \
    && rm -rf /var/lib/apt/lists/*

# --- Add WineHQ Repository and Install Wine-Staging ---
RUN mkdir -pm755 /etc/apt/keyrings && \
    wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key && \
    wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --install-recommends winehq-staging && \
    rm -rf /var/lib/apt/lists/*
# --- End WineHQ Installation ---

# Create the 'u1' user with sudo access
# Note that if using vscode, then the UID/GID of the user must match the one used in the host system
# Or, for all commands related to permission change of the user inside of the container, put them into postCreateCommand or entrypoint.sh
ARG USERNAME=u1
ARG USER_UID=1026
ARG USER_GID=1027
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
# RUN useradd -m -s /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Create audio group with the GID passed as build argument
ARG AUDIO_GID=29
RUN groupadd -f -g $AUDIO_GID audio \
    && usermod -aG audio $USERNAME \
    && usermod -aG audio root

# Install Miniconda in u1's home directory
ARG USER_HOME="/home/u1"
ENV CONDA_DIR="${USER_HOME}/miniconda3"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_25.1.1-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p ${CONDA_DIR} && \
    /bin/bash /tmp/miniconda.sh -b -u -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    chown -R ${USERNAME}:${USER_GID} ${CONDA_DIR}

# Add Conda to PATH
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Configure Conda environment
RUN conda init bash && \
    conda config --set auto_activate_base true &&\
    ${CONDA_DIR}/bin/conda run -n base pip install python-reapy

# Install yabridge following official recommendations
RUN mkdir -p ${USER_HOME}/.local/share/yabridge && \
    # Download latest yabridge release
    wget -O /tmp/yabridge.tar.gz https://github.com/robbert-vdh/yabridge/releases/download/5.1.1/yabridge-5.1.1.tar.gz && \
    # Extract to /home/u1/.local/share/yabridge
    tar -C ${USER_HOME}/.local/share -xzf /tmp/yabridge.tar.gz && \
    # Add yabridge directory to PATH for u1 user
    echo 'export PATH="$PATH:$HOME/.local/share/yabridge"' >> ${USER_HOME}/.bashrc && \
    # Ensure yabridgectl is executable
    chmod +x ${USER_HOME}/.local/share/yabridge/yabridgectl && \
    # Set ownership to u1 user
    chown -R ${USERNAME}:${USER_GID} ${USER_HOME}/.local && \
    # Cleanup
    rm /tmp/yabridge.tar.gz

# Symlink to make yabridgectl available system-wide
RUN ln -s ${USER_HOME}/.local/share/yabridge/yabridgectl /usr/local/bin/yabridgectl

# Enable realtime scheduling for jack server required by REAPER
RUN echo "@audio - rtprio 99" >> /etc/security/limits.d/audio.conf && \
    echo "@audio - memlock unlimited" >> /etc/security/limits.d/audio.conf && \
    usermod -aG audio ${USERNAME}

# Create VST3 directories
ENV VST3_PATH="${USER_HOME}/.vst3"
RUN mkdir -p ${VST3_PATH} && chown -R ${USERNAME}:${USER_GID} ${VST3_PATH}

# Create VST2 directories
ENV VST2_PATH="${USER_HOME}/.vst"
RUN mkdir -p ${VST2_PATH} && chown -R ${USERNAME}:${USER_GID} ${VST2_PATH}

# Create CLAP directories
ENV CLAP_PATH="${USER_HOME}/.clap"
RUN mkdir -p ${CLAP_PATH} && chown -R ${USERNAME}:${USER_GID} ${CLAP_PATH}

# Create LV2 directories
ENV LV2_PATH="${USER_HOME}/.lv2"
RUN mkdir -p ${LV2_PATH} && chown -R ${USERNAME}:${USER_GID} ${LV2_PATH}
# Set working directory
WORKDIR "${USER_HOME}"

# Set up conda initialization for u1 user (already done above, but keeping for clarity)
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> ${USER_HOME}/.bashrc && \
    echo "conda activate base" >> ${USER_HOME}/.bashrc && \
    chown ${USERNAME}:${USER_GID} ${USER_HOME}/.bashrc

# Entrypoint script for REAPER
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    sed -i 's/\r$//' /entrypoint.sh

# Switch to u1 user
USER ${USERNAME}

# Download and extract REAPER (portable mode)
RUN mkdir -p ${USER_HOME}/reaper
    # chown -R ${USERNAME}:${USER_GID} /opt/reaper-portable
WORKDIR ${USER_HOME}/reaper
RUN wget -O reaper.tar.xz https://www.reaper.fm/files/7.x/reaper735_linux_x86_64.tar.xz \
    && tar -xf reaper.tar.xz --strip-components=1 \
    && rm reaper.tar.xz
    # && chown -R ${USERNAME}:${USER_GID} /opt/reaper-portable

# Configure portable REAPER
ENV PATH="${USER_HOME}/reaper/REAPER:${PATH}"
ENV JACK_NO_AUDIO_RESERVATION="1"

ENTRYPOINT ["/entrypoint.sh"]