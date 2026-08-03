# Source: rust:1.97-bookworm
FROM rust@sha256:77fac8b98f9f46062bb680b6d25d5bcaabfc400143952ebc572e924bcbedc3fa

# Install AppImage build dependencies and prepare the unprivileged build user.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        appstream \
        binutils \
        ca-certificates \
        curl \
        desktop-file-utils \
        file \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --user-group appimage \
    && mkdir -p /work \
    && chown appimage:appimage /work

WORKDIR /work

# Configure user and Cargo paths.
ENV HOME=/home/appimage
ENV CARGO_TARGET_DIR=/work/target/appimage-container

# Run builds without root privileges.
USER appimage

CMD ["packaging/linux/appimage/build.sh", "/work/target/appimage/xl-view-x86_64.AppImage"]
