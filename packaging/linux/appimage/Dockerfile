# Source: rust:1.88.0-bookworm
FROM rust@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0

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
    && useradd --create-home --user-group appimage \
    && mkdir -p /work \
    && chown appimage:appimage /work

WORKDIR /work

# Configure user and Cargo paths.
ENV HOME=/home/appimage
ENV CARGO_TARGET_DIR=/work/target/appimage-container

# Run builds without root privileges.
USER appimage

CMD ["packaging/linux/appimage/build.sh", "/work/target/appimage/xl-view-x86_64.AppImage"]
