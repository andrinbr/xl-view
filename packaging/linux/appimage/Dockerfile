# Source: rust:1.97-bookworm
FROM rust@sha256:77fac8b98f9f46062bb680b6d25d5bcaabfc400143952ebc572e924bcbedc3fa

ARG CARGO_PACKAGER_VERSION=0.11.8

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
    && cargo install cargo-packager --version "${CARGO_PACKAGER_VERSION}" --locked \
    && rm -rf /usr/local/cargo/registry \
    && useradd --create-home --uid 1000 --user-group appimage \
    && mkdir -p /work \
    && chown appimage:appimage /work

COPY --chmod=0755 build-appimage.sh /usr/local/bin/build-appimage

WORKDIR /work

# Configure user and Cargo paths.
ENV HOME=/home/appimage
ENV CARGO_HOME=/home/appimage/.cargo
ENV CARGO_TARGET_DIR=/work/target/appimage-container

# Run builds without root privileges.
USER appimage

ENTRYPOINT ["/usr/local/bin/build-appimage"]
