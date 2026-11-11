# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM rust:1-slim-bookworm AS build

WORKDIR /app

# Strip symbols in this image only; local release builds keep them for profiling.
ENV CARGO_PROFILE_RELEASE_STRIP=symbols \
    CARGO_TERM_COLOR=never

COPY . .

# --locked: fail rather than silently re-resolve if Cargo.lock is out of date.
# The cache mounts keep the registry and the target directory across builds,
# so only changed crates are recompiled; the binary is copied out because a
# cache mount is not part of the image layer.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --locked --bin nexus_graph \
    && install -D -m 0555 target/release/nexus_graph /out/nexus_graph \
    && install -d -m 0750 -o 10001 -g 10001 /out/data

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
# Distroless: glibc and CA certificates, no shell, no package manager.
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

LABEL org.opencontainers.image.title="nexus_graph" \
      org.opencontainers.image.description="Hybrid vector-knowledge graph retrieval engine with deterministic claim verification"

COPY --from=build /out/nexus_graph /usr/local/bin/nexus_graph
# The only writable location. Pre-created with the runtime UID so that a named
# volume mounted here inherits the right ownership.
COPY --from=build --chown=10001:10001 /out/data /data

USER 10001:10001
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080

# SIGTERM triggers a graceful shutdown: stop accepting, finish in-flight requests.
STOPSIGNAL SIGTERM

# The image has no shell or curl; the binary probes itself.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/nexus_graph", "healthcheck", "127.0.0.1:8080"]

ENTRYPOINT ["/usr/local/bin/nexus_graph"]
CMD ["serve", "--bind", "0.0.0.0:8080", "--snapshot-dir", "/data"]
