ARG BASE_BUILDER_IMAGE=ffreis/base-builder
FROM ${BASE_BUILDER_IMAGE}

USER root

RUN mkdir -p /build \
    && chown appuser:appgroup /build \
    && chmod 0750 /build

WORKDIR /build

USER appuser:appgroup

COPY --chown=appuser:appgroup app/ .

USER appuser:appgroup

RUN cargo test --verbose --locked

ENTRYPOINT ["cargo", "build", "--locked"]
CMD ["--release"]
