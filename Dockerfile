FROM python:3.11-slim

ARG WHISPER_MODEL=small
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE=/opt/whisper-models \
    WHISPER_MODEL=${WHISPER_MODEL}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir . \
    && python -c "from faster_whisper.utils import download_model; download_model('${WHISPER_MODEL}', cache_dir='/opt/whisper-models')"

RUN useradd --create-home --uid 10001 runner \
    && chown -R runner:runner /app /opt/whisper-models
USER runner

ENTRYPOINT ["chemicheck119-speech-eval"]
