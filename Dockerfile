# The local page (`fountains serve`) in a container, for macOS, Windows or Linux.
#
#   docker build -t fountains .
#   docker run --rm -p 127.0.0.1:8765:8765 -v fountains-cache:/cache fountains
#
# then open http://localhost:8765. The volume keeps the downloaded models and
# data, your rides and your decisions between runs.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FOUNTAINS_CACHE_DIR=/cache HF_HOME=/cache/hf TABPFN_MODEL_CACHE_DIR=/cache/tabpfn \
    TABPFN_ALLOW_CPU_LARGE_DATASET=1 HF_HUB_DISABLE_TELEMETRY=1

# PyTorch for the processor only: TabPFN and OWLv2 need no GPU here.
RUN pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY licenses ./licenses
COPY data ./data
COPY src ./src
# Editable install: the data/ folder is read from the checkout.
RUN pip install -e ".[photos]" \
 && useradd --create-home --uid 1000 rider \
 && mkdir -p /cache && chown rider /cache
USER rider
VOLUME /cache
EXPOSE 8765
CMD ["fountains", "serve", "--host", "0.0.0.0", "--port", "8765"]
