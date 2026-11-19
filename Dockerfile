FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY hybrid_fedbatch_twin ./hybrid_fedbatch_twin
COPY api ./api
COPY mcp_server ./mcp_server
COPY agents ./agents
COPY mlops ./mlops
COPY scripts ./scripts
COPY docs/assets/twin_bundle.json ./docs/assets/twin_bundle.json
COPY docs/validation ./docs/validation
COPY artifacts/uhlendorff_2026_validation.json ./artifacts/uhlendorff_2026_validation.json

RUN pip install --no-cache-dir -e ".[service]"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
