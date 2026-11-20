# Hardened Production Dockerfile for Saigak-Orto 3D Craniometry Workstation
# Built following docker-expert & security-scanning-security-hardening guidelines
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8765

WORKDIR /app

# Create non-privileged clinical service user
RUN groupadd -g 10001 saigak && \
    useradd -u 10001 -g saigak -m -s /usr/sbin/nologin appuser

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/
RUN chown -R appuser:saigak /app

USER appuser

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health')" || exit 1

CMD ["python", "server.py"]
