# Base image: an official, minimal Linux distro with Python 3.12 already installed.
# "slim" = smaller than the full python image (no extra build tools you don't need).
FROM python:3.12-slim

# Sets the working directory INSIDE the container. Every command below runs
# relative to this path — like `cd /app` for everything that follows.
WORKDIR /app

ENV RENDER=true

# Copy ONLY the requirements file first, before the rest of your code.
# Why separately: Docker caches each step. If your code changes but
# requirements.txt doesn't, Docker reuses the cached "pip install" result
# instead of reinstalling everything from scratch every single build.
COPY requirements-docker.txt .

# --no-cache-dir: don't keep pip's download cache inside the image — saves space,
# since you'll never run pip again inside this exact container.
RUN pip install --no-cache-dir -r requirements-docker.txt

# Now copy your actual application code.
# NOTE: .env is intentionally NEVER copied here — see .dockerignore below.
# Secrets get passed in at runtime instead (via `docker run -e` or your
# hosting platform's environment variable settings), never baked into the image.
COPY src/ ./src/
COPY static/ ./static/

# Documents which port this container listens on. This line is informational
# for humans/tools reading the Dockerfile — it does NOT actually publish the
# port by itself (that happens with `-p` when you run the container).
EXPOSE 8001

# The actual startup command — runs when the container starts.
# Uses 0.0.0.0 (not 127.0.0.1) so the server accepts connections from
# outside the container, not just from inside it.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]