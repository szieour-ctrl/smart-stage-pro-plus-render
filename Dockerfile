FROM node:18-slim

# Install FFmpeg explicitly via apt — more reliable than Nixpacks package
# resolution, which was not installing FFmpeg correctly on this build.
# python3 + pip + libglib2.0-0 added for motionRenderer.py (opencv headless).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        python3 \
        python3-pip \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for motionRenderer.py
# --break-system-packages required on Debian Bookworm (PEP 668)
RUN pip3 install --no-cache-dir --break-system-packages \
    opencv-python-headless \
    numpy

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src ./src

EXPOSE 3000

CMD ["node", "src/server.js"]
