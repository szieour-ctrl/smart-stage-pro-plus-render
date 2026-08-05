FROM node:18-slim

# Install FFmpeg explicitly via apt — more reliable than Nixpacks package
# resolution, which was not installing FFmpeg correctly on this build.
# python3 + pip + libglib2.0-0 added for motionRenderer.py (opencv headless).
# libimage-exiftool-perl added (July 29, 2026) for hdrGainMapTest.py — the
# standalone HDR gain-map diagnostic tool. Read-only metadata/embedded-image
# extraction only, not used anywhere in the production render/correct paths.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        python3 \
        python3-pip \
        libglib2.0-0 \
        libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for motionRenderer.py
# --break-system-packages required on Debian Bookworm (PEP 668)
# scikit-image added (Aug 4, 2026) for smartCorrect.py's new surface-
# consistency pass (detect_surface_consistency_targets /
# apply_surface_consistency) -- Felzenszwalb segmentation to detect
# continuous surfaces (ceilings, walls) that region-based masking split
# into inconsistently-treated pieces. Pulls in scipy and its own
# dependencies automatically via pip.
RUN pip3 install --no-cache-dir --break-system-packages \
    opencv-python-headless \
    numpy \
    scikit-image

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src ./src

EXPOSE 3000

CMD ["node", "src/server.js"]
