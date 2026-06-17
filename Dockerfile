FROM node:18-slim

# Install FFmpeg explicitly via apt — more reliable than Nixpacks package
# resolution, which was not installing FFmpeg correctly on this build.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src ./src

EXPOSE 3000

CMD ["node", "src/server.js"]
