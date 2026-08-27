# vsr-frontend — toți biții statici: build-ul React + servirea video-urilor
# de review direct de pe volumul de date (nginx face Range nativ).
# Build (din rădăcina repo-ului):
#   docker build -f docker/frontend.Dockerfile -t vsr-frontend .

# ── etapa 1: build-ul React (Vite) ──────────────────────────────────────────
FROM node:22-alpine AS build
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── etapa 2: nginx servește dist + media + proxy /api ───────────────────────
FROM nginx:alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -q --spider http://127.0.0.1/ || exit 1
