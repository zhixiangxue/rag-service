#!/usr/bin/env bash
# infra.sh — Manage infrastructure services: qdrant, meilisearch, rqlite, redis
# Usage:
#   ./scripts/infra.sh --install            # Download binaries, install systemd services
#   ./scripts/infra.sh --start   [svc|all]  # Start service(s)
#   ./scripts/infra.sh --stop    [svc|all]  # Stop service(s)
#   ./scripts/infra.sh --restart [svc|all]  # Restart service(s)
#   ./scripts/infra.sh --status             # Show status of all services

set -e

# ============================================================
# Configuration — adjust these if your paths differ
# ============================================================
SERVICE_DIR="/etc/systemd/system"
CURRENT_USER="${SUDO_USER:-$USER}"
USER_HOME=$(eval echo "~$CURRENT_USER")
# All service files (binary + config + data) live under ZAG_DIR/<service>/
ZAG_DIR="$USER_HOME/.zag"
# Node address for rqlite advertised endpoints.
# Override via env: NODE_ADDR=1.2.3.4 sudo ./infra.sh --install
NODE_ADDR="${NODE_ADDR:-$(hostname -I | awk '{print $1}')}"

QDRANT_VERSION="v1.17.0"
MEILISEARCH_VERSION="v1.40.0"
RQLITE_VERSION="v9.4.5"

QDRANT_PORT_HTTP=16333
QDRANT_PORT_GRPC=16334
MEILISEARCH_PORT=7700
RQLITE_PORT_HTTP=4001
RQLITE_PORT_RAFT=4002
REDIS_PORT=6380

ALL_SERVICES=(qdrant meilisearch rqlite redis)

# ============================================================
# Colors
# ============================================================
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

# ============================================================
# Helpers
# ============================================================
info()    { echo -e "${GREEN}[infra]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || error "This command requires sudo. Run: sudo $0 $*"
}

resolve_services() {
    local target="${1:-all}"
    if [[ "$target" == "all" ]]; then
        echo "${ALL_SERVICES[@]}"
    elif [[ " ${ALL_SERVICES[*]} " == *" $target "* ]]; then
        echo "$target"
    else
        error "Unknown service: $target. Choose from: ${ALL_SERVICES[*]}"
    fi
}

# ============================================================
# Install helpers
# ============================================================
install_qdrant() {
    info "Installing Qdrant ${QDRANT_VERSION}..."
    local arch
    arch=$(uname -m)
    [[ "$arch" == "x86_64" ]] && arch="x86_64" || arch="aarch64"

    local url="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${arch}-unknown-linux-gnu.tar.gz"
    local dest="${ZAG_DIR}/qdrant"
    mkdir -p "$dest/data" "$dest/snapshots"

    wget -qO /tmp/qdrant.tar.gz "$url"
    tar -xzf /tmp/qdrant.tar.gz -C "$dest"
    chmod +x "$dest/qdrant"

    # Write config alongside binary
    cat > "${dest}/config.yaml" <<EOF
service:
  http_port: ${QDRANT_PORT_HTTP}
  grpc_port: ${QDRANT_PORT_GRPC}
  host: 0.0.0.0
storage:
  storage_path: ${dest}/data
  snapshots_path: ${dest}/snapshots
performance:
  max_search_threads: 0
  max_optimization_threads: 2
log_level: INFO
EOF

    # Write systemd service
    cat > "${SERVICE_DIR}/qdrant.service" <<EOF
[Unit]
Description=Qdrant vector database
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${dest}
ExecStart=${dest}/qdrant --config-path ${dest}/config.yaml
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable qdrant
    info "Qdrant installed (HTTP :${QDRANT_PORT_HTTP})"
}

install_meilisearch() {
    info "Installing Meilisearch ${MEILISEARCH_VERSION}..."
    local arch
    arch=$(uname -m)
    # meilisearch uses amd64/aarch64 naming (no x86_64)
    [[ "$arch" == "x86_64" ]] && arch="amd64" || arch="aarch64"

    local url="https://github.com/meilisearch/meilisearch/releases/download/${MEILISEARCH_VERSION}/meilisearch-linux-${arch}"
    local dest="${ZAG_DIR}/meilisearch"
    mkdir -p "$dest/data" "$dest/dumps" "$dest/snapshots"

    # Single binary, no archive to extract
    wget -qO "$dest/meilisearch" "$url"
    chmod +x "$dest/meilisearch"

    # Read MEILI_MASTER_KEY from .env if present
    local master_key=""
    local env_file
    env_file="$(dirname "$0")/../.env"
    if [[ -f "$env_file" ]]; then
        master_key=$(grep -E '^MEILISEARCH_API_KEY=' "$env_file" | cut -d= -f2- | tr -d '"' | tr -d "'")
    fi

    cat > "${SERVICE_DIR}/meilisearch.service" <<EOF
[Unit]
Description=Meilisearch search engine
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${dest}
Environment="MEILI_DB_PATH=${dest}/data"
Environment="MEILI_HTTP_ADDR=0.0.0.0:${MEILISEARCH_PORT}"
Environment="MEILI_DUMP_DIR=${dest}/dumps"
Environment="MEILI_SNAPSHOT_DIR=${dest}/snapshots"
$([ -n "$master_key" ] && echo "Environment=\"MEILI_MASTER_KEY=${master_key}\"")
ExecStart=${dest}/meilisearch
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable meilisearch
    info "Meilisearch installed (:${MEILISEARCH_PORT})"
}

install_rqlite() {
    info "Installing rqlite ${RQLITE_VERSION}..."
    local arch
    arch=$(uname -m)
    [[ "$arch" == "x86_64" ]] && arch="amd64" || arch="arm64"

    # rqlite version without 'v' prefix in filename: rqlite-v9.4.5-linux-amd64.tar.gz
    local url="https://github.com/rqlite/rqlite/releases/download/${RQLITE_VERSION}/rqlite-${RQLITE_VERSION}-linux-${arch}.tar.gz"
    local dest="${ZAG_DIR}/rqlite"
    local ver_strip="${RQLITE_VERSION}"  # keep v prefix, matches folder name
    mkdir -p "$dest/data"

    wget -qO /tmp/rqlite.tar.gz "$url"
    tar -xzf /tmp/rqlite.tar.gz -C /tmp
    # Extracted folder: rqlite-v9.4.5-linux-amd64/
    mv /tmp/rqlite-${ver_strip}-linux-${arch}/rqlited "$dest/rqlited"
    chmod +x "$dest/rqlited"

    # Startup args match current Windows usage:
    # rqlited.exe -node-id 1 -http-addr localhost:4001 -raft-addr localhost:4002 ./node_data
    cat > "${SERVICE_DIR}/rqlite.service" <<EOF
[Unit]
Description=rqlite distributed SQLite database
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
ExecStart=${dest}/rqlited -node-id 1 -http-addr 0.0.0.0:${RQLITE_PORT_HTTP} -http-adv-addr ${NODE_ADDR}:${RQLITE_PORT_HTTP} -raft-addr 0.0.0.0:${RQLITE_PORT_RAFT} -raft-adv-addr ${NODE_ADDR}:${RQLITE_PORT_RAFT} ${dest}/data
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable rqlite
    info "rqlite installed (:${RQLITE_PORT_HTTP})"
}

install_redis() {
    info "Installing Redis via apt..."
    # Wait for any background apt process to release the lock (e.g. unattended-upgrades)
    local wait_sec=0
    while fuser /var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
        if [ $wait_sec -eq 0 ]; then
            info "Waiting for apt lock to be released..."
        fi
        sleep 5
        wait_sec=$((wait_sec + 5))
        if [ $wait_sec -ge 120 ]; then
            error "apt lock not released after 120s, aborting"
            exit 1
        fi
    done
    info "Updating apt package index..."
    apt-get update
    apt-get install -y redis-server

    # Configure port and bind address
    local conf="/etc/redis/redis.conf"
    sed -i "s/^port .*/port ${REDIS_PORT}/" "$conf"
    sed -i "s/^bind .*/bind 0.0.0.0/" "$conf"
    sed -i "s/^protected-mode yes/protected-mode no/" "$conf"

    # Ensure it's enabled and running with the updated config
    systemctl enable redis-server
    systemctl restart redis-server
    info "Redis installed (:${REDIS_PORT})"
}

# ============================================================
# Commands
# ============================================================
cmd_install() {
    require_root

    # Abort if any service directory already exists to prevent accidental data loss
    local existing=()
    for svc in qdrant meilisearch rqlite; do
        [[ -d "${ZAG_DIR}/$svc" ]] && existing+=("${ZAG_DIR}/$svc")
    done
    if [[ ${#existing[@]} -gt 0 ]]; then
        warn "⚠️  Existing service data detected:"
        for d in "${existing[@]}"; do
            warn "    $d"
        done
        warn "⚠️  Aborting to prevent data loss."
        warn "    If you intend to reinstall, remove the directories manually first:"
        warn "    sudo rm -rf ~/.zag"
        exit 1
    fi

    # Stop any running services first (so binaries are not busy)
    for svc in qdrant meilisearch rqlite redis-server; do
        systemctl stop "$svc" 2>/dev/null || true
    done

    # Each service gets its own dir under ZAG_DIR
    mkdir -p "${ZAG_DIR}/qdrant" "${ZAG_DIR}/meilisearch" "${ZAG_DIR}/rqlite"

    install_qdrant
    install_meilisearch
    install_rqlite
    install_redis

    # Fix ownership: dirs created under sudo, give them back to the real user
    chown -R "$CURRENT_USER:$CURRENT_USER" "$ZAG_DIR"

    # Start services after ownership is correct
    systemctl start qdrant meilisearch rqlite redis-server

    echo ""
    info "All infrastructure services installed."
    cmd_status
}

cmd_start() {
    local services
    read -ra services <<< "$(resolve_services "${1:-all}")"
    for svc in "${services[@]}"; do
        local name="$svc"
        [[ "$svc" == "redis" ]] && name="redis-server"
        info "Starting $svc..."
        sudo systemctl start "$name"
    done
}

cmd_stop() {
    local services
    read -ra services <<< "$(resolve_services "${1:-all}")"
    for svc in "${services[@]}"; do
        local name="$svc"
        [[ "$svc" == "redis" ]] && name="redis-server"
        info "Stopping $svc..."
        sudo systemctl stop "$name"
    done
}

cmd_restart() {
    local services
    read -ra services <<< "$(resolve_services "${1:-all}")"
    for svc in "${services[@]}"; do
        local name="$svc"
        [[ "$svc" == "redis" ]] && name="redis-server"
        info "Restarting $svc..."
        sudo systemctl restart "$name"
    done
}

cmd_status() {
    echo ""
    for svc in "${ALL_SERVICES[@]}"; do
        local name="$svc"
        [[ "$svc" == "redis" ]] && name="redis-server"
        local status
        status=$(systemctl is-active "$name" 2>/dev/null) || true
        [[ -z "$status" ]] && status="not-found"
        if [[ "$status" == "active" ]]; then
            echo -e "  ${GREEN}●${NC} $svc — active"
        else
            echo -e "  ${RED}●${NC} $svc — $status"
        fi
    done
    echo ""
}

usage() {
    echo "Usage: $0 [--install | --start | --stop | --restart | --status] [service]"
    echo ""
    echo "Commands:"
    echo "  --install            Download binaries and install systemd services (requires sudo)"
    echo "  --start   [svc|all]  Start service(s)"
    echo "  --stop    [svc|all]  Stop service(s)"
    echo "  --restart [svc|all]  Restart service(s)"
    echo "  --status             Show status of all services"
    echo ""
    echo "Services: ${ALL_SERVICES[*]}"
}

# ============================================================
# Entry point
# ============================================================
ACTION=""
TARGET="all"

for arg in "$@"; do
    case $arg in
        --install|--start|--stop|--restart|--status) ACTION="$arg" ;;
        qdrant|meilisearch|rqlite|redis|all)          TARGET="$arg" ;;
        *) echo "Unknown argument: $arg"; usage; exit 1 ;;
    esac
done

[[ -z "$ACTION" ]] && { usage; exit 1; }

case $ACTION in
    --install) cmd_install ;;
    --start)   cmd_start   "$TARGET" ;;
    --stop)    cmd_stop    "$TARGET" ;;
    --restart) cmd_restart "$TARGET" ;;
    --status)  cmd_status ;;
esac
