#!/usr/bin/env bash
# infra.sh — Manage infrastructure services: qdrant, meilisearch, rqlite, redis, falkordb
#
# Local usage:
#   sudo ./scripts/infra.sh --install        # Download binaries, install systemd services
#   ./scripts/infra.sh --start   [svc|all]   # Start service(s)
#   ./scripts/infra.sh --stop    [svc|all]   # Stop service(s)
#   ./scripts/infra.sh --restart [svc|all]   # Restart service(s)
#   ./scripts/infra.sh --status              # Show status of all services
#
# Remote one-liner (repo: https://github.com/zhixiangxue/rag-service):
#   curl -fsSL https://raw.githubusercontent.com/zhixiangxue/rag-service/main/scripts/infra.sh | sudo bash -s -- --install
#
#   The script prompts for a shared password interactively (reads /dev/tty,
#   so it works even when piped). For non-interactive installs, pass the
#   password via environment variable instead:
#   curl -fsSL https://raw.githubusercontent.com/zhixiangxue/rag-service/main/scripts/infra.sh | sudo INFRA_PASSWORD=xxx bash -s -- --install
#
#   If GitHub releases are slow/unreachable (e.g. servers in mainland China),
#   prepend a proxy via GITHUB_MIRROR:
#   curl -fsSL https://raw.githubusercontent.com/zhixiangxue/rag-service/main/scripts/infra.sh | sudo GITHUB_MIRROR=https://ghfast.top/ INFRA_PASSWORD=xxx bash -s -- --install

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
FALKORDB_VERSION="v4.18.11"
# FalkorDB requires Redis 8+; built from source if system redis is older
REDIS_SRC_VERSION="8.8.0"

QDRANT_PORT_HTTP=16333
QDRANT_PORT_GRPC=16334
MEILISEARCH_PORT=7700
RQLITE_PORT_HTTP=4001
RQLITE_PORT_RAFT=4002
REDIS_PORT=6380
FALKORDB_PORT=6379

ALL_SERVICES=(qdrant meilisearch rqlite redis falkordb)

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

# Shared password for all infra services (qdrant api_key, meilisearch
# master key, rqlite basic auth, redis requirepass).
# Empty = no auth. Set via env (INFRA_PASSWORD=xxx sudo ./infra.sh --install)
# or entered interactively during --install.
INFRA_PASSWORD="${INFRA_PASSWORD:-}"

prompt_password() {
    # Already provided via env — non-interactive mode
    if [[ -n "$INFRA_PASSWORD" ]]; then
        info "Using INFRA_PASSWORD from environment"
        return
    fi
    # Read from the terminal directly so it works when piped (curl | sudo bash)
    if [[ ! -r /dev/tty ]]; then
        warn "No TTY available — installing WITHOUT auth."
        warn "To enable auth non-interactively: INFRA_PASSWORD=xxx $0 --install"
        return
    fi
    echo -n "Set a shared password for infra services (leave empty for NO auth): "
    read -rs INFRA_PASSWORD < /dev/tty
    echo
    if [[ -n "$INFRA_PASSWORD" ]]; then
        local confirm
        echo -n "Confirm password: "
        read -rs confirm < /dev/tty
        echo
        [[ "$INFRA_PASSWORD" == "$confirm" ]] || error "Passwords do not match"
    fi
}

# GitHub download proxy prefix, for regions where GitHub releases are slow/blocked.
# Usage: GITHUB_MIRROR=https://ghfast.top/ sudo ./infra.sh --install
# The mirror is prepended to the full GitHub URL: https://ghfast.top/https://github.com/...
GITHUB_MIRROR="${GITHUB_MIRROR:-}"

# download <url> <output-file>
# Shows progress, fails fast on dead connections instead of hanging forever.
# </dev/null keeps wget from eating the script itself when piped (curl | bash).
download() {
    local url="$1" out="$2"
    if [[ -n "$GITHUB_MIRROR" ]]; then
        url="${GITHUB_MIRROR%/}/${url}"
    fi
    info "Downloading: $url"
    wget --progress=dot:giga --connect-timeout=15 --read-timeout=60 --tries=3 \
        -O "$out" "$url" </dev/null \
        || error "Download failed: $url
    If GitHub is unreachable from this server, retry with a mirror:
    curl -fsSL ... | sudo GITHUB_MIRROR=https://ghfast.top/ INFRA_PASSWORD=xxx bash -s -- --install"
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

    download "$url" /tmp/qdrant.tar.gz
    tar -xzf /tmp/qdrant.tar.gz -C "$dest"
    chmod +x "$dest/qdrant"

    # Write config alongside binary; enable api_key auth when password is set
    # (client side: zag QdrantVectorStore.server(..., api_key=...))
    if [[ -n "$INFRA_PASSWORD" ]]; then
        info "Qdrant: api_key auth enabled"
    else
        warn "Qdrant: no password set, running WITHOUT auth"
    fi
    cat > "${dest}/config.yaml" <<EOF
service:
  http_port: ${QDRANT_PORT_HTTP}
  grpc_port: ${QDRANT_PORT_GRPC}
  host: 0.0.0.0
$([ -n "$INFRA_PASSWORD" ] && echo "  api_key: ${INFRA_PASSWORD}")
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
    download "$url" "$dest/meilisearch"
    chmod +x "$dest/meilisearch"

    # Master key = shared infra password
    local master_key="$INFRA_PASSWORD"
    if [[ -n "$master_key" ]]; then
        info "Meilisearch: master key enabled"
    else
        warn "Meilisearch: no password set, running WITHOUT auth"
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

    download "$url" /tmp/rqlite.tar.gz
    tar -xzf /tmp/rqlite.tar.gz -C /tmp
    # Extracted folder: rqlite-v9.4.5-linux-amd64/
    mv /tmp/rqlite-${ver_strip}-linux-${arch}/rqlited "$dest/rqlited"
    chmod +x "$dest/rqlited"

    # Optional basic auth: write credentials file when INFRA_PASSWORD is set
    local auth_arg=""
    if [[ -n "$INFRA_PASSWORD" ]]; then
        cat > "${dest}/auth.json" <<EOF
[
  {
    "username": "rag",
    "password": "${INFRA_PASSWORD}",
    "perms": ["all"]
  }
]
EOF
        chmod 600 "${dest}/auth.json"
        auth_arg="-auth ${dest}/auth.json "
        info "rqlite: basic auth enabled (user: rag)"
    else
        warn "rqlite: no password set, running WITHOUT auth"
    fi

    # Startup args match current Windows usage:
    # rqlited.exe -node-id 1 -http-addr localhost:4001 -raft-addr localhost:4002 ./node_data
    cat > "${SERVICE_DIR}/rqlite.service" <<EOF
[Unit]
Description=rqlite distributed SQLite database
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
ExecStart=${dest}/rqlited -node-id 1 ${auth_arg}-http-addr 0.0.0.0:${RQLITE_PORT_HTTP} -http-adv-addr ${NODE_ADDR}:${RQLITE_PORT_HTTP} -raft-addr 0.0.0.0:${RQLITE_PORT_RAFT} -raft-adv-addr ${NODE_ADDR}:${RQLITE_PORT_RAFT} ${dest}/data
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

    # Password: set requirepass when INFRA_PASSWORD is available
    if [[ -n "$INFRA_PASSWORD" ]]; then
        if grep -qE '^requirepass ' "$conf"; then
            sed -i "s/^requirepass .*/requirepass ${INFRA_PASSWORD}/" "$conf"
        else
            sed -i "s/^# requirepass .*/requirepass ${INFRA_PASSWORD}/" "$conf"
        fi
        # With a password set, protected-mode can stay on safely
        sed -i "s/^protected-mode no/protected-mode yes/" "$conf"
        info "Redis: requirepass enabled"
    else
        sed -i "s/^protected-mode yes/protected-mode no/" "$conf"
        warn "Redis: no password set, running WITHOUT auth (protected-mode off)"
    fi

    # Ensure it's enabled and running with the updated config
    systemctl enable redis-server
    systemctl restart redis-server
    info "Redis installed (:${REDIS_PORT})"
}

install_falkordb() {
    info "Installing FalkorDB ${FALKORDB_VERSION} (requires Redis 8+)..."

    local dest="${ZAG_DIR}/falkordb"
    mkdir -p "$dest/data" "$dest/logs"

    # ── 1. Ensure Redis 8.0+ available (FalkorDB module requires it) ──
    local redis_bin="/usr/bin/redis-server"
    local need_build=false
    if [[ -x "$redis_bin" ]]; then
        local ver
        ver=$("$redis_bin" --version 2>&1 | grep -oP 'v=\K[0-9]+' | head -1)
        if [[ -n "$ver" ]] && [[ "$ver" -lt 8 ]]; then
            warn "System redis-server is v${ver}, FalkorDB requires v8+. Will build Redis ${REDIS_SRC_VERSION}."
            need_build=true
        fi
    else
        need_build=true
    fi

    if $need_build; then
        info "Building Redis ${REDIS_SRC_VERSION} from source..."
        apt-get install -y -qq build-essential

        local redis_url="https://github.com/redis/redis/archive/refs/tags/${REDIS_SRC_VERSION}.tar.gz"
        download "$redis_url" /tmp/redis.tar.gz
        tar -xzf /tmp/redis.tar.gz -C /tmp
        (
            cd /tmp/redis-${REDIS_SRC_VERSION}
            make -j"$(nproc)" -s
        )
        cp /tmp/redis-${REDIS_SRC_VERSION}/src/redis-server "$dest/redis-server"
        rm -rf /tmp/redis-${REDIS_SRC_VERSION} /tmp/redis.tar.gz
        redis_bin="${dest}/redis-server"
        info "Redis ${REDIS_SRC_VERSION} built successfully."
    fi

    # ── 2. Download FalkorDB .so module ──
    local arch
    case $(uname -m) in
        x86_64)  arch="x64" ;;
        aarch64) arch="arm64v8" ;;
        *)       error "Unsupported architecture: $(uname -m)" ;;
    esac

    local so_file="falkordb-${arch}.so"
    local url="https://github.com/FalkorDB/FalkorDB/releases/download/${FALKORDB_VERSION}/${so_file}"

    info "Downloading ${so_file}..."
    download "$url" "${dest}/${so_file}"

    if [[ ! -f "${dest}/${so_file}" ]]; then
        error "Download failed: ${so_file} not found."
    fi

    # .so module must have execute permissions for Redis to load it
    chmod +x "${dest}/${so_file}"

    # ── 3. Write config (password via requirepass, same as plain Redis) ──
    local auth_lines="protected-mode no"
    if [[ -n "$INFRA_PASSWORD" ]]; then
        auth_lines="protected-mode yes
requirepass ${INFRA_PASSWORD}"
        info "FalkorDB: requirepass enabled"
    else
        warn "FalkorDB: no password set, running WITHOUT auth"
    fi

    cat > "${dest}/falkordb.conf" <<EOF
# FalkorDB — Redis + Graph module
loadmodule ${dest}/${so_file}

port ${FALKORDB_PORT}
bind 0.0.0.0
${auth_lines}
daemonize no
loglevel notice
logfile ${dest}/logs/falkordb.log
dir ${dest}/data

# Persistence
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb

# Memory limit (adjust as needed)
maxmemory 2gb
maxmemory-policy noeviction
EOF
    chmod 600 "${dest}/falkordb.conf"

    # ── 4. Write systemd service ──
    cat > "${SERVICE_DIR}/falkordb.service" <<EOF
[Unit]
Description=FalkorDB graph database (Redis + graph module)
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${dest}
ExecStart=${redis_bin} ${dest}/falkordb.conf
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable falkordb
    info "FalkorDB installed (:${FALKORDB_PORT})"
}

# ============================================================
# Commands
# ============================================================
cmd_install() {
    require_root
    prompt_password

    # Abort if any service directory already exists to prevent accidental data loss
    local existing=()
    for svc in qdrant meilisearch rqlite falkordb; do
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
    for svc in qdrant meilisearch rqlite redis-server falkordb; do
        systemctl stop "$svc" 2>/dev/null || true
    done

    # Each service gets its own dir under ZAG_DIR
    mkdir -p "${ZAG_DIR}/qdrant" "${ZAG_DIR}/meilisearch" "${ZAG_DIR}/rqlite" "${ZAG_DIR}/falkordb"

    install_qdrant
    install_meilisearch
    install_rqlite
    install_redis
    install_falkordb

    # Fix ownership: dirs created under sudo, give them back to the real user
    chown -R "$CURRENT_USER:$CURRENT_USER" "$ZAG_DIR"

    # Start services after ownership is correct
    systemctl start qdrant meilisearch rqlite redis-server falkordb

    echo ""
    info "All infrastructure services installed."
    if [[ -n "$INFRA_PASSWORD" ]]; then
        info "Auth enabled for: qdrant, meilisearch, rqlite (user: rag), redis, falkordb"
        info "Client-side .env reminders (all use the same password):"
        info "  VECTOR_STORE_API_KEY=<password>"
        info "  MEILISEARCH_API_KEY=<password>"
        info "  DATABASE_URI=http://rag:<password>@${NODE_ADDR}:${RQLITE_PORT_HTTP}"
        info "  REDIS_PASSWORD=<password>"
        info "  FALKORDB_PASSWORD=<password>"
    else
        warn "All services are running WITHOUT auth."
        warn "Re-run install and enter a password (or INFRA_PASSWORD=xxx) to enable."
    fi
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
        --install|--start|--stop|--restart|--status)   ACTION="$arg" ;;
        qdrant|meilisearch|rqlite|redis|falkordb|all)  TARGET="$arg" ;;
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
