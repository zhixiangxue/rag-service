#!/bin/bash
# ============================================================
# Milvus Standalone 启动脚本
# ============================================================
# 
# ⚠️  重要：此脚本必须在 WSL2 (Ubuntu) 中运行！
# 
# 使用方法：
#   bash start_milvus.sh          # 启动 Milvus
#   bash start_milvus.sh stop     # 停止 Milvus
#   bash start_milvus.sh restart  # 重启 Milvus
#   bash start_milvus.sh delete   # 删除 Milvus 数据和容器
#
# 连接信息：
#   - gRPC 端口: localhost:19530 (Python 客户端使用)
#   - Web UI: http://localhost:9091/webui/
# ============================================================

set -e

SCRIPT_NAME="standalone_embed.sh"
SCRIPT_URL="https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Milvus Standalone 管理脚本${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# 检查是否在 WSL 中
if ! grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo -e "${RED}⚠️  警告：此脚本应该在 WSL2 中运行！${NC}"
    echo ""
fi

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker 未安装！${NC}"
    echo ""
    echo "请先安装 Docker："
    echo "  sudo apt-get update"
    echo "  sudo apt-get install docker.io -y"
    echo "  sudo service docker start"
    exit 1
fi

# 检查 Docker 是否运行
if ! sudo docker info &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker 未运行，正在启动...${NC}"
    sudo service docker start
    sleep 2
fi

# 下载安装脚本（如果不存在）
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "${YELLOW}📥 正在下载 Milvus 安装脚本...${NC}"
    curl -sfL "$SCRIPT_URL" -o "$SCRIPT_NAME"
    echo -e "${GREEN}✓ 下载完成${NC}"
    echo ""
fi

# 执行命令
ACTION=${1:-start}

case "$ACTION" in
    start)
        echo -e "${YELLOW}🚀 启动 Milvus...${NC}"
        bash "$SCRIPT_NAME" start
        echo ""
        echo -e "${GREEN}✓ Milvus 已启动！${NC}"
        echo ""
        echo "连接信息："
        echo "  - gRPC 端口: localhost:19530"
        echo "  - Web UI: http://localhost:9091/webui/"
        echo ""
        echo "在 Windows Python 中连接："
        echo "  store = MilvusVectorStore.server(host='localhost', port=19530)"
        ;;
    stop)
        echo -e "${YELLOW}⏸️  停止 Milvus...${NC}"
        bash "$SCRIPT_NAME" stop
        echo -e "${GREEN}✓ Milvus 已停止${NC}"
        ;;
    restart)
        echo -e "${YELLOW}🔄 重启 Milvus...${NC}"
        bash "$SCRIPT_NAME" restart
        echo -e "${GREEN}✓ Milvus 已重启${NC}"
        ;;
    delete)
        echo -e "${RED}🗑️  删除 Milvus 数据和容器...${NC}"
        read -p "确定要删除吗？(y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            bash "$SCRIPT_NAME" delete
            echo -e "${GREEN}✓ Milvus 已删除${NC}"
        else
            echo "已取消"
        fi
        ;;
    *)
        echo -e "${RED}❌ 未知命令: $ACTION${NC}"
        echo ""
        echo "使用方法："
        echo "  bash start_milvus.sh          # 启动"
        echo "  bash start_milvus.sh stop     # 停止"
        echo "  bash start_milvus.sh restart  # 重启"
        echo "  bash start_milvus.sh delete   # 删除"
        exit 1
        ;;
esac
