#!/bin/bash
# setup.sh
#
# Sets up the DigitalOcean Claude Code module.
# Run this script to verify prerequisites and configure authentication.

set -e

echo "================================================"
echo "  DigitalOcean Claude Code Module Setup"
echo "================================================"
echo ""

# Check for doctl
if command -v doctl &> /dev/null; then
    DOCTL_VERSION=$(doctl version | head -1)
    echo "[OK] doctl found: $DOCTL_VERSION"
else
    echo "[!!] doctl not found."
    echo "     Install from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    echo ""
    echo "     Quick install:"
    echo "       macOS:   brew install doctl"
    echo "       Linux:   snap install doctl"
    echo "       Windows: scoop install doctl"
    echo ""
    echo "     doctl is recommended but not strictly required if using the MCP server."
fi

echo ""

# Check for API token
if [ -n "$DIGITALOCEAN_API_TOKEN" ]; then
    echo "[OK] DIGITALOCEAN_API_TOKEN is set"

    # Verify token
    echo "     Verifying authentication..."
    if command -v doctl &> /dev/null; then
        if doctl account get --access-token "$DIGITALOCEAN_API_TOKEN" > /dev/null 2>&1; then
            ACCOUNT_EMAIL=$(doctl account get --access-token "$DIGITALOCEAN_API_TOKEN" --format Email --no-header 2>/dev/null)
            echo "[OK] Authenticated as: $ACCOUNT_EMAIL"
        else
            echo "[!!] Token verification failed. Check your API token."
        fi
    else
        # Verify via curl
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer $DIGITALOCEAN_API_TOKEN" \
            https://api.digitalocean.com/v2/account)
        if [ "$HTTP_STATUS" = "200" ]; then
            echo "[OK] API token is valid"
        else
            echo "[!!] Token verification failed (HTTP $HTTP_STATUS). Check your API token."
        fi
    fi
else
    echo "[!!] DIGITALOCEAN_API_TOKEN is not set."
    echo ""
    echo "     Get your token from:"
    echo "     https://cloud.digitalocean.com/account/api/tokens"
    echo ""
    read -p "     Enter your DigitalOcean API token (or press Enter to skip): " TOKEN
    if [ -n "$TOKEN" ]; then
        export DIGITALOCEAN_API_TOKEN="$TOKEN"
        echo ""
        echo "     Token set for this session."
        echo "     To persist, add to your shell profile:"
        echo "       export DIGITALOCEAN_API_TOKEN=$TOKEN"
    fi
fi

echo ""

# Check for Python (MCP server)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "[OK] Python found: $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "[OK] Python found: $PYTHON_VERSION"
else
    echo "[!!] Python not found. Required for MCP server."
fi

# Check for requests package
if python3 -c "import requests" 2>/dev/null || python -c "import requests" 2>/dev/null; then
    echo "[OK] requests package installed"
else
    echo "[!!] requests package not found."
    echo "     Install with: pip install requests"
fi

# Check for jq (used by hook scripts)
if command -v jq &> /dev/null; then
    echo "[OK] jq found"
else
    echo "[!!] jq not found. Required for hook validation scripts."
    echo "     Install: brew install jq (macOS) / apt install jq (Linux)"
fi

echo ""

# Make scripts executable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null
echo "[OK] Hook scripts set to executable"

echo ""
echo "================================================"
echo "  Setup Summary"
echo "================================================"
echo ""
echo "Module directory structure:"
echo ""
echo "  .claude/"
echo "    settings.json     - Permission rules"
echo "    CLAUDE.md          - Project instructions"
echo "    rules/             - Conditional rules"
echo "    skills/            - DO management skills"
echo "      do-droplets/     - Droplet management"
echo "      do-kubernetes/   - Kubernetes management"
echo "      do-dns/          - DNS management"
echo "      do-spaces/       - Spaces storage"
echo "      do-deploy/       - App deployment"
echo "    agents/            - Specialized agents"
echo "      do-manager.md    - Infrastructure manager"
echo "      do-monitor.md    - Monitoring specialist"
echo "      do-deploy.md     - Deployment specialist"
echo "  mcp-server/          - Custom MCP server"
echo "  scripts/             - Hook scripts"
echo ""
echo "To use with Claude Code:"
echo "  1. Copy this module to your project"
echo "  2. Set DIGITALOCEAN_API_TOKEN"
echo "  3. Start Claude Code"
echo ""
echo "Try these commands:"
echo "  'List my droplets'"
echo "  'Create a new Ubuntu droplet in NYC'"
echo "  '/do-droplets create a staging server'"
echo "  'Check my DigitalOcean balance'"
echo ""
