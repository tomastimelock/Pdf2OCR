# DigitalOcean Claude Code Integration Module

A comprehensive guide for creating a reusable module that gives Claude Code the power to interact with DigitalOcean. This module can be copied into any project to enable DigitalOcean management capabilities.

## Table of Contents

- [Overview](#overview)
- [Module Structure](#module-structure)
- [MCP Server Setup](#mcp-server-setup)
- [Skills Configuration](#skills-configuration)
- [Sub-Agents Configuration](#sub-agents-configuration)
- [Memory Files (CLAUDE.md)](#memory-files-claudemd)
- [Settings Configuration](#settings-configuration)
- [Installation & Usage](#installation--usage)

---

## Overview

This module provides Claude Code with the ability to:
- Manage DigitalOcean Droplets (create, list, delete, resize)
- Manage Kubernetes clusters
- Handle DNS records and domains
- Manage Spaces (object storage)
- Deploy applications via App Platform
- Monitor resources and billing

### Prerequisites

- DigitalOcean API Token (from https://cloud.digitalocean.com/account/api/tokens)
- `doctl` CLI installed (optional but recommended)
- Python 3.9+ (for MCP server)

---

## Module Structure

```
digitalocean-claude/
├── .claude/
│   ├── settings.json           # Project-level permissions
│   ├── CLAUDE.md              # Project memory/instructions
│   ├── rules/
│   │   └── digitalocean.md    # DigitalOcean-specific rules
│   ├── skills/
│   │   ├── do-droplets/
│   │   │   └── SKILL.md       # Droplet management skill
│   │   ├── do-kubernetes/
│   │   │   └── SKILL.md       # Kubernetes management skill
│   │   ├── do-dns/
│   │   │   └── SKILL.md       # DNS management skill
│   │   ├── do-spaces/
│   │   │   └── SKILL.md       # Spaces (S3) management skill
│   │   └── do-deploy/
│   │       └── SKILL.md       # App deployment skill
│   └── agents/
│       ├── do-manager.md      # General DO management agent
│       ├── do-monitor.md      # Monitoring/alerting agent
│       └── do-deploy.md       # Deployment specialist agent
├── mcp-server/
│   ├── digitalocean_mcp.py    # MCP server implementation
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # MCP server docs
├── scripts/
│   ├── validate-do-command.sh # Hook for command validation
│   └── setup.sh               # Installation script
├── .mcp.json                  # Project MCP server config
└── .env.example               # Environment variables template
```

---

## MCP Server Setup

### Option 1: Use doctl via Bash (Simple)

If you have `doctl` installed, you can use it directly without a custom MCP server.

Add to your `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(doctl:*)",
      "Bash(doctl compute droplet list:*)",
      "Bash(doctl compute droplet create:*)",
      "Bash(doctl kubernetes cluster list:*)"
    ],
    "deny": [
      "Bash(doctl compute droplet delete:*)"
    ],
    "ask": [
      "Bash(doctl compute droplet delete:*)"
    ]
  }
}
```

### Option 2: Custom MCP Server (Recommended)

Create a custom MCP server that wraps the DigitalOcean API.

#### `mcp-server/digitalocean_mcp.py`

```python
#!/usr/bin/env python3
"""
DigitalOcean MCP Server for Claude Code

Provides tools for managing DigitalOcean resources via the Model Context Protocol.
"""

import os
import json
import sys
from typing import Any
import requests

# DigitalOcean API configuration
DO_API_URL = "https://api.digitalocean.com/v2"
DO_TOKEN = os.environ.get("DIGITALOCEAN_API_TOKEN")

def get_headers():
    return {
        "Authorization": f"Bearer {DO_TOKEN}",
        "Content-Type": "application/json"
    }

# Tool definitions for MCP
TOOLS = [
    {
        "name": "list_droplets",
        "description": "List all DigitalOcean droplets in your account",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag_name": {
                    "type": "string",
                    "description": "Filter droplets by tag name (optional)"
                }
            }
        }
    },
    {
        "name": "create_droplet",
        "description": "Create a new DigitalOcean droplet",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Droplet name"},
                "region": {"type": "string", "description": "Region slug (e.g., nyc1, sfo3)"},
                "size": {"type": "string", "description": "Size slug (e.g., s-1vcpu-1gb)"},
                "image": {"type": "string", "description": "Image slug or ID (e.g., ubuntu-22-04-x64)"},
                "ssh_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "SSH key IDs or fingerprints"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to apply to the droplet"
                }
            },
            "required": ["name", "region", "size", "image"]
        }
    },
    {
        "name": "get_droplet",
        "description": "Get details of a specific droplet",
        "inputSchema": {
            "type": "object",
            "properties": {
                "droplet_id": {"type": "string", "description": "Droplet ID"}
            },
            "required": ["droplet_id"]
        }
    },
    {
        "name": "delete_droplet",
        "description": "Delete a DigitalOcean droplet (requires confirmation)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "droplet_id": {"type": "string", "description": "Droplet ID to delete"},
                "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"}
            },
            "required": ["droplet_id", "confirm"]
        }
    },
    {
        "name": "list_kubernetes_clusters",
        "description": "List all Kubernetes clusters",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "list_domains",
        "description": "List all domains in your account",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "create_dns_record",
        "description": "Create a DNS record for a domain",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain name"},
                "type": {"type": "string", "description": "Record type (A, AAAA, CNAME, MX, TXT, NS, SRV)"},
                "name": {"type": "string", "description": "Record name (@ for root)"},
                "data": {"type": "string", "description": "Record data (IP, hostname, etc.)"},
                "ttl": {"type": "integer", "description": "TTL in seconds (default: 3600)"}
            },
            "required": ["domain", "type", "name", "data"]
        }
    },
    {
        "name": "list_spaces",
        "description": "List all Spaces (object storage buckets)",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_account_balance",
        "description": "Get current account balance and billing info",
        "inputSchema": {"type": "object", "properties": {}}
    }
]

def list_droplets(tag_name: str = None) -> dict:
    url = f"{DO_API_URL}/droplets"
    if tag_name:
        url += f"?tag_name={tag_name}"
    response = requests.get(url, headers=get_headers())
    return response.json()

def create_droplet(name: str, region: str, size: str, image: str,
                   ssh_keys: list = None, tags: list = None) -> dict:
    payload = {
        "name": name,
        "region": region,
        "size": size,
        "image": image
    }
    if ssh_keys:
        payload["ssh_keys"] = ssh_keys
    if tags:
        payload["tags"] = tags

    response = requests.post(f"{DO_API_URL}/droplets",
                            headers=get_headers(),
                            json=payload)
    return response.json()

def get_droplet(droplet_id: str) -> dict:
    response = requests.get(f"{DO_API_URL}/droplets/{droplet_id}",
                           headers=get_headers())
    return response.json()

def delete_droplet(droplet_id: str, confirm: bool) -> dict:
    if not confirm:
        return {"error": "Deletion not confirmed. Set confirm=true to delete."}
    response = requests.delete(f"{DO_API_URL}/droplets/{droplet_id}",
                              headers=get_headers())
    if response.status_code == 204:
        return {"success": True, "message": f"Droplet {droplet_id} deleted"}
    return response.json()

def list_kubernetes_clusters() -> dict:
    response = requests.get(f"{DO_API_URL}/kubernetes/clusters",
                           headers=get_headers())
    return response.json()

def list_domains() -> dict:
    response = requests.get(f"{DO_API_URL}/domains", headers=get_headers())
    return response.json()

def create_dns_record(domain: str, record_type: str, name: str,
                      data: str, ttl: int = 3600) -> dict:
    payload = {
        "type": record_type,
        "name": name,
        "data": data,
        "ttl": ttl
    }
    response = requests.post(f"{DO_API_URL}/domains/{domain}/records",
                            headers=get_headers(),
                            json=payload)
    return response.json()

def list_spaces() -> dict:
    # Note: Spaces uses a different API endpoint and authentication
    # This is a simplified example
    return {"message": "Use doctl or s3cmd for Spaces management"}

def get_account_balance() -> dict:
    response = requests.get(f"{DO_API_URL}/customers/my/balance",
                           headers=get_headers())
    return response.json()

def handle_tool_call(name: str, arguments: dict) -> Any:
    """Route tool calls to appropriate functions."""
    handlers = {
        "list_droplets": lambda: list_droplets(arguments.get("tag_name")),
        "create_droplet": lambda: create_droplet(**arguments),
        "get_droplet": lambda: get_droplet(arguments["droplet_id"]),
        "delete_droplet": lambda: delete_droplet(
            arguments["droplet_id"],
            arguments.get("confirm", False)
        ),
        "list_kubernetes_clusters": list_kubernetes_clusters,
        "list_domains": list_domains,
        "create_dns_record": lambda: create_dns_record(
            arguments["domain"],
            arguments["type"],
            arguments["name"],
            arguments["data"],
            arguments.get("ttl", 3600)
        ),
        "list_spaces": list_spaces,
        "get_account_balance": get_account_balance
    }

    if name in handlers:
        return handlers[name]()
    return {"error": f"Unknown tool: {name}"}

def main():
    """MCP server main loop using stdio transport."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            method = request.get("method")

            if method == "tools/list":
                response = {"tools": TOOLS}
            elif method == "tools/call":
                tool_name = request["params"]["name"]
                arguments = request["params"].get("arguments", {})
                result = handle_tool_call(tool_name, arguments)
                response = {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            else:
                response = {"error": f"Unknown method: {method}"}

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    if not DO_TOKEN:
        print("Error: DIGITALOCEAN_API_TOKEN environment variable not set",
              file=sys.stderr)
        sys.exit(1)
    main()
```

#### `mcp-server/requirements.txt`

```
requests>=2.28.0
```

### Configure the MCP Server

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "digitalocean": {
      "command": "python",
      "args": ["${CLAUDE_PROJECT_ROOT}/mcp-server/digitalocean_mcp.py"],
      "env": {
        "DIGITALOCEAN_API_TOKEN": "${DIGITALOCEAN_API_TOKEN}"
      }
    }
  }
}
```

Or add via CLI:

```bash
claude mcp add --transport stdio --env DIGITALOCEAN_API_TOKEN=$DIGITALOCEAN_API_TOKEN \
  --scope project digitalocean -- python mcp-server/digitalocean_mcp.py
```

---

## Skills Configuration

Skills are reusable prompts that teach Claude specific workflows.

### Droplet Management Skill

Create `.claude/skills/do-droplets/SKILL.md`:

```yaml
---
name: do-droplets
description: Manage DigitalOcean droplets - create, list, resize, and delete virtual machines. Use when working with DO infrastructure.
allowed-tools: Bash(doctl:*), Read, Write
---

# DigitalOcean Droplet Management

You are a DigitalOcean infrastructure expert. Help users manage their droplets efficiently.

## Available Operations

### List Droplets
```bash
doctl compute droplet list --format ID,Name,PublicIPv4,Region,Size,Status
```

### Create Droplet
```bash
doctl compute droplet create <name> \
  --region <region> \
  --size <size> \
  --image <image> \
  --ssh-keys <key-id> \
  --tag-name <tag> \
  --wait
```

Common sizes:
- `s-1vcpu-1gb` - Basic ($6/mo)
- `s-2vcpu-2gb` - Standard ($18/mo)
- `s-4vcpu-8gb` - Production ($48/mo)

Common images:
- `ubuntu-22-04-x64`
- `debian-12-x64`
- `rocky-9-x64`

### Delete Droplet
Always confirm with the user before deletion:
```bash
doctl compute droplet delete <droplet-id> --force
```

### Resize Droplet
```bash
doctl compute droplet-action resize <droplet-id> --size <new-size> --wait
```

## Best Practices
1. Always use SSH keys instead of passwords
2. Apply appropriate tags for organization
3. Use private networking when possible
4. Enable monitoring and backups for production
```

### Kubernetes Skill

Create `.claude/skills/do-kubernetes/SKILL.md`:

```yaml
---
name: do-kubernetes
description: Manage DigitalOcean Kubernetes (DOKS) clusters. Use for container orchestration tasks.
allowed-tools: Bash(doctl:*), Bash(kubectl:*), Read, Write
---

# DigitalOcean Kubernetes Management

## Cluster Operations

### List Clusters
```bash
doctl kubernetes cluster list
```

### Create Cluster
```bash
doctl kubernetes cluster create <name> \
  --region <region> \
  --version <k8s-version> \
  --node-pool "name=default;size=s-2vcpu-4gb;count=3" \
  --wait
```

### Get Kubeconfig
```bash
doctl kubernetes cluster kubeconfig save <cluster-id>
```

### Scale Node Pool
```bash
doctl kubernetes cluster node-pool update <cluster-id> <pool-id> --count <new-count>
```

## Best Practices
1. Use multiple node pools for different workloads
2. Enable auto-scaling for production
3. Use managed databases instead of running DBs in cluster
4. Set up container registry for private images
```

### DNS Management Skill

Create `.claude/skills/do-dns/SKILL.md`:

```yaml
---
name: do-dns
description: Manage DigitalOcean DNS records and domains. Use for domain configuration tasks.
allowed-tools: Bash(doctl:*), Read
---

# DigitalOcean DNS Management

## Domain Operations

### List Domains
```bash
doctl compute domain list
```

### Add Domain
```bash
doctl compute domain create <domain-name>
```

### List Records
```bash
doctl compute domain records list <domain-name>
```

### Create Record
```bash
doctl compute domain records create <domain-name> \
  --record-type <A|AAAA|CNAME|MX|TXT|NS|SRV> \
  --record-name <name> \
  --record-data <value> \
  --record-ttl 3600
```

## Common Records
- A Record: Point domain to IPv4 address
- AAAA Record: Point domain to IPv6 address
- CNAME: Create alias to another domain
- MX: Configure email routing
- TXT: Add verification records (SPF, DKIM, etc.)
```

### Deployment Skill

Create `.claude/skills/do-deploy/SKILL.md`:

```yaml
---
name: do-deploy
description: Deploy applications to DigitalOcean App Platform. Use for deploying web apps, APIs, and static sites.
disable-model-invocation: true
allowed-tools: Bash(doctl:*), Read, Write
---

# DigitalOcean App Platform Deployment

## App Operations

### List Apps
```bash
doctl apps list
```

### Create App from Spec
```bash
doctl apps create --spec app.yaml
```

### Deploy Update
```bash
doctl apps create-deployment <app-id>
```

## Sample App Spec (app.yaml)

```yaml
name: my-app
region: nyc
services:
  - name: api
    github:
      repo: username/repo
      branch: main
      deploy_on_push: true
    build_command: npm run build
    run_command: npm start
    http_port: 8080
    instance_size_slug: basic-xxs
    instance_count: 1
    envs:
      - key: NODE_ENV
        value: production
```

## Best Practices
1. Use environment variables for secrets
2. Enable deploy-on-push for CI/CD
3. Configure health checks
4. Use managed databases via `databases` spec
```

---

## Sub-Agents Configuration

Sub-agents are specialized AI assistants for specific tasks.

### General DO Manager Agent

Create `.claude/agents/do-manager.md`:

```yaml
---
name: do-manager
description: DigitalOcean infrastructure manager. Use proactively for any DO resource management tasks including droplets, databases, and networking.
tools: Bash, Read, Grep, Glob
model: sonnet
---

You are a DigitalOcean infrastructure specialist. You help users manage their cloud resources efficiently and securely.

## Your Capabilities

1. **Droplet Management**: Create, configure, and manage virtual machines
2. **Networking**: Configure VPCs, firewalls, load balancers
3. **Storage**: Manage volumes and Spaces
4. **Databases**: Work with managed databases
5. **Kubernetes**: Manage DOKS clusters

## Guidelines

- Always verify the current state before making changes
- Ask for confirmation before destructive operations
- Suggest cost-effective alternatives when appropriate
- Follow DigitalOcean best practices

## Common Commands

List all resources:
```bash
doctl compute droplet list
doctl databases list
doctl kubernetes cluster list
doctl compute volume list
```

Check account:
```bash
doctl account get
doctl balance get
```
```

### Monitoring Agent

Create `.claude/agents/do-monitor.md`:

```yaml
---
name: do-monitor
description: DigitalOcean monitoring and alerting specialist. Use for checking resource status, setting up alerts, and troubleshooting performance issues.
tools: Bash, Read, Grep
model: haiku
---

You are a monitoring specialist for DigitalOcean infrastructure.

## Your Focus

1. Check resource health and status
2. Review metrics and logs
3. Set up monitoring alerts
4. Troubleshoot performance issues

## Monitoring Commands

### Droplet Metrics
```bash
doctl monitoring droplet bandwidth get <droplet-id>
doctl monitoring droplet cpu get <droplet-id>
doctl monitoring droplet memory free get <droplet-id>
```

### Alert Policies
```bash
doctl monitoring alert list
doctl monitoring alert create --type <type> --compare <gt|lt> --value <threshold>
```

## Key Metrics to Watch

- CPU utilization > 80%
- Memory usage > 85%
- Disk usage > 90%
- Network bandwidth spikes
- 5xx error rates
```

### Deployment Agent

Create `.claude/agents/do-deploy.md`:

```yaml
---
name: do-deploy
description: Deployment specialist for DigitalOcean App Platform and container deployments. Use when deploying or updating applications.
tools: Bash, Read, Write, Glob
model: sonnet
permissionMode: acceptEdits
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-deploy-command.sh"
---

You are a deployment specialist for DigitalOcean infrastructure.

## Deployment Workflow

1. **Review**: Check current deployment status
2. **Prepare**: Validate configuration and dependencies
3. **Deploy**: Execute deployment with proper rollback plan
4. **Verify**: Confirm successful deployment
5. **Monitor**: Watch for issues post-deployment

## Pre-Deployment Checklist

- [ ] Environment variables configured
- [ ] Secrets properly stored
- [ ] Database migrations ready
- [ ] Health check endpoints configured
- [ ] Rollback plan documented

## App Platform Commands

```bash
# List apps
doctl apps list

# Get app details
doctl apps get <app-id>

# Trigger deployment
doctl apps create-deployment <app-id>

# View logs
doctl apps logs <app-id>

# View deployment history
doctl apps list-deployments <app-id>
```
```

---

## Memory Files (CLAUDE.md)

Create `.claude/CLAUDE.md` for project-level instructions:

```markdown
# DigitalOcean Project Configuration

This project is configured to work with DigitalOcean infrastructure.

## Authentication

The DigitalOcean API token should be set as:
```bash
export DIGITALOCEAN_API_TOKEN=your_token_here
```

Or use `doctl auth init` for CLI authentication.

## Project Resources

- **Region**: nyc3 (primary), sfo3 (secondary)
- **Default SSH Key**: Use key ID from `doctl compute ssh-key list`
- **Naming Convention**: `{project}-{env}-{service}` (e.g., `myapp-prod-api`)

## Common Workflows

### Deploy New Service
1. Create droplet with appropriate size
2. Configure firewall rules
3. Set up DNS records
4. Deploy application code

### Scale Infrastructure
1. Check current resource usage
2. Resize or add droplets as needed
3. Update load balancer if applicable

## Safety Rules

- NEVER delete production resources without explicit confirmation
- ALWAYS backup data before migrations
- Use staging environment for testing changes
- Tag all resources for cost tracking

@docs/digitalocean-architecture.md
@.env.example
```

### Modular Rules

Create `.claude/rules/digitalocean.md`:

```yaml
---
paths:
  - "**/*.tf"
  - "**/doctl*"
  - "**/*digitalocean*"
---

# DigitalOcean Infrastructure Rules

When working with DigitalOcean resources:

1. **Resource Naming**: Use format `{project}-{environment}-{type}-{index}`
2. **Tagging**: Always apply tags: `project`, `environment`, `owner`
3. **Regions**: Prefer `nyc3` for US East, `sfo3` for US West
4. **Sizing**: Start small and scale up as needed

## Terraform Best Practices

- Use `digitalocean_project` to organize resources
- Store state remotely in DO Spaces
- Use variables for sensitive values
- Apply consistent tagging

## Security Requirements

- Enable VPC for private networking
- Use firewalls to restrict access
- Enable monitoring on all production droplets
- Regular security audits
```

---

## Settings Configuration

Create `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(doctl compute droplet list:*)",
      "Bash(doctl compute droplet get:*)",
      "Bash(doctl kubernetes cluster list:*)",
      "Bash(doctl compute domain list:*)",
      "Bash(doctl compute domain records list:*)",
      "Bash(doctl account get:*)",
      "Bash(doctl balance get:*)",
      "Bash(doctl apps list:*)",
      "Bash(doctl apps get:*)",
      "Read(*.yaml)",
      "Read(*.tf)"
    ],
    "ask": [
      "Bash(doctl compute droplet create:*)",
      "Bash(doctl compute droplet delete:*)",
      "Bash(doctl kubernetes cluster create:*)",
      "Bash(doctl apps create:*)",
      "Bash(doctl apps create-deployment:*)"
    ],
    "deny": [
      "Bash(doctl auth:*)",
      "Bash(rm -rf:*)"
    ]
  },
  "env": {
    "DO_DEFAULT_REGION": "nyc3"
  }
}
```

---

## Installation & Usage

### Quick Setup Script

Create `scripts/setup.sh`:

```bash
#!/bin/bash
set -e

echo "Setting up DigitalOcean Claude Module..."

# Check for doctl
if ! command -v doctl &> /dev/null; then
    echo "doctl not found. Install from: https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check for API token
if [ -z "$DIGITALOCEAN_API_TOKEN" ]; then
    echo "DIGITALOCEAN_API_TOKEN not set."
    echo "Get your token from: https://cloud.digitalocean.com/account/api/tokens"
    read -p "Enter your DigitalOcean API token: " token
    export DIGITALOCEAN_API_TOKEN=$token
    echo "export DIGITALOCEAN_API_TOKEN=$token" >> ~/.bashrc
fi

# Verify authentication
echo "Verifying DigitalOcean authentication..."
doctl account get

# Create directory structure
mkdir -p .claude/skills/do-droplets
mkdir -p .claude/skills/do-kubernetes
mkdir -p .claude/skills/do-dns
mkdir -p .claude/skills/do-deploy
mkdir -p .claude/agents
mkdir -p .claude/rules
mkdir -p mcp-server
mkdir -p scripts

echo "Setup complete! Claude Code can now manage your DigitalOcean resources."
echo ""
echo "Try these commands in Claude Code:"
echo "  - 'List my droplets'"
echo "  - 'Create a new Ubuntu droplet in NYC'"
echo "  - 'Check my account balance'"
```

### Copy to New Project

To use this module in another project:

```bash
# Clone or copy the digitalocean-claude directory
cp -r digitalocean-claude/.claude /path/to/your/project/
cp -r digitalocean-claude/mcp-server /path/to/your/project/
cp -r digitalocean-claude/scripts /path/to/your/project/
cp digitalocean-claude/.mcp.json /path/to/your/project/

# Set up environment
cd /path/to/your/project
export DIGITALOCEAN_API_TOKEN=your_token_here

# Start Claude Code
claude
```

### Usage Examples

Once configured, you can ask Claude:

```
List all my DigitalOcean droplets

Create a new droplet named web-server in NYC with Ubuntu 22.04

Use the do-deploy skill to deploy my app to App Platform

Check my current DigitalOcean balance

Set up DNS records for mydomain.com pointing to 192.168.1.1
```

---

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   ```bash
   doctl auth init
   # Or set DIGITALOCEAN_API_TOKEN environment variable
   ```

2. **MCP Server Not Loading**
   - Check `.mcp.json` syntax
   - Verify Python path
   - Run `claude mcp list` to check status

3. **Skills Not Available**
   - Run `/agents` to see loaded agents
   - Check skill file syntax (YAML frontmatter)
   - Ensure files are in `.claude/skills/` directory

4. **Permission Denied**
   - Review `.claude/settings.json` permissions
   - Check if command is in `deny` list
   - Use `/permissions` to manage permissions

---

## Security Considerations

1. **API Token Security**
   - Never commit tokens to version control
   - Use environment variables or secure vaults
   - Rotate tokens regularly

2. **Permission Boundaries**
   - Use `ask` mode for destructive operations
   - Limit access to production resources
   - Implement approval workflows for critical changes

3. **Audit Logging**
   - Enable DigitalOcean audit logs
   - Review Claude Code session logs
   - Track infrastructure changes
