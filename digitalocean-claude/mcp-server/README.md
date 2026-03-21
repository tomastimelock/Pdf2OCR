# DigitalOcean MCP Server

A Model Context Protocol (MCP) server that provides Claude Code with tools for managing DigitalOcean infrastructure.

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set API token

```bash
export DIGITALOCEAN_API_TOKEN=your_token_here
```

### Register with Claude Code

```bash
claude mcp add --transport stdio \
  --env DIGITALOCEAN_API_TOKEN=$DIGITALOCEAN_API_TOKEN \
  digitalocean -- python mcp-server/digitalocean_mcp.py
```

Or add to `.mcp.json` for project-level configuration.

## Available Tools

### Account
- `get_account` - Account information
- `get_balance` - Billing and balance

### Compute
- `list_droplets` - List all droplets
- `get_droplet` - Get droplet details
- `create_droplet` - Create a new droplet
- `delete_droplet` - Delete a droplet
- `droplet_action` - Perform droplet actions (reboot, resize, snapshot, etc.)
- `list_ssh_keys` - List SSH keys
- `list_regions` - List available regions
- `list_sizes` - List available sizes

### Networking
- `list_domains` - List domains
- `list_domain_records` - List DNS records
- `create_domain_record` - Create DNS record
- `list_firewalls` - List firewalls
- `list_load_balancers` - List load balancers

### Storage
- `list_volumes` - List block storage volumes

### Databases
- `list_databases` - List managed databases
- `get_database` - Get database details

### Kubernetes
- `list_kubernetes_clusters` - List K8s clusters
- `get_kubernetes_cluster` - Get cluster details

### App Platform
- `list_apps` - List applications
- `get_app` - Get app details
- `list_app_deployments` - List app deployments
