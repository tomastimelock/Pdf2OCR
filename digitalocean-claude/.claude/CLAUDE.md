# DigitalOcean Project Configuration

This project is configured to work with DigitalOcean infrastructure using Claude Code.

## Authentication

The DigitalOcean API token should be set as an environment variable:

```bash
export DIGITALOCEAN_API_TOKEN=your_token_here
```

Or authenticate via doctl CLI:

```bash
doctl auth init
```

## Available Skills

- `/do-droplets` - Manage DigitalOcean droplets
- `/do-kubernetes` - Manage Kubernetes clusters
- `/do-dns` - Manage DNS records and domains
- `/do-spaces` - Manage Spaces object storage
- `/do-deploy` - Deploy to App Platform

## Available Agents

- `do-manager` - General infrastructure management
- `do-monitor` - Monitoring and alerting
- `do-deploy` - Deployment specialist

## Project Conventions

### Naming Convention
Use the format: `{project}-{environment}-{service}-{index}`

Examples:
- `myapp-prod-web-01`
- `myapp-staging-api-01`
- `myapp-dev-db-01`

### Required Tags
All resources should have these tags:
- `project` - Project name
- `environment` - prod, staging, dev
- `owner` - Team or person responsible
- `created-by` - claude-code or manual

### Default Regions
- Primary: `nyc3` (US East)
- Secondary: `sfo3` (US West)
- Europe: `ams3` (Amsterdam)

## Common Workflows

### Create New Droplet
1. List available SSH keys: `doctl compute ssh-key list`
2. Create droplet with appropriate size and tags
3. Configure firewall rules
4. Set up DNS records if needed

### Deploy Application
1. Verify app spec configuration
2. Run pre-deployment checks
3. Create deployment
4. Monitor deployment status
5. Verify health checks

### Scale Infrastructure
1. Check current resource utilization
2. Plan scaling changes
3. Execute scaling operations
4. Verify new resource health

## Safety Rules

- NEVER delete production resources without explicit user confirmation
- ALWAYS verify resource state before modifications
- Use staging environment for testing infrastructure changes
- Backup data before any migration or deletion
- Tag all resources for cost tracking and organization

## Quick Reference

### List Resources
```bash
doctl compute droplet list --format ID,Name,PublicIPv4,Region,Size,Status
doctl kubernetes cluster list
doctl databases list
doctl apps list
```

### Check Account
```bash
doctl account get
doctl balance get
```

## File References

@.env.example
@scripts/setup.sh
