---
paths:
  - "**/*.tf"
  - "**/*.tfvars"
  - "**/doctl*"
  - "**/*digitalocean*"
  - "**/app.yaml"
  - "**/app-spec.yaml"
---

# DigitalOcean Infrastructure Rules

When working with DigitalOcean resources, follow these guidelines:

## Resource Naming

Use the format: `{project}-{environment}-{type}-{index}`

Examples:
- `webapp-prod-droplet-01`
- `api-staging-db-01`
- `frontend-dev-lb-01`

## Required Tags

Always apply these tags to resources:
- `project`: The project name
- `environment`: prod, staging, or dev
- `owner`: Team or individual responsible
- `managed-by`: terraform, claude-code, or manual

## Region Selection

- US East: `nyc1`, `nyc3` (prefer nyc3)
- US West: `sfo2`, `sfo3` (prefer sfo3)
- Europe: `ams3`, `fra1`, `lon1`
- Asia: `sgp1`, `blr1`

## Sizing Guidelines

### Droplets
- Development: `s-1vcpu-1gb` or `s-1vcpu-2gb`
- Staging: `s-2vcpu-2gb` or `s-2vcpu-4gb`
- Production: `s-4vcpu-8gb` or higher

### Kubernetes
- Development: 1-2 nodes, basic sizes
- Staging: 2-3 nodes, standard sizes
- Production: 3+ nodes with auto-scaling

## Security Requirements

1. **SSH Access**: Always use SSH keys, never passwords
2. **Firewalls**: Configure firewalls for all public-facing resources
3. **VPC**: Use private networking for internal communication
4. **Monitoring**: Enable monitoring on all production resources

## Terraform Best Practices

When working with Terraform files:

1. Use `digitalocean_project` to organize resources
2. Store state remotely in DigitalOcean Spaces
3. Use variables for sensitive values (never hardcode)
4. Apply consistent tagging via default_tags
5. Use data sources to reference existing resources

## App Platform Specifications

When creating app specs:

1. Use environment variables for configuration
2. Configure health checks for all services
3. Set appropriate instance sizes
4. Enable deploy-on-push for CI/CD
5. Use managed databases when possible

## Cost Awareness

- Always mention estimated costs when creating resources
- Suggest cost-effective alternatives when appropriate
- Warn about expensive operations (large droplets, HA databases)
- Consider reserved instances for long-running production workloads
