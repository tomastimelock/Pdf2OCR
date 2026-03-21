---
name: do-deploy
description: Deploy applications to DigitalOcean App Platform - create apps, manage deployments, configure services. Use when deploying web apps, APIs, workers, and static sites.
disable-model-invocation: true
allowed-tools: Bash(doctl:*), Read, Write
---

# DigitalOcean App Platform Deployment

You are a deployment expert specializing in DigitalOcean App Platform.

## App Operations

### List Apps
```bash
doctl apps list --format ID,DefaultIngress,ActiveDeployment.Phase,UpdatedAt
```

### Get App Details
```bash
doctl apps get <app-id>
```

### Create App from Spec
```bash
doctl apps create --spec app.yaml
```

### Update App
```bash
doctl apps update <app-id> --spec app.yaml
```

### Delete App
```bash
doctl apps delete <app-id> --force
```

## Deployment Operations

### Trigger Deployment
```bash
doctl apps create-deployment <app-id>
```

### List Deployments
```bash
doctl apps list-deployments <app-id>
```

### Get Deployment Details
```bash
doctl apps get-deployment <app-id> <deployment-id>
```

### View Logs
```bash
# All components
doctl apps logs <app-id>

# Specific component
doctl apps logs <app-id> --component <component-name>

# Follow logs
doctl apps logs <app-id> --follow

# Deployment logs
doctl apps logs <app-id> --type deploy
```

## App Spec Reference

### Basic Web Service
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
    routes:
      - path: /
    envs:
      - key: NODE_ENV
        value: production
      - key: DATABASE_URL
        scope: RUN_TIME
        type: SECRET
```

### Static Site
```yaml
name: my-website
region: nyc
static_sites:
  - name: frontend
    github:
      repo: username/frontend
      branch: main
      deploy_on_push: true
    build_command: npm run build
    output_dir: dist
    routes:
      - path: /
    envs:
      - key: VITE_API_URL
        value: https://api.example.com
```

### Worker (Background Job)
```yaml
name: my-worker
region: nyc
workers:
  - name: processor
    github:
      repo: username/worker
      branch: main
    build_command: pip install -r requirements.txt
    run_command: python worker.py
    instance_size_slug: basic-xs
    instance_count: 1
    envs:
      - key: QUEUE_URL
        scope: RUN_TIME
        type: SECRET
```

### Full Stack App
```yaml
name: fullstack-app
region: nyc

services:
  - name: api
    github:
      repo: username/api
      branch: main
      deploy_on_push: true
    dockerfile_path: Dockerfile
    http_port: 8000
    instance_size_slug: basic-xs
    instance_count: 2
    routes:
      - path: /api
    health_check:
      http_path: /health
      initial_delay_seconds: 10
      period_seconds: 10
    envs:
      - key: DATABASE_URL
        scope: RUN_TIME
        value: ${db.DATABASE_URL}

static_sites:
  - name: frontend
    github:
      repo: username/frontend
      branch: main
    build_command: npm run build
    output_dir: build
    routes:
      - path: /

databases:
  - name: db
    engine: PG
    version: "15"
    size: db-s-dev-database
    num_nodes: 1
```

## Instance Sizes

| Slug | vCPUs | RAM | Price |
|------|-------|-----|-------|
| `basic-xxs` | 1 shared | 256MB | $5/mo |
| `basic-xs` | 1 shared | 512MB | $10/mo |
| `basic-s` | 1 shared | 1GB | $20/mo |
| `basic-m` | 1 shared | 2GB | $40/mo |
| `professional-xs` | 1 | 1GB | $25/mo |
| `professional-s` | 1 | 2GB | $50/mo |
| `professional-m` | 2 | 4GB | $100/mo |

## Environment Variables

### Types
- `GENERAL`: Visible in logs and app spec
- `SECRET`: Hidden, only accessible at runtime

### Scopes
- `BUILD_TIME`: Available during build
- `RUN_TIME`: Available during runtime (default)
- `BUILD_AND_RUN_TIME`: Available during both

### Database Connection
```yaml
envs:
  - key: DATABASE_URL
    scope: RUN_TIME
    value: ${db.DATABASE_URL}
```

## Health Checks

```yaml
health_check:
  http_path: /health
  initial_delay_seconds: 20
  period_seconds: 10
  timeout_seconds: 5
  success_threshold: 2
  failure_threshold: 3
```

## Alerts

```yaml
alerts:
  - rule: DEPLOYMENT_FAILED
  - rule: DOMAIN_FAILED
  - rule: CPU_UTILIZATION
    value: 85
    operator: GREATER_THAN
    window: FIVE_MINUTES
```

## Custom Domains

```bash
# Add domain
doctl apps create-domain <app-id> --domain example.com

# List domains
doctl apps list-domains <app-id>
```

## Best Practices

1. **Use deploy-on-push** - Automate deployments from git
2. **Configure health checks** - Ensure reliable deployments
3. **Use secrets for sensitive data** - Never hardcode credentials
4. **Set appropriate instance sizes** - Start small, scale as needed
5. **Use managed databases** - Easier management and backups
6. **Configure alerts** - Be notified of issues

## Deployment Workflow

1. **Prepare app spec** - Create `app.yaml`
2. **Validate locally** - Review configuration
3. **Create app** - `doctl apps create --spec app.yaml`
4. **Monitor deployment** - `doctl apps logs <app-id> --type deploy --follow`
5. **Verify health** - Check app status and endpoints
6. **Configure domain** - Add custom domain if needed

## Rollback

```bash
# List deployments
doctl apps list-deployments <app-id>

# Get previous deployment ID and redeploy that spec
doctl apps get-deployment <app-id> <previous-deployment-id> --format Spec > previous.yaml
doctl apps update <app-id> --spec previous.yaml
```
