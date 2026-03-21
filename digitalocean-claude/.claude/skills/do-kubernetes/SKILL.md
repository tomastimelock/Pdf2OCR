---
name: do-kubernetes
description: Manage DigitalOcean Kubernetes (DOKS) clusters - create, scale, upgrade, and configure Kubernetes infrastructure. Use for container orchestration tasks.
allowed-tools: Bash(doctl:*), Bash(kubectl:*), Read, Write
---

# DigitalOcean Kubernetes Management

You are a Kubernetes expert specializing in DigitalOcean Kubernetes Service (DOKS).

## Cluster Operations

### List Clusters
```bash
doctl kubernetes cluster list --format ID,Name,Region,Version,Status,NodePools
```

### Get Cluster Details
```bash
doctl kubernetes cluster get <cluster-id>
```

### Create Cluster
```bash
doctl kubernetes cluster create <name> \
  --region <region> \
  --version <k8s-version> \
  --node-pool "name=default;size=s-2vcpu-4gb;count=3;auto-scale=true;min-nodes=2;max-nodes=5" \
  --vpc-uuid <vpc-id> \
  --wait
```

### Delete Cluster
```bash
doctl kubernetes cluster delete <cluster-id> --force
```

### Get Kubeconfig
```bash
doctl kubernetes cluster kubeconfig save <cluster-id>
```

### List Available Versions
```bash
doctl kubernetes options versions
```

## Node Pool Management

### List Node Pools
```bash
doctl kubernetes cluster node-pool list <cluster-id>
```

### Create Node Pool
```bash
doctl kubernetes cluster node-pool create <cluster-id> \
  --name <pool-name> \
  --size <size> \
  --count <node-count> \
  --auto-scale \
  --min-nodes <min> \
  --max-nodes <max>
```

### Update Node Pool (Scale)
```bash
doctl kubernetes cluster node-pool update <cluster-id> <pool-id> --count <new-count>
```

### Delete Node Pool
```bash
doctl kubernetes cluster node-pool delete <cluster-id> <pool-id> --force
```

### Recycle Nodes
```bash
doctl kubernetes cluster node-pool recycle <cluster-id> <pool-id> --node-ids <node-id>
```

## Cluster Upgrades

### Check Available Upgrades
```bash
doctl kubernetes cluster get-upgrades <cluster-id>
```

### Upgrade Cluster
```bash
doctl kubernetes cluster upgrade <cluster-id> --version <new-version>
```

## Common Node Sizes

| Slug | vCPUs | RAM | Best For |
|------|-------|-----|----------|
| `s-1vcpu-2gb` | 1 | 2GB | Development |
| `s-2vcpu-4gb` | 2 | 4GB | Small workloads |
| `s-4vcpu-8gb` | 4 | 8GB | Standard workloads |
| `s-8vcpu-16gb` | 8 | 16GB | Production |
| `g-2vcpu-8gb` | 2 | 8GB | Memory-intensive |
| `c-4` | 4 | 8GB | CPU-intensive |

## kubectl Commands After Connection

### Verify Connection
```bash
kubectl cluster-info
kubectl get nodes
```

### Check Workloads
```bash
kubectl get pods --all-namespaces
kubectl get deployments --all-namespaces
kubectl get services --all-namespaces
```

### View Resources
```bash
kubectl top nodes
kubectl top pods --all-namespaces
```

## Best Practices

1. **Use auto-scaling** - Enable auto-scale for production clusters
2. **Multiple node pools** - Separate pools for different workload types
3. **Private cluster** - Use VPC for enhanced security
4. **Regular upgrades** - Keep Kubernetes version current
5. **Resource limits** - Always set resource requests and limits
6. **Managed databases** - Use DO managed databases instead of running in cluster

## Example: Create Production Cluster

```bash
# Create a production-ready cluster
doctl kubernetes cluster create myapp-prod \
  --region nyc3 \
  --version latest \
  --node-pool "name=general;size=s-4vcpu-8gb;count=3;auto-scale=true;min-nodes=3;max-nodes=10;label=workload=general" \
  --node-pool "name=workers;size=s-8vcpu-16gb;count=2;auto-scale=true;min-nodes=1;max-nodes=5;label=workload=workers" \
  --wait

# Save kubeconfig
doctl kubernetes cluster kubeconfig save myapp-prod

# Verify
kubectl get nodes
```

## Container Registry

### Create Registry
```bash
doctl registry create <name> --subscription-tier basic
```

### Login to Registry
```bash
doctl registry login
```

### Configure Cluster for Registry
```bash
doctl kubernetes cluster registry add <cluster-id>
```
