---
name: do-droplets
description: Manage DigitalOcean droplets - create, list, resize, delete, and configure virtual machines. Use when working with DO compute infrastructure.
allowed-tools: Bash(doctl:*), Read, Write
---

# DigitalOcean Droplet Management

You are a DigitalOcean infrastructure expert specializing in droplet management.

## Quick Reference Commands

### List Droplets
```bash
doctl compute droplet list --format ID,Name,PublicIPv4,Region,Size,Status,Tags
```

### Get Droplet Details
```bash
doctl compute droplet get <droplet-id> --format ID,Name,PublicIPv4,PrivateIPv4,Region,Size,Status,VpcUUID
```

### Create Droplet
```bash
doctl compute droplet create <name> \
  --region <region> \
  --size <size> \
  --image <image> \
  --ssh-keys <key-id> \
  --tag-name <tag> \
  --enable-monitoring \
  --wait
```

### Delete Droplet
```bash
doctl compute droplet delete <droplet-id> --force
```

### Resize Droplet
```bash
# Power off first for CPU/RAM changes
doctl compute droplet-action power-off <droplet-id> --wait
doctl compute droplet-action resize <droplet-id> --size <new-size> --wait
doctl compute droplet-action power-on <droplet-id> --wait
```

### Droplet Actions
```bash
doctl compute droplet-action reboot <droplet-id> --wait
doctl compute droplet-action power-cycle <droplet-id> --wait
doctl compute droplet-action snapshot <droplet-id> --snapshot-name <name> --wait
```

## Common Sizes

| Slug | vCPUs | RAM | Price |
|------|-------|-----|-------|
| `s-1vcpu-512mb-10gb` | 1 | 512MB | $4/mo |
| `s-1vcpu-1gb` | 1 | 1GB | $6/mo |
| `s-1vcpu-2gb` | 1 | 2GB | $12/mo |
| `s-2vcpu-2gb` | 2 | 2GB | $18/mo |
| `s-2vcpu-4gb` | 2 | 4GB | $24/mo |
| `s-4vcpu-8gb` | 4 | 8GB | $48/mo |
| `s-8vcpu-16gb` | 8 | 16GB | $96/mo |

## Common Images

| Slug | Description |
|------|-------------|
| `ubuntu-24-04-x64` | Ubuntu 24.04 LTS |
| `ubuntu-22-04-x64` | Ubuntu 22.04 LTS |
| `debian-12-x64` | Debian 12 |
| `rocky-9-x64` | Rocky Linux 9 |
| `almalinux-9-x64` | AlmaLinux 9 |
| `fedora-40-x64` | Fedora 40 |

## Regions

| Slug | Location |
|------|----------|
| `nyc1`, `nyc3` | New York |
| `sfo2`, `sfo3` | San Francisco |
| `ams3` | Amsterdam |
| `fra1` | Frankfurt |
| `lon1` | London |
| `sgp1` | Singapore |
| `blr1` | Bangalore |
| `tor1` | Toronto |
| `syd1` | Sydney |

## Helper Commands

### List SSH Keys
```bash
doctl compute ssh-key list --format ID,Name,FingerPrint
```

### List Available Sizes
```bash
doctl compute size list --format Slug,Memory,VCPUs,Disk,PriceMonthly
```

### List Available Images
```bash
doctl compute image list --public --format ID,Slug,Name,Type
```

### List Regions
```bash
doctl compute region list --format Slug,Name,Available
```

## Best Practices

1. **Always use SSH keys** - Never use password authentication
2. **Enable monitoring** - Add `--enable-monitoring` flag
3. **Use VPC** - Add `--vpc-uuid` for private networking
4. **Apply tags** - Use `--tag-name` for organization
5. **Enable backups** - Add `--enable-backups` for production
6. **Wait for completion** - Use `--wait` flag for synchronous operations

## Example: Create Production Web Server

```bash
# Get SSH key ID
SSH_KEY=$(doctl compute ssh-key list --format ID --no-header | head -1)

# Create droplet
doctl compute droplet create webapp-prod-01 \
  --region nyc3 \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --ssh-keys $SSH_KEY \
  --tag-name production \
  --tag-name webapp \
  --enable-monitoring \
  --enable-backups \
  --wait
```
