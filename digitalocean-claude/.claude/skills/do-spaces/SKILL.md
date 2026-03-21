---
name: do-spaces
description: Manage DigitalOcean Spaces object storage - create buckets, upload files, configure CDN and access policies. Use for S3-compatible storage tasks.
allowed-tools: Bash(doctl:*), Bash(s3cmd:*), Bash(aws:*), Read, Write
---

# DigitalOcean Spaces Management

You are an object storage expert specializing in DigitalOcean Spaces.

## Spaces Overview

DigitalOcean Spaces is S3-compatible object storage. You can use:
- `doctl` for basic operations
- `s3cmd` for advanced S3 operations
- `aws` CLI with custom endpoint

## doctl Spaces Commands

### List Spaces
```bash
doctl compute cdn list
```

### Create Space (via API)
Note: Space creation is typically done via the DigitalOcean control panel or API, not doctl.

## s3cmd Configuration

### Configure s3cmd
Create `~/.s3cfg`:
```ini
[default]
access_key = YOUR_SPACES_ACCESS_KEY
secret_key = YOUR_SPACES_SECRET_KEY
host_base = nyc3.digitaloceanspaces.com
host_bucket = %(bucket)s.nyc3.digitaloceanspaces.com
```

### List Buckets
```bash
s3cmd ls
```

### List Objects in Space
```bash
s3cmd ls s3://my-space/
s3cmd ls s3://my-space/folder/
```

### Upload Files
```bash
# Single file
s3cmd put file.txt s3://my-space/

# Directory (recursive)
s3cmd put --recursive ./folder/ s3://my-space/folder/

# With public access
s3cmd put --acl-public file.txt s3://my-space/
```

### Download Files
```bash
# Single file
s3cmd get s3://my-space/file.txt ./

# Directory (recursive)
s3cmd get --recursive s3://my-space/folder/ ./folder/
```

### Delete Files
```bash
# Single file
s3cmd del s3://my-space/file.txt

# Delete all files in prefix
s3cmd del --recursive s3://my-space/folder/
```

### Sync Directories
```bash
# Sync local to Space
s3cmd sync ./local-folder/ s3://my-space/folder/

# Sync Space to local
s3cmd sync s3://my-space/folder/ ./local-folder/

# Sync with delete (mirror)
s3cmd sync --delete-removed ./local-folder/ s3://my-space/folder/
```

## AWS CLI Configuration

### Configure AWS CLI for Spaces
```bash
aws configure --profile digitalocean
# Access Key: YOUR_SPACES_ACCESS_KEY
# Secret Key: YOUR_SPACES_SECRET_KEY
# Region: nyc3
# Output: json
```

### AWS CLI Commands
```bash
# List buckets
aws s3 ls --endpoint-url https://nyc3.digitaloceanspaces.com --profile digitalocean

# List objects
aws s3 ls s3://my-space/ --endpoint-url https://nyc3.digitaloceanspaces.com --profile digitalocean

# Upload
aws s3 cp file.txt s3://my-space/ --endpoint-url https://nyc3.digitaloceanspaces.com --profile digitalocean

# Sync
aws s3 sync ./folder s3://my-space/folder --endpoint-url https://nyc3.digitaloceanspaces.com --profile digitalocean
```

## Spaces Regions

| Region | Endpoint |
|--------|----------|
| NYC3 | `nyc3.digitaloceanspaces.com` |
| SFO3 | `sfo3.digitaloceanspaces.com` |
| AMS3 | `ams3.digitaloceanspaces.com` |
| SGP1 | `sgp1.digitaloceanspaces.com` |
| FRA1 | `fra1.digitaloceanspaces.com` |

## CDN Configuration

### Enable CDN (via doctl)
```bash
doctl compute cdn create \
  --origin my-space.nyc3.digitaloceanspaces.com \
  --ttl 3600
```

### List CDN Endpoints
```bash
doctl compute cdn list
```

### Flush CDN Cache
```bash
doctl compute cdn flush <cdn-id> --files "/*"
```

## Access Control

### Set Object ACL
```bash
# Make public
s3cmd setacl --acl-public s3://my-space/file.txt

# Make private
s3cmd setacl --acl-private s3://my-space/file.txt
```

### Bucket Policy Example
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-space/public/*"
    }
  ]
}
```

## CORS Configuration

Create `cors.json`:
```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["https://example.com"],
      "AllowedMethods": ["GET", "PUT", "POST"],
      "AllowedHeaders": ["*"],
      "MaxAgeSeconds": 3600
    }
  ]
}
```

Apply CORS:
```bash
aws s3api put-bucket-cors \
  --bucket my-space \
  --cors-configuration file://cors.json \
  --endpoint-url https://nyc3.digitaloceanspaces.com \
  --profile digitalocean
```

## Best Practices

1. **Use CDN for public content** - Reduces latency and bandwidth costs
2. **Set appropriate ACLs** - Default to private, only make public what's needed
3. **Use lifecycle policies** - Automatically delete old files
4. **Enable versioning** - For important data
5. **Use server-side encryption** - For sensitive data
6. **Organize with prefixes** - Use folder-like structures

## Common Use Cases

### Static Website Hosting
```bash
# Upload website files
s3cmd sync --acl-public ./dist/ s3://my-website/

# Enable as static website (via control panel or API)
```

### Backup Storage
```bash
# Daily backup script
DATE=$(date +%Y-%m-%d)
tar -czf backup-$DATE.tar.gz /data
s3cmd put backup-$DATE.tar.gz s3://my-backups/daily/
```

### Application Assets
```bash
# Upload and get CDN URL
s3cmd put --acl-public image.png s3://my-assets/images/
# Access via: https://my-assets.nyc3.cdn.digitaloceanspaces.com/images/image.png
```
