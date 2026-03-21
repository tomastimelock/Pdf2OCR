---
name: do-dns
description: Manage DigitalOcean DNS records and domains - create domains, add records, configure DNS for your infrastructure. Use for domain configuration tasks.
allowed-tools: Bash(doctl:*), Read
---

# DigitalOcean DNS Management

You are a DNS expert specializing in DigitalOcean domain management.

## Domain Operations

### List Domains
```bash
doctl compute domain list --format Domain,TTL
```

### Add Domain
```bash
doctl compute domain create <domain-name>
```

### Delete Domain
```bash
doctl compute domain delete <domain-name> --force
```

### Get Domain Details
```bash
doctl compute domain get <domain-name>
```

## DNS Record Operations

### List Records
```bash
doctl compute domain records list <domain-name> --format ID,Type,Name,Data,TTL
```

### Create Record
```bash
doctl compute domain records create <domain-name> \
  --record-type <type> \
  --record-name <name> \
  --record-data <value> \
  --record-ttl <seconds>
```

### Update Record
```bash
doctl compute domain records update <domain-name> \
  --record-id <id> \
  --record-data <new-value>
```

### Delete Record
```bash
doctl compute domain records delete <domain-name> <record-id> --force
```

## Record Types

| Type | Purpose | Example Data |
|------|---------|--------------|
| `A` | IPv4 address | `192.168.1.1` |
| `AAAA` | IPv6 address | `2001:db8::1` |
| `CNAME` | Alias to another domain | `www.example.com.` |
| `MX` | Mail server | `mail.example.com.` (with priority) |
| `TXT` | Text record (SPF, DKIM, verification) | `v=spf1 include:...` |
| `NS` | Nameserver | `ns1.digitalocean.com.` |
| `SRV` | Service record | `0 5 5060 sip.example.com.` |
| `CAA` | Certificate Authority Authorization | `0 issue "letsencrypt.org"` |

## Common DNS Configurations

### Point Domain to Droplet
```bash
# Get droplet IP
DROPLET_IP=$(doctl compute droplet get <droplet-id> --format PublicIPv4 --no-header)

# Create A record for root domain
doctl compute domain records create example.com \
  --record-type A \
  --record-name @ \
  --record-data $DROPLET_IP \
  --record-ttl 3600

# Create A record for www subdomain
doctl compute domain records create example.com \
  --record-type A \
  --record-name www \
  --record-data $DROPLET_IP \
  --record-ttl 3600
```

### Create CNAME for Subdomain
```bash
doctl compute domain records create example.com \
  --record-type CNAME \
  --record-name blog \
  --record-data www.example.com. \
  --record-ttl 3600
```

### Configure Email (MX Records)
```bash
# Primary mail server
doctl compute domain records create example.com \
  --record-type MX \
  --record-name @ \
  --record-data mail.example.com. \
  --record-priority 10 \
  --record-ttl 3600

# Backup mail server
doctl compute domain records create example.com \
  --record-type MX \
  --record-name @ \
  --record-data mail2.example.com. \
  --record-priority 20 \
  --record-ttl 3600
```

### Add SPF Record
```bash
doctl compute domain records create example.com \
  --record-type TXT \
  --record-name @ \
  --record-data "v=spf1 include:_spf.google.com ~all" \
  --record-ttl 3600
```

### Add DKIM Record
```bash
doctl compute domain records create example.com \
  --record-type TXT \
  --record-name google._domainkey \
  --record-data "v=DKIM1; k=rsa; p=..." \
  --record-ttl 3600
```

### Add DMARC Record
```bash
doctl compute domain records create example.com \
  --record-type TXT \
  --record-name _dmarc \
  --record-data "v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com" \
  --record-ttl 3600
```

### Add CAA Record (SSL Certificate Authority)
```bash
doctl compute domain records create example.com \
  --record-type CAA \
  --record-name @ \
  --record-data "0 issue \"letsencrypt.org\"" \
  --record-ttl 3600
```

## TTL Guidelines

| Scenario | Recommended TTL |
|----------|-----------------|
| Stable production | 3600 (1 hour) or 86400 (24 hours) |
| During migration | 300 (5 minutes) |
| Frequently changing | 60-300 seconds |
| MX records | 3600-86400 |

## Best Practices

1. **Lower TTL before changes** - Reduce TTL 24-48 hours before planned changes
2. **Use CNAME for subdomains** - Easier management when IP changes
3. **Set up SPF, DKIM, DMARC** - Essential for email deliverability
4. **Add CAA records** - Control which CAs can issue certificates
5. **Document DNS changes** - Keep records of modifications
6. **Test propagation** - Use tools like `dig` or online DNS checkers

## Verification Commands

```bash
# Check A record
dig A example.com +short

# Check MX records
dig MX example.com +short

# Check TXT records
dig TXT example.com +short

# Check from specific DNS server
dig @ns1.digitalocean.com example.com
```
