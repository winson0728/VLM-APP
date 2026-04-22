# API Schema

Base path: `/api/v1`

## Endpoints

- `GET /healthz`
- `GET /interfaces`
- `GET /profiles`
- `GET /lines`
- `POST /lines`
- `GET /lines/{line_id}`
- `PATCH /lines/{line_id}`
- `POST /lines/{line_id}/start`
- `POST /lines/{line_id}/stop`
- `POST /lines/{line_id}/apply-profile/{profile_name}`
- `GET /lines/{line_id}/plan`

## Core Types

### LineSpec

```json
{
  "id": "line1",
  "description": "WAN 1 via enp3s0",
  "mode": "routing",
  "profile": "4g",
  "enabled": true,
  "routing": {
    "lan_if": "br-lan",
    "lan_cidr": "192.168.10.0/24",
    "wan_if": "enp3s0",
    "wan_gateway": "203.0.113.1",
    "route_table": 101,
    "fwmark": 101
  },
  "bridge": null,
  "impairments": {
    "up_mbps": { "min": 5, "max": 20, "base": 10 },
    "down_mbps": { "min": 20, "max": 80, "base": 40 },
    "delay_ms": { "min": 25, "max": 60, "base": 35 },
    "jitter_ms": { "min": 5, "max": 20, "base": 10 },
    "reorder_pct": { "min": 0, "max": 2, "base": 0.5 },
    "reorder_gap": 5,
    "disconnect": {
      "enabled": true,
      "probability_per_hour": 2,
      "duration_sec_min": 3,
      "duration_sec_max": 15,
      "method": "nft_drop"
    },
    "randomization": {
      "enabled": true,
      "update_interval_sec": 5,
      "distribution": "uniform",
      "hysteresis_pct": 10
    }
  }
}
```

### RoutingBinding

Use this when a line forwards between LAN and WAN.

```json
{
  "lan_if": "br-lan",
  "lan_cidr": "192.168.10.0/24",
  "wan_if": "enp3s0",
  "wan_gateway": "203.0.113.1",
  "route_table": 101,
  "fwmark": 101
}
```

### BridgeBinding

Use this when a line is a transparent bridge between two NICs.

```json
{
  "port_a": "enp4s0",
  "port_b": "enp5s0",
  "bridge_name": "br-line2",
  "stp": false
}
```

### Plan Response

`GET /lines/{line_id}/plan` returns a full plan bundle:

- `apply`
- `disconnect`
- `reconnect`
- `destroy`

Each command step contains:

- `tool`
- `argv`
- `shell`
- `rationale`

## API Design Notes

- `POST /lines` creates desired state only.
- `POST /lines/{line_id}/start` marks the line enabled and lets the future agent reconcile it.
- `GET /lines/{line_id}/plan` exposes the exact Linux command model before building the executor.
- Profile application is merge-by-default: a profile replaces the impairment block, and later user edits can override it.
