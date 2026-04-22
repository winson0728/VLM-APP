# Command Model

This skeleton treats the agent as a reconciler. The planner emits exact command sequences for a clean host, while the future executor should make them idempotent by checking current state first.

## Routing Mode

Assume:

- `lan_if=br-lan`
- `lan_cidr=192.168.10.0/24`
- `wan_if=enp3s0`
- `wan_gateway=203.0.113.1`
- `route_table=101`
- `fwmark=101`

### Apply

```bash
sysctl -w net.ipv4.ip_forward=1
ip route replace 192.168.10.0/24 dev br-lan table 101
ip route replace default via 203.0.113.1 dev enp3s0 table 101
ip rule add fwmark 101 table 101

ip link add ifbd6af4d5c type ifb
ip link set dev ifbd6af4d5c up
tc qdisc replace dev enp3s0 clsact
tc filter replace dev enp3s0 ingress matchall action mirred egress redirect dev ifbd6af4d5c

tc qdisc replace dev enp3s0 root handle 1: htb default 10
tc class replace dev enp3s0 parent 1: classid 1:10 htb rate 10mbit ceil 10mbit
tc qdisc replace dev enp3s0 parent 1:10 handle 10: netem delay 35ms 10ms distribution normal reorder 0.5% gap 5

tc qdisc replace dev ifbd6af4d5c root handle 2: htb default 20
tc class replace dev ifbd6af4d5c parent 2: classid 2:20 htb rate 40mbit ceil 40mbit
tc qdisc replace dev ifbd6af4d5c parent 2:20 handle 20: netem delay 35ms 10ms distribution normal reorder 0.5% gap 5

nft add table inet netsim
nft add chain inet netsim forward { type filter hook forward priority filter ; policy accept ; }
nft add set inet netsim disabled_pairs { type ifname . ifname ; }
nft add rule inet netsim forward meta iifname . meta oifname @disabled_pairs drop
```

### Disconnect

```bash
nft add element inet netsim disabled_pairs { "br-lan" . "enp3s0", "enp3s0" . "br-lan" }
```

### Reconnect

```bash
nft delete element inet netsim disabled_pairs { "br-lan" . "enp3s0", "enp3s0" . "br-lan" }
```

### Destroy

```bash
tc qdisc del dev enp3s0 root
tc qdisc del dev enp3s0 clsact
tc qdisc del dev ifbd6af4d5c root
ip link del dev ifbd6af4d5c
ip rule del fwmark 101 table 101
ip route flush table 101
nft delete element inet netsim disabled_pairs { "br-lan" . "enp3s0", "enp3s0" . "br-lan" }
```

## Bridge Mode

Assume:

- `port_a=enp4s0`
- `port_b=enp5s0`
- `bridge_name=br-line2`

### Apply

```bash
ip link add name br-line2 type bridge
ip link set dev br-line2 type bridge stp_state 0
ip link set dev enp4s0 master br-line2
ip link set dev enp5s0 master br-line2
ip link set dev enp4s0 up
ip link set dev enp5s0 up
ip link set dev br-line2 up

ip link add ifba9944a76 type ifb
ip link add ifbb9944a76 type ifb
ip link set dev ifba9944a76 up
ip link set dev ifbb9944a76 up

tc qdisc replace dev enp4s0 clsact
tc qdisc replace dev enp5s0 clsact
tc filter replace dev enp4s0 ingress matchall action mirred egress redirect dev ifba9944a76
tc filter replace dev enp5s0 ingress matchall action mirred egress redirect dev ifbb9944a76

tc qdisc replace dev ifba9944a76 root handle 1: htb default 10
tc class replace dev ifba9944a76 parent 1: classid 1:10 htb rate 40mbit ceil 40mbit
tc qdisc replace dev ifba9944a76 parent 1:10 handle 10: netem delay 35ms 10ms distribution normal reorder 0.5% gap 5

tc qdisc replace dev ifbb9944a76 root handle 2: htb default 20
tc class replace dev ifbb9944a76 parent 2: classid 2:20 htb rate 40mbit ceil 40mbit
tc qdisc replace dev ifbb9944a76 parent 2:20 handle 20: netem delay 35ms 10ms distribution normal reorder 0.5% gap 5

nft add table bridge netsim
nft add chain bridge netsim forward { type filter hook forward priority filter ; policy accept ; }
nft add set bridge netsim disabled_pairs { type ifname . ifname ; }
nft add rule bridge netsim forward meta iifname . meta oifname @disabled_pairs drop
```

### Disconnect

```bash
nft add element bridge netsim disabled_pairs { "enp4s0" . "enp5s0", "enp5s0" . "enp4s0" }
```

### Reconnect

```bash
nft delete element bridge netsim disabled_pairs { "enp4s0" . "enp5s0", "enp5s0" . "enp4s0" }
```

### Destroy

```bash
tc qdisc del dev enp4s0 clsact
tc qdisc del dev enp5s0 clsact
tc qdisc del dev ifba9944a76 root
tc qdisc del dev ifbb9944a76 root
ip link del dev ifba9944a76
ip link del dev ifbb9944a76
ip link set dev enp4s0 nomaster
ip link set dev enp5s0 nomaster
ip link del dev br-line2 type bridge
nft delete element bridge netsim disabled_pairs { "enp4s0" . "enp5s0", "enp5s0" . "enp4s0" }
```

## Why The Model Looks Like This

- `htb` handles bandwidth caps cleanly.
- `netem` adds delay, jitter, and reorder on top of the shaped class.
- IFB is required for reliable ingress shaping.
- `nftables` is used for temporary disconnects because dropping at the filter layer is cleaner than bringing interfaces down.
