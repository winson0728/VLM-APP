from __future__ import annotations

import hashlib
import shlex

from netsim_common.models import CommandPhase, CommandStep, LineMode, LinePlan, LineSpec


class CommandPlanner:
    def build_plan(self, line: LineSpec) -> LinePlan:
        if line.mode == LineMode.ROUTING:
            apply = self._build_routing_apply(line)
            disconnect = self._build_routing_disconnect(line)
            reconnect = self._build_routing_reconnect(line)
            destroy = self._build_routing_destroy(line)
            notes = [
                "Routing mode shapes WAN egress directly and WAN ingress through IFB.",
                "The future executor should treat nft table, chain, and set creation as create-if-missing operations.",
            ]
        else:
            apply = self._build_bridge_apply(line)
            disconnect = self._build_bridge_disconnect(line)
            reconnect = self._build_bridge_reconnect(line)
            destroy = self._build_bridge_destroy(line)
            notes = [
                "Bridge mode shapes both directions through per-port IFB devices.",
                "The bridge nft table uses the bridge family so disconnects happen at L2 forwarding time.",
            ]

        return LinePlan(
            line_id=line.id,
            mode=line.mode,
            apply=CommandPhase(name="apply", commands=apply),
            disconnect=CommandPhase(name="disconnect", commands=disconnect),
            reconnect=CommandPhase(name="reconnect", commands=reconnect),
            destroy=CommandPhase(name="destroy", commands=destroy),
            notes=notes,
        )

    def _build_routing_apply(self, line: LineSpec) -> list[CommandStep]:
        assert line.routing is not None
        route = line.routing
        impair = line.impairments
        ifb_down = self._ifb_name(line.id, "d")

        commands = [
            self._cmd("sysctl", "-w", "net.ipv4.ip_forward=1", rationale="Enable IPv4 forwarding on the host for routed lines."),
            self._cmd("ip", "route", "replace", route.lan_cidr, "dev", route.lan_if, "table", str(route.route_table), rationale="Keep the LAN subnet reachable inside the per-line route table."),
            self._cmd("ip", "route", "replace", "default", "via", route.wan_gateway, "dev", route.wan_if, "table", str(route.route_table), rationale="Send the line's default route through its WAN interface."),
        ]
        if route.fwmark is not None:
            commands.append(self._cmd("ip", "rule", "add", "fwmark", str(route.fwmark), "table", str(route.route_table), rationale="Bind marked traffic to the per-line routing table."))

        commands.extend(
            [
                self._cmd("ip", "link", "add", ifb_down, "type", "ifb", rationale="Create the IFB used to shape WAN ingress."),
                self._cmd("ip", "link", "set", "dev", ifb_down, "up", rationale="Bring the IFB device up before redirecting ingress traffic."),
                self._cmd("tc", "qdisc", "replace", "dev", route.wan_if, "clsact", rationale="Attach a clsact qdisc so ingress traffic can be mirrored to IFB."),
                self._cmd("tc", "filter", "replace", "dev", route.wan_if, "ingress", "matchall", "action", "mirred", "egress", "redirect", "dev", ifb_down, rationale="Redirect WAN ingress to IFB for downstream shaping."),
                self._cmd("tc", "qdisc", "replace", "dev", route.wan_if, "root", "handle", "1:", "htb", "default", "10", rationale="Create the WAN egress HTB root qdisc."),
                self._cmd("tc", "class", "replace", "dev", route.wan_if, "parent", "1:", "classid", "1:10", "htb", "rate", self._mbit(impair.up_mbps.resolved()), "ceil", self._mbit(impair.up_mbps.resolved()), rationale="Apply uplink bandwidth shaping on WAN egress."),
                self._cmd("tc", "qdisc", "replace", "dev", route.wan_if, "parent", "1:10", "handle", "10:", "netem", "delay", self._ms(impair.delay_ms.resolved()), self._ms(impair.jitter_ms.resolved()), "distribution", "normal", "reorder", self._pct(impair.reorder_pct.resolved()), "gap", str(impair.reorder_gap), rationale="Apply delay, jitter, and reorder on routed uplink traffic."),
                self._cmd("tc", "qdisc", "replace", "dev", ifb_down, "root", "handle", "2:", "htb", "default", "20", rationale="Create the downstream HTB root on the IFB device."),
                self._cmd("tc", "class", "replace", "dev", ifb_down, "parent", "2:", "classid", "2:20", "htb", "rate", self._mbit(impair.down_mbps.resolved()), "ceil", self._mbit(impair.down_mbps.resolved()), rationale="Apply downlink bandwidth shaping through IFB."),
                self._cmd("tc", "qdisc", "replace", "dev", ifb_down, "parent", "2:20", "handle", "20:", "netem", "delay", self._ms(impair.delay_ms.resolved()), self._ms(impair.jitter_ms.resolved()), "distribution", "normal", "reorder", self._pct(impair.reorder_pct.resolved()), "gap", str(impair.reorder_gap), rationale="Apply delay, jitter, and reorder on routed downlink traffic."),
            ]
        )
        commands.extend(self._ensure_nft_pair_blocker("inet"))
        return commands

    def _build_routing_disconnect(self, line: LineSpec) -> list[CommandStep]:
        assert line.routing is not None
        return [self._pair_element_command("nft", "add", "inet", line.routing.lan_if, line.routing.wan_if, "Block both forward directions for a routed line without touching carrier state.")]

    def _build_routing_reconnect(self, line: LineSpec) -> list[CommandStep]:
        assert line.routing is not None
        return [self._pair_element_command("nft", "delete", "inet", line.routing.lan_if, line.routing.wan_if, "Re-enable forwarding for the routed line.")]

    def _build_routing_destroy(self, line: LineSpec) -> list[CommandStep]:
        assert line.routing is not None
        route = line.routing
        ifb_down = self._ifb_name(line.id, "d")

        commands = [
            self._cmd("tc", "qdisc", "del", "dev", route.wan_if, "root", rationale="Remove routed uplink shaping."),
            self._cmd("tc", "qdisc", "del", "dev", route.wan_if, "clsact", rationale="Remove WAN ingress redirection state."),
            self._cmd("tc", "qdisc", "del", "dev", ifb_down, "root", rationale="Remove IFB shaping state."),
            self._cmd("ip", "link", "del", "dev", ifb_down, rationale="Delete the routed IFB device."),
        ]
        if route.fwmark is not None:
            commands.append(self._cmd("ip", "rule", "del", "fwmark", str(route.fwmark), "table", str(route.route_table), rationale="Delete the policy rule for the routed line."))
        commands.extend([
            self._cmd("ip", "route", "flush", "table", str(route.route_table), rationale="Clear the per-line route table."),
            self._build_routing_reconnect(line)[0],
        ])
        return commands

    def _build_bridge_apply(self, line: LineSpec) -> list[CommandStep]:
        assert line.bridge is not None
        bridge = line.bridge
        impair = line.impairments
        bridge_name = bridge.bridge_name or self._bridge_name(line.id)
        ifb_a = self._ifb_name(line.id, "a")
        ifb_b = self._ifb_name(line.id, "b")

        commands = [
            self._cmd("ip", "link", "add", "name", bridge_name, "type", "bridge", rationale="Create the bridge used by transparent mode."),
            self._cmd("ip", "link", "set", "dev", bridge_name, "type", "bridge", "stp_state", "1" if bridge.stp else "0", rationale="Set bridge spanning tree state."),
            self._cmd("ip", "link", "set", "dev", bridge.port_a, "master", bridge_name, rationale="Attach port A to the line bridge."),
            self._cmd("ip", "link", "set", "dev", bridge.port_b, "master", bridge_name, rationale="Attach port B to the line bridge."),
            self._cmd("ip", "link", "set", "dev", bridge.port_a, "up", rationale="Bring bridge port A up."),
            self._cmd("ip", "link", "set", "dev", bridge.port_b, "up", rationale="Bring bridge port B up."),
            self._cmd("ip", "link", "set", "dev", bridge_name, "up", rationale="Bring the line bridge up."),
            self._cmd("ip", "link", "add", ifb_a, "type", "ifb", rationale="Create IFB for traffic arriving from port A."),
            self._cmd("ip", "link", "add", ifb_b, "type", "ifb", rationale="Create IFB for traffic arriving from port B."),
            self._cmd("ip", "link", "set", "dev", ifb_a, "up", rationale="Bring IFB A up."),
            self._cmd("ip", "link", "set", "dev", ifb_b, "up", rationale="Bring IFB B up."),
            self._cmd("tc", "qdisc", "replace", "dev", bridge.port_a, "clsact", rationale="Attach clsact to bridge port A."),
            self._cmd("tc", "qdisc", "replace", "dev", bridge.port_b, "clsact", rationale="Attach clsact to bridge port B."),
            self._cmd("tc", "filter", "replace", "dev", bridge.port_a, "ingress", "matchall", "action", "mirred", "egress", "redirect", "dev", ifb_a, rationale="Redirect traffic entering port A into IFB A for shaping."),
            self._cmd("tc", "filter", "replace", "dev", bridge.port_b, "ingress", "matchall", "action", "mirred", "egress", "redirect", "dev", ifb_b, rationale="Redirect traffic entering port B into IFB B for shaping."),
            self._cmd("tc", "qdisc", "replace", "dev", ifb_a, "root", "handle", "1:", "htb", "default", "10", rationale="Create shaping root for traffic entering from port A."),
            self._cmd("tc", "class", "replace", "dev", ifb_a, "parent", "1:", "classid", "1:10", "htb", "rate", self._mbit(impair.down_mbps.resolved()), "ceil", self._mbit(impair.down_mbps.resolved()), rationale="Apply bridge-mode shaping for A-to-B traffic."),
            self._cmd("tc", "qdisc", "replace", "dev", ifb_a, "parent", "1:10", "handle", "10:", "netem", "delay", self._ms(impair.delay_ms.resolved()), self._ms(impair.jitter_ms.resolved()), "distribution", "normal", "reorder", self._pct(impair.reorder_pct.resolved()), "gap", str(impair.reorder_gap), rationale="Apply delay, jitter, and reorder for A-to-B traffic."),
            self._cmd("tc", "qdisc", "replace", "dev", ifb_b, "root", "handle", "2:", "htb", "default", "20", rationale="Create shaping root for traffic entering from port B."),
            self._cmd("tc", "class", "replace", "dev", ifb_b, "parent", "2:", "classid", "2:20", "htb", "rate", self._mbit(impair.up_mbps.resolved()), "ceil", self._mbit(impair.up_mbps.resolved()), rationale="Apply bridge-mode shaping for B-to-A traffic."),
            self._cmd("tc", "qdisc", "replace", "dev", ifb_b, "parent", "2:20", "handle", "20:", "netem", "delay", self._ms(impair.delay_ms.resolved()), self._ms(impair.jitter_ms.resolved()), "distribution", "normal", "reorder", self._pct(impair.reorder_pct.resolved()), "gap", str(impair.reorder_gap), rationale="Apply delay, jitter, and reorder for B-to-A traffic."),
        ]
        commands.extend(self._ensure_nft_pair_blocker("bridge"))
        return commands

    def _build_bridge_disconnect(self, line: LineSpec) -> list[CommandStep]:
        assert line.bridge is not None
        return [self._pair_element_command("nft", "add", "bridge", line.bridge.port_a, line.bridge.port_b, "Block both bridge forwarding directions for the line.")]

    def _build_bridge_reconnect(self, line: LineSpec) -> list[CommandStep]:
        assert line.bridge is not None
        return [self._pair_element_command("nft", "delete", "bridge", line.bridge.port_a, line.bridge.port_b, "Re-enable bridge forwarding for the line.")]

    def _build_bridge_destroy(self, line: LineSpec) -> list[CommandStep]:
        assert line.bridge is not None
        bridge = line.bridge
        bridge_name = bridge.bridge_name or self._bridge_name(line.id)
        ifb_a = self._ifb_name(line.id, "a")
        ifb_b = self._ifb_name(line.id, "b")

        return [
            self._cmd("tc", "qdisc", "del", "dev", bridge.port_a, "clsact", rationale="Remove ingress redirection from bridge port A."),
            self._cmd("tc", "qdisc", "del", "dev", bridge.port_b, "clsact", rationale="Remove ingress redirection from bridge port B."),
            self._cmd("tc", "qdisc", "del", "dev", ifb_a, "root", rationale="Remove shaping state from IFB A."),
            self._cmd("tc", "qdisc", "del", "dev", ifb_b, "root", rationale="Remove shaping state from IFB B."),
            self._cmd("ip", "link", "del", "dev", ifb_a, rationale="Delete IFB A."),
            self._cmd("ip", "link", "del", "dev", ifb_b, rationale="Delete IFB B."),
            self._cmd("ip", "link", "set", "dev", bridge.port_a, "nomaster", rationale="Detach port A from the bridge."),
            self._cmd("ip", "link", "set", "dev", bridge.port_b, "nomaster", rationale="Detach port B from the bridge."),
            self._cmd("ip", "link", "del", "dev", bridge_name, "type", "bridge", rationale="Delete the bridge device for the line."),
            self._build_bridge_reconnect(line)[0],
        ]

    def _ensure_nft_pair_blocker(self, family: str) -> list[CommandStep]:
        return [
            self._cmd("nft", "add", "table", family, "netsim", rationale=f"Create the shared {family} table for disconnect control."),
            self._cmd("nft", "add", "chain", family, "netsim", "forward", "{", "type", "filter", "hook", "forward", "priority", "filter", ";", "policy", "accept", ";", "}", rationale="Create the forward hook chain used by disconnect rules."),
            self._cmd("nft", "add", "set", family, "netsim", "disabled_pairs", "{", "type", "ifname", ".", "ifname", ";", "}", rationale="Create the set that stores blocked interface pairs."),
            self._cmd("nft", "add", "rule", family, "netsim", "forward", "meta", "iifname", ".", "meta", "oifname", "@disabled_pairs", "drop", rationale="Drop traffic when the ingress and egress pair is marked disconnected."),
        ]

    def _pair_element_command(self, tool: str, verb: str, family: str, left: str, right: str, rationale: str) -> CommandStep:
        return self._cmd(tool, verb, "element", family, "netsim", "disabled_pairs", "{", left, ".", right, ",", right, ".", left, "}", rationale=rationale)

    def _cmd(self, tool: str, *argv: str, rationale: str) -> CommandStep:
        args = list(argv)
        shell = shlex.join([tool, *args])
        return CommandStep(tool=tool, argv=args, shell=shell, rationale=rationale)

    def _mbit(self, value: float) -> str:
        rounded = int(value) if float(value).is_integer() else round(value, 2)
        return f"{rounded}mbit"

    def _ms(self, value: float) -> str:
        rounded = int(value) if float(value).is_integer() else round(value, 2)
        return f"{rounded}ms"

    def _pct(self, value: float) -> str:
        rounded = int(value) if float(value).is_integer() else round(value, 2)
        return f"{rounded}%"

    def _ifb_name(self, line_id: str, suffix: str) -> str:
        digest = hashlib.sha1(line_id.encode("utf-8")).hexdigest()[:8]
        return f"ifb{suffix}{digest[:7]}"[:15]

    def _bridge_name(self, line_id: str) -> str:
        digest = hashlib.sha1(line_id.encode("utf-8")).hexdigest()[:8]
        return f"br{digest}"[:15]
