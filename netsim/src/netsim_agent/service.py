from netsim_agent.planner import CommandPlanner
from netsim_common.models import LinePlan, LineSpec


class AgentService:
    def __init__(self) -> None:
        self._planner = CommandPlanner()

    def build_plan(self, line: LineSpec) -> LinePlan:
        return self._planner.build_plan(line)
