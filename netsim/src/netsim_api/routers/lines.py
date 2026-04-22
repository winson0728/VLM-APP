from fastapi import APIRouter, HTTPException, Request

from netsim_agent.service import AgentService
from netsim_common.models import ActionResponse, LinePatchRequest, LinePlan, LineSpec


router = APIRouter(tags=["lines"])
agent_service = AgentService()


def _load_line_or_404(request: Request, line_id: str) -> LineSpec:
    try:
        return request.app.state.store.get_line(line_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown line '{line_id}'") from exc


@router.get("/lines", response_model=list[LineSpec])
def list_lines(request: Request) -> list[LineSpec]:
    return request.app.state.store.list_lines()


@router.post("/lines", response_model=LineSpec, status_code=201)
def create_line(line: LineSpec, request: Request) -> LineSpec:
    return request.app.state.store.create_line(line)


@router.get("/lines/{line_id}", response_model=LineSpec)
def get_line(line_id: str, request: Request) -> LineSpec:
    return _load_line_or_404(request, line_id)


@router.patch("/lines/{line_id}", response_model=LineSpec)
def patch_line(line_id: str, patch: LinePatchRequest, request: Request) -> LineSpec:
    _load_line_or_404(request, line_id)
    return request.app.state.store.patch_line(line_id, patch)


@router.post("/lines/{line_id}/start", response_model=ActionResponse)
def start_line(line_id: str, request: Request) -> ActionResponse:
    _load_line_or_404(request, line_id)
    line = request.app.state.store.set_enabled(line_id, True)
    return ActionResponse(status="enabled", line=line)


@router.post("/lines/{line_id}/stop", response_model=ActionResponse)
def stop_line(line_id: str, request: Request) -> ActionResponse:
    _load_line_or_404(request, line_id)
    line = request.app.state.store.set_enabled(line_id, False)
    return ActionResponse(status="disabled", line=line)


@router.post("/lines/{line_id}/apply-profile/{profile_name}", response_model=ActionResponse)
def apply_profile(line_id: str, profile_name: str, request: Request) -> ActionResponse:
    _load_line_or_404(request, line_id)
    try:
        line = request.app.state.store.apply_profile(line_id, profile_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown profile '{profile_name}'") from exc
    return ActionResponse(status=f"profile:{profile_name}", line=line)


@router.get("/lines/{line_id}/plan", response_model=LinePlan)
def get_line_plan(line_id: str, request: Request) -> LinePlan:
    line = _load_line_or_404(request, line_id)
    return agent_service.build_plan(line)
