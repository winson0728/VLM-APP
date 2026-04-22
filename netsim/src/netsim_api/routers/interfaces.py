from fastapi import APIRouter, Request

from netsim_common.models import InterfaceSummary


router = APIRouter(tags=["interfaces"])


@router.get("/interfaces", response_model=list[InterfaceSummary])
def list_interfaces(request: Request) -> list[InterfaceSummary]:
    return request.app.state.store.list_interfaces()
