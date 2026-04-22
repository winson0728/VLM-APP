from fastapi import APIRouter, Request

from netsim_common.models import ProfileSpec


router = APIRouter(tags=["profiles"])


@router.get("/profiles", response_model=list[ProfileSpec])
def list_profiles(request: Request) -> list[ProfileSpec]:
    return request.app.state.store.list_profiles()
