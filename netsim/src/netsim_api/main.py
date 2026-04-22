from fastapi import FastAPI

from netsim_api.routers.interfaces import router as interfaces_router
from netsim_api.routers.lines import router as lines_router
from netsim_api.routers.profiles import router as profiles_router
from netsim_api.state import InMemoryStore


app = FastAPI(title="netsim-api", version="0.1.0")
app.state.store = InMemoryStore()

app.include_router(interfaces_router, prefix="/api/v1")
app.include_router(profiles_router, prefix="/api/v1")
app.include_router(lines_router, prefix="/api/v1")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
