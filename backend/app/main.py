from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import health, generation, carter
from .core.config import get_settings
from .storage.paths import prepare_storage

settings = get_settings()
prepare_storage(settings)
app = FastAPI(title="Document to Dataset API")
allowed_origins = list(dict.fromkeys([settings.frontend_origin, "http://127.0.0.1:5173", "http://localhost:5173"]))
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["*"])

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response
app.include_router(health.router, prefix="/api")
app.include_router(generation.router, prefix="/api")
app.include_router(carter.router, prefix="/api")
