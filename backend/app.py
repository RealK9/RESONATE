"""
RESONATE — FastAPI application setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.health import router as health_router
from routes.analyze import router as analyze_router
from routes.samples import router as samples_router
from routes.track import router as track_router
from routes.similarity import router as similarity_router
from routes.sessions import router as sessions_router
from routes.batch import router as batch_router
from routes.preferences import router as preferences_router
from routes.export import router as export_router
from routes.layering import router as layering_router
from routes.bridge import router as bridge_router
from routes.taste_profile import router as taste_router
from routes.chart_intelligence import router as chart_router
from routes.versions import router as versions_router
from routes.collections import router as collections_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(title="RESONATE")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    application.include_router(health_router)
    application.include_router(analyze_router)
    application.include_router(samples_router)
    application.include_router(track_router)
    application.include_router(similarity_router)
    application.include_router(sessions_router)
    application.include_router(batch_router)
    application.include_router(preferences_router)
    application.include_router(export_router)
    application.include_router(layering_router)
    application.include_router(bridge_router)
    application.include_router(taste_router)
    application.include_router(chart_router)
    application.include_router(versions_router)
    application.include_router(collections_router)

    # Start bridge server for VST3 plugin communication
    from bridge import run_bridge_in_thread
    run_bridge_in_thread(port=9876)

    return application


application = create_app()
