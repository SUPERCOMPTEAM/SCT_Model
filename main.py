import os
import logging

from fastapi import FastAPI

from api import api_router
from config import get_app_settings
from services.balancer import balancer

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_application() -> FastAPI:
    settings = get_app_settings()
    application = FastAPI(**settings.fastapi_kwargs)

    application.add_event_handler("startup", on_startup)
    application.add_event_handler("shutdown", on_shutdown)

    application.include_router(api_router)

    return application


# Process startup events
async def on_startup():
    logger.info("Application startup")
    await balancer.init_session()


# Process shutdown events
async def on_shutdown():
    logger.info("Application shutdown")
    await balancer.close_session()


app = get_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
