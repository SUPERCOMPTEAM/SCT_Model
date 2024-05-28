from pydantic_settings import BaseSettings
from functools import lru_cache

import logging
from typing import Any, Dict, Tuple


class AppSettings(BaseSettings):

    debug: bool = True
    docs_url: str = "/swagger"
    openapi_prefix: str = ""
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/swagger"
    title: str = 'Avito price service API'
    version: str = '0.0.1'

    api_prefix: str = "/api/v1"

    logging_level: int = logging.INFO
    loggers: Tuple[str, str] = ("uvicorn.asgi", "uvicorn.access")

    @property
    def fastapi_kwargs(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "docs_url": self.docs_url,
            "openapi_prefix": self.openapi_prefix,
            "openapi_url": self.openapi_url,
            "redoc_url": self.redoc_url,
            "title": self.title,
            "version": self.version,
        }


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()
