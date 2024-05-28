from fastapi import APIRouter
from services.balancer import balancer
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def upstream_response():
    response = await balancer.aux_filter()
    return response
