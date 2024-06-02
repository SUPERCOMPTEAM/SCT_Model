import asyncio
from aiohttp import ClientSession
import logging
import random


class Peer:
    def __init__(self, url: str, cnt_requests: int, cnt_responses: int, neuro_weight: float):
        self.url = url
        self.cnt_requests = cnt_requests
        self.cnt_responses = cnt_responses
        self.neuro_weight = neuro_weight


class ShmBlock:
    def __init__(self, cnt_requests: int, cnt_responses: int):
        self.cnt_requests = cnt_requests
        self.cnt_responses = cnt_responses

    def __str__(self):
        return f"{self.cnt_requests} {self.cnt_responses}"


class Balancer:
    def __init__(self, model, gap_in_requests, postfix="load/random"):
        self.model = model
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.peers = [
            Peer(f"http://mock_server1:8001/{postfix}", 0, 0, 0.0),
            Peer(f"http://mock_server2:8002/{postfix}", 0, 0, 0.0),
            Peer(f"http://mock_server3:8003/{postfix}", 0, 0, 0.0),
            Peer(f"http://mock_server4:8004/{postfix}", 0, 0, 0.0),
            Peer(f"http://mock_server5:8005/{postfix}", 0, 0, 0.0)
        ]
        self.shm = {peer.url: ShmBlock(0, 0) for peer in self.peers}
        self.nreq_since_last_weight_update = 0
        self.gap_in_requests = gap_in_requests
        self.lock = asyncio.Lock()
        self.logger.info("init sct neuro")

    async def init_session(self):
        self.session = ClientSession()
        self.logger.info("Session initialized")

    async def close_session(self):
        await self.session.close()
        self.logger.info("Session closed")

    async def get_peer(self) -> Peer:
        async with self.lock:
            for peer in self.peers:
                block = self.shm[peer.url]
                peer.cnt_requests = block.cnt_requests
                peer.cnt_responses = block.cnt_responses

            if self.nreq_since_last_weight_update >= self.gap_in_requests:
                self.nreq_since_last_weight_update = 0
            else:
                self.nreq_since_last_weight_update += 1

            if self.nreq_since_last_weight_update == 0:



                for peer in self.peers:
                    peer.neuro_weight = random.random()

            best = max(self.peers, key=lambda p: p.neuro_weight)
            self.shm[best.url].cnt_requests += 1

            self.logger.info("get sct neuro peer")
            self.logger.info(self.shm[best.url])
            return best

    async def make_request(self, peer: Peer):
        async with self.session.get(peer.url) as response:
            async with self.lock:
                self.shm[str(response.url)].cnt_responses += 1
            response_text = await response.text()
            self.logger.info(f"Response from {peer.url}: {response_text}")
            return response_text

    async def aux_filter(self):
        peer = await self.get_peer()
        return await self.make_request(peer)


balancer = Balancer(model=None, gap_in_requests=5)
