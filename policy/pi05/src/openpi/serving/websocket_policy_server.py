import asyncio
import http
import logging
import threading
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._inference_lock = threading.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        session_id = id(websocket)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = await self._infer(obs, session_id)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                await asyncio.to_thread(self._close_session_serialized, session_id)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await asyncio.to_thread(self._close_session_serialized, session_id)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _infer(self, obs, session_id=0):
        # Yield the event loop for keepalives while keeping the policy single-threaded.
        return await asyncio.to_thread(self._infer_serialized, obs, session_id)

    def _infer_serialized(self, obs, session_id):
        # A cancelled coroutine cannot stop its worker thread. Keep serialization in that
        # thread so the next request cannot overlap the still-running inference.
        with self._inference_lock:
            infer_session = getattr(self._policy, "infer_session", None)
            if infer_session is not None:
                return infer_session(session_id, obs)
            return self._policy.infer(obs)

    def _close_session_serialized(self, session_id):
        with self._inference_lock:
            close_session = getattr(self._policy, "close_session", None)
            if close_session is not None:
                close_session(session_id)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
