# ruff: noqa: SLF001
import asyncio
import contextlib
import threading
import time

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class _BlockingPolicy:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def infer(self, obs):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._state_lock:
            self.active -= 1
        return {"observation": obs}


class _ControllablePolicy:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0

    def infer(self, obs):
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        if not self.release.wait(timeout=1):
            raise TimeoutError("test did not release inference")
        with self._state_lock:
            self.active -= 1
        return {"observation": obs}


def test_inference_does_not_block_event_loop():
    policy = _BlockingPolicy()
    server = WebsocketPolicyServer(policy)

    async def scenario():
        inference = asyncio.create_task(server._infer({"id": 1}))
        event_loop_turns = 0
        while not inference.done():
            event_loop_turns += 1
            await asyncio.sleep(0)
        result = await inference
        return event_loop_turns, result

    event_loop_turns, result = asyncio.run(scenario())

    assert event_loop_turns > 1
    assert result == {"observation": {"id": 1}}


def test_inference_is_serialized_across_clients():
    policy = _BlockingPolicy()
    server = WebsocketPolicyServer(policy)

    async def scenario():
        return await asyncio.gather(
            server._infer({"id": 1}),
            server._infer({"id": 2}),
        )

    results = asyncio.run(scenario())

    assert policy.max_active == 1
    assert results == [
        {"observation": {"id": 1}},
        {"observation": {"id": 2}},
    ]


def test_cancelled_inference_keeps_policy_serialized():
    policy = _ControllablePolicy()
    server = WebsocketPolicyServer(policy)

    async def scenario():
        first = asyncio.create_task(server._infer({"id": 1}))
        assert await asyncio.to_thread(policy.started.wait, 1)
        first.cancel()
        second = asyncio.create_task(server._infer({"id": 2}))
        await asyncio.sleep(0)
        assert not second.done()
        assert policy.active == 1
        policy.release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await first
        return await second

    result = asyncio.run(scenario())

    assert policy.max_active == 1
    assert result == {"observation": {"id": 2}}
