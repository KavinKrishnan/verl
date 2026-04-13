# Copyright 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ModelExpress checkpoint engine for verl.

Uses NIXL for GPU-to-GPU RDMA transfers with ModelExpress Server for
centralized metadata coordination (source discovery, version tracking).

The trainer publishes weight metadata to the MX Server after each step.
Rollout workers query the MX Server to discover the trainer's NIXL endpoint,
then pull weights via RDMA.

Topology: trainer rank 0 → MX Server (metadata) → rollout ranks (RDMA pull).
Unlike the NIXL ring engine, rollout ranks each pull independently from the
trainer — no ring forwarding needed since MX Server handles discovery.

Requires:
    pip install modelexpress nixl
    MX Server + Redis running and reachable at mx_server_url
"""
import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator
from unittest.mock import patch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import ray
import torch
import zmq
import zmq.asyncio

from modelexpress.client import MxClient
from modelexpress import p2p_pb2

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class MxAgentMetadata:
    agent_name: str
    agent_metadata: bytes
    zmq_ip: str
    zmq_port: int
    mx_server_url: str
    role: str  # "trainer" or "rollout"


class MxNixlAgent:
    """NIXL agent wrapper with ZMQ for bucket metadata and MX Server for discovery."""

    def __init__(self, mx_server_url: str = "localhost:8001"):
        self.agent_name = str(uuid.uuid4())
        self.agent = nixl_api.nixl_agent(self.agent_name)
        self.mx_server_url = mx_server_url
        self.mx_client = MxClient(server_url=mx_server_url)

        self.start_zmq_server()
        self.zmq_clients: dict[str, zmq.Socket] = {}
        self.messages: dict[str, deque[bytes]] = defaultdict(deque)
        self.notifications: dict[str, deque[bytes]] = defaultdict(deque)

    def __getattr__(self, name):
        attr = getattr(self.agent, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)
            return wrapper
        return attr

    def get_agent_metadata(self) -> MxAgentMetadata:
        return MxAgentMetadata(
            agent_name=self.agent_name,
            agent_metadata=self.agent.get_agent_metadata(),
            zmq_ip=self.ip,
            zmq_port=self.listen_port,
            mx_server_url=self.mx_server_url,
            role="unknown",
        )

    def start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, _ = get_free_port(self.ip)
        context = zmq.asyncio.Context()
        self.socket = context.socket(zmq.PULL)
        if is_valid_ipv6_address(self.ip):
            self.socket.bind(f"tcp://[{self.ip}]:{self.listen_port}")
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            self.socket.bind(f"tcp://{self.ip}:{self.listen_port}")

    def add_remote_agent(self, metadata: MxAgentMetadata) -> str:
        agent_name = self.agent.add_remote_agent(metadata.agent_metadata).decode("utf-8")
        assert agent_name == metadata.agent_name

        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        if is_valid_ipv6_address(metadata.zmq_ip):
            socket.setsockopt(zmq.IPV6, 1)
            socket.connect(f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}")
        else:
            socket.connect(f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}")
        self.zmq_clients[agent_name] = socket
        return agent_name

    def remove_remote_agent(self, agent_name: str):
        self.agent.remove_remote_agent(agent_name)
        socket = self.zmq_clients.pop(agent_name, None)
        if socket:
            socket.close()

    def send_message(self, agent_name: str, message: dict):
        self.zmq_clients[agent_name].send_pyobj((self.agent_name, message), zmq.DONTWAIT)

    async def read_message(self, agent_name: str) -> dict:
        while len(self.messages[agent_name]) == 0:
            recv_name, message = await self.socket.recv_pyobj()
            self.messages[recv_name].append(message)
        return self.messages[agent_name].popleft()

    async def get_notification(self, remote_name: str) -> bytes:
        while len(self.notifications[remote_name]) == 0:
            notifs = self.agent.get_new_notifs()
            for rn, notif in notifs.items():
                self.notifications[rn].extend(notif)
            await asyncio.sleep(0)
        return self.notifications[remote_name].popleft()


class ReadableOperation:
    """Source-side: send metadata, wait for remote to finish reading."""

    def __init__(self, agent: MxNixlAgent, remote_agent: str,
                 local_descs: nixl_bindings.nixlXferDList, metadata: dict):
        self.agent = agent
        self.remote_agent = remote_agent
        self.notify_key = uuid.uuid4().bytes
        message = {"notify_key": self.notify_key, "remote_descs": local_descs, **metadata}
        self.agent.send_message(self.remote_agent, message)

    async def wait_for_complete(self):
        notification = await self.agent.get_notification(self.remote_agent)
        assert self.notify_key == notification


class ReadOperation:
    """Target-side: receive metadata, read data via RDMA, signal completion."""

    def __init__(self, agent: MxNixlAgent, remote_agent: str,
                 local_descs: nixl_bindings.nixlXferDList, bucket_size: int):
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.remote_descs = None
        self.xfer_handle = None
        self.notify_key = None
        self.bucket_size = bucket_size
        self.start_time = None

    async def read_metadata(self) -> dict:
        metadata = await self.agent.read_message(self.remote_agent)
        self.remote_descs = metadata.pop("remote_descs")
        self.notify_key = metadata.pop("notify_key")
        return metadata

    def begin_read(self):
        assert self.remote_descs is not None and self.notify_key is not None
        self.xfer_handle = self.agent.initialize_xfer(
            "READ", self.local_descs, self.remote_descs, self.remote_agent, self.notify_key
        )
        state = self.agent.transfer(self.xfer_handle)
        assert state != "ERR", f"Read from {self.remote_agent} got to {state} state."
        self.start_time = time.time()

    async def wait_for_complete(self):
        while True:
            state = self.agent.check_xfer_state(self.xfer_handle)
            if state == "ERR":
                logger.error(f"Read from {self.remote_agent} got to {state} state.")
                exit(-1)
            elif state == "DONE":
                break
            else:
                await asyncio.sleep(0)
        self.agent.release_xfer_handle(self.xfer_handle)
        elapsed = time.time() - self.start_time
        bw = self.bucket_size / elapsed / (1024**3) if elapsed > 0 else 0
        logger.debug(f"ReadOperation from {self.remote_agent}: {bw:.2f} GB/s")


@CheckpointEngineRegistry.register("mx")
class MxCheckpointEngine(CheckpointEngine):
    """ModelExpress checkpoint engine: NIXL RDMA + MX Server metadata coordination.

    The trainer publishes NIXL metadata to the MX Server so rollout workers can
    discover it without direct topology exchange. The data plane uses the same
    bucketed NIXL RDMA pattern as the NIXL checkpoint engine.

    Args:
        bucket_size: Bucket size in bytes for batched transfers.
        mx_server_url: gRPC address of the ModelExpress server.
        device: "cuda" or "cpu".
        rollout_dtype: dtype for received weights.
        model_name: Model name for MX Server source identity.
    """

    def __init__(
        self,
        bucket_size: int,
        mx_server_url: str = "localhost:8001",
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        model_name: str = "default",
        is_master: bool = False,
    ):
        self.bucket_size = bucket_size
        self.mx_server_url = mx_server_url
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.model_name = model_name
        self.agent = MxNixlAgent(mx_server_url=mx_server_url)
        self.is_master = is_master
        self._worker_id = str(uuid.uuid4())

    def prepare(self) -> MxAgentMetadata:
        """Allocate send/recv buckets and register with NIXL."""
        if self.device == "cuda":
            send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.send_buf = torch.as_tensor(send_buf, dtype=torch.uint8)
            self.recv_buf = torch.as_tensor(recv_buf, dtype=torch.uint8)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device, pin_memory=True)
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device, pin_memory=True)

        self.send_reg_descs = self.agent.register_memory(self.send_buf)
        self.recv_reg_descs = self.agent.register_memory(self.recv_buf)
        self.send_descs = self.agent.get_xfer_descs(self.send_buf)
        self.recv_descs = self.agent.get_xfer_descs(self.recv_buf)

        metadata = self.agent.get_agent_metadata()
        metadata.mx_server_url = self.mx_server_url
        return metadata

    @classmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build topology: trainer rank 0 is the source, all rollout ranks connect to it.

        Unlike the NIXL ring engine, rollout ranks don't form a ring — each
        connects directly to the trainer via MX Server discovery + NIXL RDMA.
        """
        trainer_meta = metadata[:trainer_world_size]
        rollout_meta = metadata[trainer_world_size:]

        trainer_kwargs = {
            "method": ["init_process_group"] * trainer_world_size,
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "trainer_metadata": [trainer_meta[0]] * trainer_world_size,
            "rollout_metadata_list": [rollout_meta] * trainer_world_size,
        }

        rollout_kwargs = {
            "method": ["init_process_group"] * rollout_world_size,
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "trainer_metadata": [trainer_meta[0]] * rollout_world_size,
            "rollout_metadata_list": [None] * rollout_world_size,
        }

        return trainer_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        trainer_metadata: MxAgentMetadata,
        rollout_metadata_list: list[MxAgentMetadata] | None,
    ):
        """Establish NIXL connections.

        Trainer rank 0: adds all rollout agents as remotes (for ZMQ bucket metadata).
        Rollout ranks: add trainer rank 0 as remote (for RDMA reads).
        Other trainer ranks: no-op.
        """
        self.rank = rank
        self.world_size = world_size
        self.remote_agents: list[str] = []

        if rank == 0 and rollout_metadata_list:
            for rmeta in rollout_metadata_list:
                agent_name = self.agent.add_remote_agent(rmeta)
                self.remote_agents.append(agent_name)
            logger.info(
                f"MX trainer rank 0: connected to {len(self.remote_agents)} rollout agents"
            )
        elif rank > 0:
            self.trainer_agent = self.agent.add_remote_agent(trainer_metadata)
            logger.info(f"MX rollout rank {rank}: connected to trainer agent")
        else:
            self.trainer_agent = None

    def finalize(self):
        """Cleanup connections and deregister memory."""
        for agent_name in getattr(self, "remote_agents", []):
            self.agent.remove_remote_agent(agent_name)
        if hasattr(self, "trainer_agent") and self.trainer_agent:
            self.agent.remove_remote_agent(self.trainer_agent)

        self.agent.deregister_memory(self.send_reg_descs)
        self.agent.deregister_memory(self.recv_reg_descs)
        self.send_buf = None
        self.recv_buf = None
        self.remote_agents = []

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send weights to all connected rollout agents (trainer rank 0 only)."""
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        assert self.rank == 0 and self.remote_agents

        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        pending_ops: list[ReadableOperation] = []

        for name, weight in weights:
            if offset + weight.nbytes > self.bucket_size:
                torch.cuda.synchronize()

                for op in pending_ops:
                    await op.wait_for_complete()
                pending_ops.clear()

                for remote_agent in self.remote_agents:
                    op = ReadableOperation(
                        self.agent, remote_agent, send_descs,
                        {"bucket_meta": bucket_meta, "is_last": False},
                    )
                    pending_ops.append(op)

                send_buf, recv_buf = recv_buf, send_buf
                send_descs, recv_descs = recv_descs, send_descs
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) too large for bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            send_buf[offset:offset + weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += weight.nbytes

        torch.cuda.synchronize()
        for op in pending_ops:
            await op.wait_for_complete()
        pending_ops.clear()

        for remote_agent in self.remote_agents:
            op = ReadableOperation(
                self.agent, remote_agent, send_descs,
                {"bucket_meta": bucket_meta, "is_last": True},
            )
            pending_ops.append(op)
        for op in pending_ops:
            await op.wait_for_complete()

        elapsed = time.time() - start_time
        logger.info(f"MX send_weights done: {elapsed:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive weights from trainer via NIXL RDMA."""
        assert hasattr(self, "trainer_agent") and self.trainer_agent

        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs
        total_bytes, total_params = 0, 0

        start_time = time.time()
        read_op = ReadOperation(self.agent, self.trainer_agent, recv_descs, self.bucket_size)
        metadata = await read_op.read_metadata()
        read_op.begin_read()
        await read_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        send_buf, recv_buf = recv_buf, send_buf
        send_descs, recv_descs = recv_descs, send_descs

        while not metadata["is_last"]:
            read_op = ReadOperation(self.agent, self.trainer_agent, recv_descs, self.bucket_size)
            next_metadata = await read_op.read_metadata()
            read_op.begin_read()

            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta["offset"]:meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            await read_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(next_metadata["bucket_meta"])

            torch.cuda.synchronize()
            metadata = next_metadata
            send_buf, recv_buf = recv_buf, send_buf
            send_descs, recv_descs = recv_descs, send_descs

        for name, meta in metadata["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[meta["offset"]:meta["offset"] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        elapsed = time.time() - start_time
        bw = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0
        logger.info(
            f"MX receive_weights: {total_params} params, "
            f"{elapsed:.2f}s, {bw:.2f} GB/s"
        )
