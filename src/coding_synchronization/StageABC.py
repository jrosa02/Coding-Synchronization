import abc
import asyncio
from typing import Any

import numpy as np


class StageABC(abc.ABC):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.input_queue = None
        self.output_queue = None

    def __repr__(self) -> str:
        return self.__class__.__name__

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def connect(self, input_queue: asyncio.Queue | None, output_queue: asyncio.Queue | None):
        if input_queue:
            self.input_queue = input_queue
        if output_queue:
            self.output_queue = output_queue

    def connect_to(self, other: "StageABC"):
        q = self.output_queue or other.input_queue or asyncio.Queue(maxsize=2)
        other.connect(q, None)
        self.connect(None, q)

    def connect_from(self, other: "StageABC"):
        q = other.output_queue or self.input_queue or asyncio.Queue(maxsize=2)
        other.connect(None, q)
        self.connect(q, None)

    async def run(self):
        if self.input_queue and self.output_queue:
            await self.run_pipe()
        elif self.input_queue:
            await self.run_sink()
        elif self.output_queue:
            await self.run_source()
        else:
            raise ValueError(f"Module not connected{self.__repr__()}")

    async def run_pipe(self):
        while True:
            item = await self.input_queue.get()  # pyright: ignore[reportOptionalMemberAccess]
            if item is None:
                break
            result = self.process(item)
            await self.output_queue.put(result)  # pyright: ignore[reportOptionalMemberAccess]
        await self.output_queue.put(None)  # pyright: ignore[reportOptionalMemberAccess]

    async def run_source(self):
        while True:
            item = self.generate()
            if item is None:
                break
            await self.output_queue.put(item)  # pyright: ignore[reportOptionalMemberAccess]
        await self.output_queue.put(None)  # pyright: ignore[reportOptionalMemberAccess]

    async def run_sink(self):
        while True:
            item = await self.input_queue.get()  # pyright: ignore[reportOptionalMemberAccess]
            if item is None:
                break
            self.consume(item)

    def process(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def generate(self) -> np.ndarray | None:
        raise NotImplementedError()

    def consume(self, signal: np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class CompoundStage(StageABC):
    def __init__(self, stages: list[StageABC], seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.stages = stages

        for stage in self.stages:
            stage.set_seed(seed)

        for i in range(len(self.stages) - 1):
            self.stages[i].connect_to(self.stages[i + 1])

    def connect(self, input_queue: asyncio.Queue | None, output_queue: asyncio.Queue | None):
        super().connect(input_queue, output_queue)
        if input_queue and self.stages:
            self.stages[0].input_queue = input_queue
        if output_queue and self.stages:
            self.stages[-1].output_queue = output_queue

    async def run(self):
        await asyncio.gather(*[stage.run() for stage in self.stages])

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def process(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CompoundStage uses async queue-based run(), not process()")

    def generate(self) -> np.ndarray | None:
        raise NotImplementedError("CompoundStage uses async queue-based run(), not generate()")

    def consume(self, signal: np.ndarray) -> None:
        raise NotImplementedError("CompoundStage uses async queue-based run(), not consume()")


class Terminator(StageABC):
    def consume(self, signal: np.ndarray[tuple[Any, ...], np.dtype[Any]]):
        return

    def reset(self) -> None:
        pass


class StageRunner:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.pipe_elements: list[StageABC] = []

    def append(self, pipe_element: StageABC):
        pipe_element.set_seed(self.seed)
        if len(self.pipe_elements):
            pipe_element.connect_from(self.pipe_elements[-1])
        self.pipe_elements.append(pipe_element)

    async def _run_all(self):
        await asyncio.gather(*[elem.run() for elem in self.pipe_elements])

    def run(self):
        asyncio.run(self._run_all())

    def __repr__(self) -> str:
        repr = ""
        for elem in self.pipe_elements[:-1]:
            repr += str(elem) + "->"
        repr += str(self.pipe_elements[-1])
        return repr

    def reset(self):
        for elem in self.pipe_elements:
            elem.reset()
