import asyncio
import time
import uuid
from typing import Any, Dict, Optional
import numpy as np


class BatchInferenceProcessor:
    """Async batch processor using asyncio.Queue.

    Collects incoming inference requests, batches them together,
    runs ONNX inference, and returns results via futures.
    Tracks results by task_id for polling support.
    """

    def __init__(
        self,
        session,
        batch_size: int = 4,
        timeout_sec: float = 0.5,
        max_queue_size: int = 100,
    ):
        self.session = session
        self.batch_size = batch_size
        self.timeout_sec = timeout_sec
        self.max_queue_size = max_queue_size

        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        # Store futures keyed by task_id for polling
        self._results: Dict[str, asyncio.Future] = {}

    async def start(self):
        """Start the background worker task."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.queue.put(None)  # sentinel to unblock worker

    async def infer(self, input_data: Any, task_id: str = None) -> Any:
        """Submit a single inference request and wait for the result.

        Args:
            input_data: Pre-processed input ready for ONNX Runtime.
            task_id: Optional task ID for result tracking. If None, generates one.

        Returns:
            Model output tensor.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        future = asyncio.get_event_loop().create_future()
        self._results[task_id] = future

        await self.queue.put((input_data, future))

        try:
            return await asyncio.wait_for(future, timeout=self.timeout_sec)
        except asyncio.TimeoutError:
            # Remove from results if timed out
            self._results.pop(task_id, None)
            raise

    async def _worker_loop(self):
        """Background loop that batches requests and runs inference."""
        batch: list = []

        while self._running:
            try:
                # Wait for an item with a short timeout to allow checking _running
                item = await asyncio.wait_for(
                    self.queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                # Timeout elapsed; process any accumulated batch
                if batch:
                    await self._process_batch(batch)
                    batch = []
                continue

            if item is None:  # sentinel
                if batch:
                    await self._process_batch(batch)
                break

            input_data, future = item
            batch.append((input_data, future))

            if len(batch) >= self.batch_size:
                await self._process_batch(batch)
                batch = []

        # Drain remaining items
        while not self.queue.empty():
            try:
                data, fut = self.queue.get_nowait()
                batch.append((data, fut))
            except asyncio.QueueError:
                break
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list):
        """Run ONNX inference on a batch of inputs."""
        start = time.time()
        inputs = np.stack([item[0] for item in batch], axis=0)

        try:
            outputs = self.session.run(None, {"input": inputs})
            output_tensor = outputs[0]
        except Exception as e:
            # Signal error to all futures in the batch
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
            return

        for input_data, future in batch:
            if not future.done():
                future.set_result((output_tensor, time.time() - start))


# Convenience function
async def inference_with_queue(
    session,
    input_data: Any,
    batch_size: int = 4,
    timeout_sec: float = 0.5,
) -> Any:
    """Run a single inference via the queue-based batch processor."""
    processor = BatchInferenceProcessor(
        session=session,
        batch_size=batch_size,
        timeout_sec=timeout_sec,
    )
    await processor.start()
    try:
        return await processor.infer(input_data)
    finally:
        await processor.stop()