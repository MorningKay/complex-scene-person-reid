"""Batch samplers for Re-ID training."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from reid.data.common import ReIDSample


class PKBatchSampler(Sampler[list[int]]):
    """Sample batches with P identities and K images per identity."""

    def __init__(
        self,
        samples: Sequence[ReIDSample],
        num_pids: int,
        num_instances: int,
        batches_per_epoch: int | None = None,
    ) -> None:
        if num_pids <= 0:
            raise ValueError("num_pids must be a positive integer")
        if num_instances <= 0:
            raise ValueError("num_instances must be a positive integer")

        pid_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            pid = int(sample.pid)
            if pid >= 0:
                pid_to_indices[pid].append(index)

        pids = sorted(pid_to_indices)
        if len(pids) < num_pids:
            raise ValueError(
                "PK sampler requires at least num_pids train identities, "
                f"got {len(pids)} and num_pids={num_pids}"
            )

        batch_size = num_pids * num_instances
        if batches_per_epoch is None:
            batches_per_epoch = math.ceil(len(samples) / batch_size)
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be a positive integer")

        self.pid_to_indices = dict(pid_to_indices)
        self.pids = pids
        self.num_pids = num_pids
        self.num_instances = num_instances
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            selected_pids = random.sample(self.pids, self.num_pids)
            for pid in selected_pids:
                indices = self.pid_to_indices[pid]
                if len(indices) >= self.num_instances:
                    batch.extend(random.sample(indices, self.num_instances))
                else:
                    batch.extend(random.choices(indices, k=self.num_instances))
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
