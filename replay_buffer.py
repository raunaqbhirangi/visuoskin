import random

import numpy as np
import torch

def _worker_init_fn(worker_id):
    # seed = np.random.get_state()[1][0] + worker_id
    # np.random.seed(seed)
    # random.seed(seed)
    np.random.seed(worker_id)
    random.seed(worker_id)


def make_expert_replay_loader(iterable, batch_size):
    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
