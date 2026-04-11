"""
src/routing/hard_router.py
---------------------------
Hard router: routes each sample to a specific expert based on node label.

Usage:
  router = HardRouter(labels)
  expert_id = router.route(node_idx)        # 0 or 1
  node_list  = router.nodes_for_expert(0)   # all CBD nodes
"""
from __future__ import annotations

import numpy as np

LABEL_CBD = 0
LABEL_RES = 1


class HardRouter:
    def __init__(self, labels: np.ndarray):
        """
        Parameters
        ----------
        labels : (N,) int array – 0=CBD, 1=Residential
        """
        self.labels = labels

    def route(self, node_idx: int) -> int:
        """Return the expert id (0 or 1) for a given node."""
        return int(self.labels[node_idx])

    def nodes_for_expert(self, expert_id: int) -> list[int]:
        """Return list of node indices assigned to expert_id."""
        return [i for i, l in enumerate(self.labels) if l == expert_id]

    def filter_samples(self, samples: list[dict], expert_id: int) -> list[dict]:
        """
        Filter a list of samples to only those whose node_idx belongs to expert_id.
        Samples must have been built per-node (i.e. have a "node_idx" key) OR
        we treat the whole feature matrix and filter only by the node dimension.

        When samples are built for ALL nodes (multi-node), we wrap them.
        Here we assume samples are per-node and have key "node_idx".
        """
        assigned = set(self.nodes_for_expert(expert_id))
        return [s for s in samples if s.get("node_idx") in assigned]

    def split_samples_by_expert(
        self, samples: list[dict]
    ) -> dict[int, list[dict]]:
        """Split sample list into {expert_id: [samples]} dict."""
        result: dict[int, list] = {0: [], 1: []}
        for s in samples:
            eid = self.route(s.get("node_idx", 0))
            result[eid].append(s)
        return result
