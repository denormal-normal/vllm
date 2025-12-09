# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

import heapq

import numpy as np
import torch

from .abstract import AbstractEplbPolicy


class DefaultEplbPolicy(AbstractEplbPolicy):
    @classmethod
    def balanced_packing(
        cls, weight: np.ndarray, num_packs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack n weighted objects to m packs, such that each bin contains exactly
        n/m objects and the weights of all packs are as balanced as possible.

        Parameters:
            weight: [X, n], the weight of each item
            num_packs: number of packs

        Returns:
            pack_index: [X, n], the pack index of each item
            rank_in_pack: [X, n], the rank of the item in the pack
        """
        num_layers, num_groups = weight.shape
        assert num_groups % num_packs == 0
        groups_per_pack = num_groups // num_packs

        if groups_per_pack == 1:
            pack_index = np.tile(np.arange(num_groups, dtype=np.int64), (num_layers, 1))
            rank_in_pack = np.zeros((num_layers, num_groups), dtype=np.int64)
            return pack_index, rank_in_pack

        # Sort and get indices in decending order
        indices = np.argsort(-weight, axis=-1)

        pack_index = np.full((num_layers, num_groups), -1, dtype=np.int64)
        rank_in_pack = np.full((num_layers, num_groups), -1, dtype=np.int64)

        # Run the packing algorithm
        for i in range(num_layers):
            # Initialize heap: (current_weight, pack_id, num_items)
            # Heap is ordered by current_weight (first element of tuple)
            heap = [(0.0, pack_id, 0) for pack_id in range(num_packs)]
            heapq.heapify(heap)

            for group in indices[i]:
                # Pop the pack with minimum weight
                current_weight, pack_id, num_items = heapq.heappop(heap)

                # Assign group to this pack
                assert num_items < groups_per_pack
                pack_index[i, group] = pack_id
                rank_in_pack[i, group] = num_items

                # Update pack weight and item count
                new_weight = current_weight + float(weight[i, group])
                new_num_items = num_items + 1

                # Push back to heap if pack is not full
                if new_num_items < groups_per_pack:
                    heapq.heappush(heap, (new_weight, pack_id, new_num_items))

        return pack_index, rank_in_pack

    @classmethod
    def replicate_experts(
        cls, weight: np.ndarray, num_phy: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Replicate `num_log` experts to `num_phy` replicas, such that the maximum
        load of all replicas is minimized.

        Parameters:
            weight: [X, num_log]
            num_phy: total number of experts after replication

        Returns:
            phy2log: [X, num_phy], logical expert id of each physical expert
            rank: [X, num_phy], the replica rank
            logcnt: [X, num_log], number of replicas for each logical expert
        """
        n, num_log = weight.shape
        num_redundant = num_phy - num_log
        assert num_redundant >= 0
        phy2log = np.tile(np.arange(num_phy, dtype=np.int64), (n, 1))
        rank = np.zeros((n, num_phy), dtype=np.int64)
        logcnt = np.ones((n, num_log), dtype=np.int64)
        arangen = np.arange(n, dtype=np.int64)
        for i in range(num_log, num_phy):
            redundant_indices = np.argmax(weight / logcnt, axis=-1)
            phy2log[:, i] = redundant_indices
            rank[:, i] = logcnt[arangen, redundant_indices]
            logcnt[arangen, redundant_indices] += 1
        return phy2log, rank, logcnt

    @classmethod
    def rebalance_experts_hierarchical(
        cls,
        weight: np.ndarray,
        num_physical_experts: int,
        num_groups: int,
        num_nodes: int,
        num_gpus: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters:
            weight: [num_moe_layers, num_logical_experts]
            num_physical_experts: number of physical experts after replication
            num_groups: number of expert groups
            num_nodes: number of server nodes, where the intra-node network
                (e.g, NVLink) is faster
            num_gpus: number of GPUs, must be a multiple of `num_nodes`

        Returns:
            phy2log: [layers, num_replicas], the expert
                index of each replica
            log2phy: [layers, num_logical_experts, X],
                the replica indices for each expert
            logcnt: [layers, num_logical_experts], number of
                physical replicas for each logical expert
        """
        num_layers, num_logical_experts = weight.shape
        assert num_logical_experts % num_groups == 0
        group_size = num_logical_experts // num_groups
        assert num_groups % num_nodes == 0
        groups_per_node = num_groups // num_nodes
        assert num_gpus % num_nodes == 0
        assert num_physical_experts % num_gpus == 0
        phy_experts_per_gpu = num_physical_experts // num_gpus

        def inverse(perm: np.ndarray) -> np.ndarray:
            inv = np.empty_like(perm)
            for i in range(perm.shape[0]):
                inv[i, perm[i]] = np.arange(perm.shape[1], dtype=np.int64)
            return inv

        # Step 1: pack groups to nodes
        tokens_per_group = weight.reshape(num_layers, num_groups, group_size).sum(
            axis=-1
        )
        group_pack_index, group_rank_in_pack = cls.balanced_packing(
            tokens_per_group, num_nodes
        )
        log2mlog = (
            (group_pack_index * groups_per_node + group_rank_in_pack)[:, :, np.newaxis]
            * group_size
            + np.arange(group_size, dtype=np.int64)
        ).reshape(num_layers, -1)
        mlog2log = inverse(log2mlog)

        # Step 2: construct redundant experts within nodes
        # [num_layers * num_nodes, num_logical_experts // num_nodes]
        tokens_per_mlog = np.empty(
            (num_layers, num_logical_experts), dtype=weight.dtype
        )
        for i in range(num_layers):
            tokens_per_mlog[i] = weight[i, mlog2log[i]]
        tokens_per_mlog = tokens_per_mlog.reshape(-1, num_logical_experts // num_nodes)
        phy2mlog, phyrank, mlogcnt = cls.replicate_experts(
            tokens_per_mlog, num_physical_experts // num_nodes
        )

        # Step 3: pack physical_experts to GPUs
        # [num_layers * num_nodes, num_physical_experts // num_nodes]
        tokens_per_phy = np.empty_like(phy2mlog, dtype=weight.dtype)
        for i in range(tokens_per_phy.shape[0]):
            tokens_per_phy[i] = (tokens_per_mlog[i] / mlogcnt[i])[phy2mlog[i]]
        pack_index, rank_in_pack = cls.balanced_packing(
            tokens_per_phy, num_gpus // num_nodes
        )
        phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
        pphy2phy = inverse(phy2pphy)

        pphy2mlog = np.empty_like(phy2mlog)
        for i in range(pphy2mlog.shape[0]):
            pphy2mlog[i] = phy2mlog[i, pphy2phy[i]]
        pphy2mlog = (
            pphy2mlog.reshape(num_layers, num_nodes, -1)
            + np.arange(
                0, num_logical_experts, num_logical_experts // num_nodes, dtype=np.int64
            ).reshape(1, -1, 1)
        ).reshape(num_layers, -1)

        pphy2log = np.empty_like(pphy2mlog)
        for i in range(num_layers):
            pphy2log[i] = mlog2log[i, pphy2mlog[i]]

        pphyrank = np.empty_like(phyrank)
        for i in range(pphyrank.shape[0]):
            pphyrank[i] = phyrank[i, pphy2phy[i]]
        pphyrank = pphyrank.reshape(num_layers, -1)

        logcnt = mlogcnt.reshape(num_layers, -1)
        logcnt_reordered = np.empty_like(logcnt)
        for i in range(num_layers):
            logcnt_reordered[i] = logcnt[i, log2mlog[i]]

        return pphy2log, pphyrank, logcnt_reordered

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Entry point for expert-parallelism load balancer.

        Parameters:
            weight: [layers, num_logical_experts], the load statistics for all
                logical experts
            num_replicas: number of physical experts, must be a multiple of
                `num_gpus`
            num_groups: number of expert groups
            num_nodes: number of server nodes, where the intra-node network
                (e.g, NVLink) is faster
            num_ranks: number of ranks, must be a multiple of `num_nodes`

        Returns:
            phy2log: [layers, num_replicas], the expert
                index of each replica
            log2phy: [layers, num_logical_experts, X],
                the replica indices for each expert
            logcnt: [layers, num_logical_experts], number of
                physical replicas for each logical expert
        """
        num_layers, num_logical_experts = weight.shape
        device = weight.device

        # Convert input tensor to numpy
        weight_np = weight.float().cpu().numpy()

        if num_groups % num_nodes == 0:
            # use hierarchical load-balance policy
            phy2log, phyrank, logcnt = cls.rebalance_experts_hierarchical(
                weight_np, num_replicas, num_groups, num_nodes, num_ranks
            )
        else:
            # use global load-balance policy
            phy2log, phyrank, logcnt = cls.rebalance_experts_hierarchical(
                weight_np, num_replicas, 1, 1, num_ranks
            )
        num_redundant_experts = num_replicas - num_logical_experts
        maxlogcnt = num_redundant_experts + 1
        log2phy = np.full(
            (num_layers, num_logical_experts, maxlogcnt),
            -1,
            dtype=np.int64,
        )

        # Scatter operation using numpy (equivalent to PyTorch's view().scatter_())
        # Create a view once, then modify it (modifications affect the original log2phy)
        log2phy_view = log2phy.reshape(num_layers, -1)
        flat_indices = phy2log * maxlogcnt + phyrank
        replica_indices = np.tile(
            np.arange(num_replicas, dtype=np.int64), (num_layers, 1)
        )

        for i in range(num_layers):
            log2phy_view[i, flat_indices[i]] = replica_indices[i]

        # Convert results back to tensors
        phy2log = torch.from_numpy(phy2log).to(device)
        log2phy = torch.from_numpy(log2phy).to(device)
        logcnt = torch.from_numpy(logcnt).to(device)

        return phy2log, log2phy, logcnt
