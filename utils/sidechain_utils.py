"""
Utility functions for sidechain management and cycle detection.
"""

from typing import Dict, Set


def has_sidechain_cycle(
    sidechain_graph: Dict[int, int], source: int, target: int
) -> bool:
    """
    Check if adding a sidechain from source to target would create a cycle.

    Args:
        sidechain_graph: Current sidechain dependencies (consumer -> source)
        source: Chain that would provide the sidechain signal
        target: Chain that would consume the sidechain signal

    Returns:
        True if adding this sidechain would create a cycle

    Examples:
        >>> # No cycle - empty graph
        >>> has_sidechain_cycle({}, 0, 1)
        False

        >>> # Direct cycle: 0->1, trying to add 1->0
        >>> has_sidechain_cycle({1: 0}, 1, 0)
        True

        >>> # Indirect cycle: 0->1->2, trying to add 2->0
        >>> has_sidechain_cycle({1: 0, 2: 1}, 2, 0)
        True

        >>> # Valid chain: 0->1->2, adding 3->1
        >>> has_sidechain_cycle({1: 0, 2: 1}, 3, 1)
        False
    """
    # Create a temporary graph with the new sidechain edge
    temp_graph = sidechain_graph.copy()
    temp_graph[target] = source

    # Use DFS to detect cycles
    visited = set()
    rec_stack = set()

    def dfs(node: int) -> bool:
        if node in rec_stack:
            return True  # Cycle detected
        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)

        # Follow the sidechain dependency if it exists
        if node in temp_graph:
            next_node = temp_graph[node]
            if dfs(next_node):
                return True

        rec_stack.remove(node)
        return False

    # Check for cycles starting from the new target node
    return dfs(target)


def detect_sidechain_cycles(sidechain_graph: Dict[int, int]) -> Set[int]:
    """
    Detect all cycles in an existing sidechain dependency graph.

    Args:
        sidechain_graph: Sidechain dependencies (consumer -> source)

    Returns:
        Set of chain indices that are part of cycles

    Examples:
        >>> # No cycles
        >>> detect_sidechain_cycles({1: 0, 2: 1})
        set()

        >>> # Direct cycle: 0->1->0
        >>> detect_sidechain_cycles({0: 1, 1: 0})
        {0, 1}

        >>> # Indirect cycle: 0->1->2->0
        >>> detect_sidechain_cycles({0: 2, 1: 0, 2: 1})
        {0, 1, 2}
    """
    visited = set()
    rec_stack = set()
    cycle_nodes = set()

    def dfs(node: int) -> bool:
        if node in rec_stack:
            # Cycle detected - mark all nodes in the current recursion stack as part of cycle
            cycle_nodes.update(rec_stack)
            return True
        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)

        # Follow the sidechain dependency if it exists
        if node in sidechain_graph:
            next_node = sidechain_graph[node]
            if dfs(next_node):
                cycle_nodes.add(node)  # This node is also part of the cycle
                rec_stack.remove(node)
                return True

        rec_stack.remove(node)
        return False

    # Check all nodes in the sidechain graph
    for node in sidechain_graph:
        if node not in visited:
            dfs(node)

    return cycle_nodes


def build_sidechain_graph(fx_chains) -> Dict[int, int]:
    """
    Build a sidechain dependency graph from FX chains.

    Args:
        fx_chains: List of ChainDefinition objects or similar with FxChain attribute

    Returns:
        Dictionary mapping consumer chain index to source chain index
    """
    sidechain_graph = {}

    for chain_idx, chain in enumerate(fx_chains):
        for fx_setting in chain.FxChain:
            if (
                hasattr(fx_setting, "sidechain_input")
                and fx_setting.sidechain_input is not None
            ):
                sidechain_graph[chain_idx] = fx_setting.sidechain_input
                break  # Only one sidechain per chain allowed

    return sidechain_graph
