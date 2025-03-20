"""
(Improving) Incremental Stochastic Beam Search (SBS) implementation for NumPy, mainly based on the implementation in UniqueRandomizer
https://github.com/google-research/unique-randomizer/blob/master/unique_randomizer/unique_randomizer.py
with alterations, such as:
    - Our log prob modification, see below.
    - batching SBS to allow higher batch sizes in the policy network
    - flag for keeping memory low (i.e., deleting client states after they have been visited but keeping the logits)

The key idea is to modify the log-probabilities of nodes in sampled trajectories. We do this in two ways:
1.) Subtract the log-prob of a sampled sequence from all intermediate nodes in the sequence. This is incremental
    SBS, where we perform sampling without replacement over many batches.
2.) Use the sampled trajectories to compute the expected objective over the policy (derived from priority sampling),
    then update the log-probs of all intermediate nodes with the advantage of the trajectory over the expectation.
    Note: we need to be careful about proper normalization.
"""
import typing

import sys
import numpy as np
import core.stochastic_beam_search as sbs
from typing import Callable, List, Optional, Tuple, NoReturn

sys.setrecursionlimit(10000)

def log_subtract(x: float, y: float) -> float:
    """Returns log(exp(x) - exp(y)), or negative infinity if x <= y."""
    # Inspired by https://stackoverflow.com/questions/778047.
    with np.errstate(divide='ignore'):
        result = x + np.log1p(-np.exp(np.minimum(y - x, 0)))
    return result


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel.
    Adapted from `https://github.com/wouterkool/stochastic-beam-search/blob/stochastic-beam-search/fairseq/gumbel.py`.
    In practice, we will need to set x := \kappa - \phi_i
    """
    with np.errstate(divide='ignore'):
        y = np.exp(-x)
        result = np.where(
            x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
            -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
            np.log(-np.expm1(-np.exp(-x)))  # Hope for the best
        )
    return result


def gumbel_without_replacement_expectation(
        sbs_leaves: List[sbs.BeamLeaf],
        leaf_eval_fn: Callable,
        normalize_importance_weights: bool = True) -> float:
    """
    Compute the expected outcome of rolling out the current policy given the sampled trajectories from SBS.
    For details of how to compute it, we refer to the original SBS paper
    "https://proceedings.mlr.press/v97/kool19a/kool19a.pdf", Section 4.2 "BlEU score estimation".

    Parameters:
        sbs_leaves [List[sbs.BeamLeaf]]: Sampled trajectories from which we compute the expected value.
        leaf_eval_fn: [Callable]
        normalize_importance_weights [bool]: In practice, the importance weighted estimator can have high variance,
            and it can be preferable to normalize them by their sum. Default is `True`.

    Returns:
        expected_outcome [float]: Expectation computed from sbs_leaves using optional importance weights.
    """
    if len(sbs_leaves) == 1:
        return leaf_eval_fn(sbs_leaves[0].state[1])
    # Sort leaves in descending order by their sampled Gumbels
    sbs_leaves = sorted(sbs_leaves, key=lambda x: x.gumbel, reverse=True)

    # We need the log-probs, gumbels and outcomes for each leaf
    log_probs = np.array([leaf.log_probability for leaf in sbs_leaves])
    outcomes = np.array([leaf_eval_fn(leaf.state[1]) for leaf in sbs_leaves])

    # We have the beam leaves sorted by their sampled Gumbels, so the last element is the smallest gumbel
    kappa = sbs_leaves[-1].gumbel
    # Discard the last entry, we won't need it
    log_probs = log_probs[:-1]
    _outcomes = outcomes[:-1]
    # See Appendix C "Numerical stability of importance weights" in SBS paper
    importance_weights = np.exp(log_probs - gumbel_log_survival(kappa - log_probs))
    if normalize_importance_weights:
        importance_weights = importance_weights / np.sum(importance_weights)
    expected_outcome = float(np.sum(importance_weights * _outcomes))
    return expected_outcome


class _TrieNode(object):
    """A trie node as in UniqueRandomizer.

    Attributes:
        parent: The _TrieNode parent of this node, or `None` if this node is the root.
        index_in_parent: The action index of this node in the parent, or `None` if this node
            is the root.
        children: A list of _TrieNode children. A child may be `None` if it is not
            expanded yet. The entire list will be `None` if this node has never visited
            a child yet. The list will be empty if this node is a leaf in the trie.
        unsampled_log_masses: A numpy array containing the current (!!) (unsampled) log
            probability mass of each child, or `None` if this node has never sampled a
            child yet.
    """

    def __init__(self, parent: Optional['_TrieNode'], index_in_parent: Optional[int]) -> None:
        """Initializes a _TrieNode.

        Parameters:
            parent [Optional[_TrieNode]]: The parent of this node, or `None` if this node is the root.
            index_in_parent [Optional[int]]: This node's action index in the parent node, or `None` if this
                node is the root.
        """
        self.parent = parent
        self.index_in_parent = index_in_parent
        self.children = None
        self.unsampled_log_masses = None
        # Caches (state, is_leaf)-tuples obtained in SBS so we only transition to new states if needed.
        self.sbs_child_state_cache: Optional[List[Tuple[sbs.State, bool]]] = None
        if self.parent is None:
            self.ancestors = []  # List of ancestor nodes
        elif self.parent.ancestors is None:
            self.ancestors = None
        else:
            self.ancestors = [self.parent] + self.parent.ancestors

    def get_action_sequence(self):
        # Walks up the tree up to the root and returns the first action leading to this node.
        seq = []
        node = self
        while node.parent is not None:
            seq.append(node.index_in_parent)
            node = node.parent
        seq.reverse()
        return seq

    def initial_log_mass_if_not_sampled(self) -> float:
        """Returns this node's initial log probability mass.

        This assumes that no samples have been drawn from this node yet.
        """
        # If no samples have been drawn yet, the unsampled log mass equals the
        # desired initial log mass.
        return (self.parent.unsampled_log_masses[self.index_in_parent]
                # If the node is the root, the initial log mass is 0.0.
                if self.parent else 0.0)

    def mark_leaf(self) -> None:
        """Marks this node as a leaf."""
        self.children = []

    def exhausted(self) -> bool:
        """Returns whether all of the mass at this node has been sampled."""
        # Distinguish [] and None.
        if self.children is not None and not len(self.children):
            return True
        if self.unsampled_log_masses is None:
            return False  # This node is not a leaf but has never been sampled from.
        return all(np.isneginf(self.unsampled_log_masses))

    def mark_mass_sampled(self, log_mass: float) -> None:
        """Recursively subtracts log_mass from this node and its ancestors."""
        if not self.parent:  # is the root.
            return
        if self.exhausted():
            new_log_mass = -np.inf  # explicitly set the node's log_mass to -inf to prevent sampling from it again.
        else:
            new_log_mass = log_subtract(self.parent.unsampled_log_masses[self.index_in_parent], log_mass)
        self.parent.unsampled_log_masses[self.index_in_parent] = new_log_mass
        self.parent.mark_mass_sampled(log_mass)


class IncrementalSBS:
    """
    Main class for incrementally performing Stochastic Beam Search and updating the logits of subsequences met on the way.

    Construct an instance of this class for a batch of "initial states" corresponding to problem instances.
    """
    def __init__(self, initial_states: List[sbs.State],
                 child_log_probability_fn: Callable[[List[sbs.State]], List[np.ndarray]],
                 child_transition_fn: Callable[[List[Tuple[sbs.State, int]]], List[Tuple[sbs.State, bool]]],
                 leaf_evaluation_fn: Optional[Callable[[sbs.State], float]] = None,
                 batch_leaf_evaluation_fn: Optional[Callable[[List[sbs.State]], np.array]] = None,
                 memory_aggressive: bool = False):
        """
        Parameters:
            initial_states: List of initial states used as root nodes.

            child_log_probability_fn: A function that takes a list of states and returns
                the log probabilities of the child states of each input state.

            child_transition_fn: A function that takes a list of (state, i) pairs and maps
                each to a (ith_child, is_leaf) pair. If ith_child is a leaf state, is_leaf
                should be True, and ith_child will potentially be an actual sampled item
                that should be returned by stochastic_beam_search (it may have a different
                form than other non-leaf states).
                (Wrapped and passed directly to sbs.stochastic_beam_search)

            leaf_evaluation_fn: An optional function that takes the sbs.State of a leaf (i.e., a finished trajectory)
                and returns some "outcome" of the trajectory, such as an objective. Must be provided for
                "policy_improvement"-type updates where the log-probs are updated according to their advantage,
                so we update with them with the goal of MAXIMIZING the outcome. If the original goal of
                the problem is to minimize some objective (e.g., routing problems), make sure that you flip the
                sign of the objective.

            memory_aggressive [bool]: If this is True, the internal states in the search tree are erased after passing them,
                thus on the one hand needing to re-transition to them when passing them again, but on the other hand
                saving memory.
        """
        # A node will always be a tuple (_TrieNode, sbs.State).
        self.root_nodes = [
            (_TrieNode(None, None), state)
            for state in initial_states
        ]
        self.root_nodes_exhausted = [False] * len(self.root_nodes)  # will be True if all probability mass has been returned
        self.leaf_evaluation_fn = leaf_evaluation_fn
        self.batch_leaf_evaluation_fn = batch_leaf_evaluation_fn
        self.child_log_probability_fn = child_log_probability_fn
        self.child_transition_fn = child_transition_fn
        self.memory_aggressive = memory_aggressive

    def perform_tasar(self, beam_width: int,
                                deterministic: bool = False,
                                nucleus_top_p: float = 1.,
                                replan_steps: int = 10,
                                sbs_keep_intermediate: bool = False
                                ) -> List[List[sbs.BeamLeaf]]:

        best_leaf_batch: List[Optional[sbs.BeamLeaf]] = [None] * len(self.root_nodes)  # carries the best leaves
        best_leaf_action_seqs_batch = [[] for _ in self.root_nodes]
        all_leaves = [[] for _ in self.root_nodes]  # List of all leaves that have been encountered during the search
        child_log_probability_fn = self.wrap_child_log_probability_fn(self.child_log_probability_fn, False)
        child_transition_fn = self.wrap_child_transition_fn(self.child_transition_fn, self.memory_aggressive)

        # We start at the root nodes.
        root_nodes = self.root_nodes
        for node, _ in root_nodes:
            node.ancestors = None  # so that we don't track ancestors

        step_count = -1
        while True:
            step_count = (step_count + 1) % replan_steps
            unfinished_root_nodes = [x for x in root_nodes if x is not None]
            unfinished_root_indices = [i for i, x in enumerate(root_nodes) if x is not None]
            if not len(unfinished_root_nodes):
                # We are done
                break

            # As long as we have root nodes and we should replan, perform SBS
            if step_count == 0:
                sbs_leaves_batch = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=unfinished_root_nodes,
                    beam_width=beam_width,
                    deterministic=deterministic,
                    top_p=1 if deterministic else nucleus_top_p,
                    keep_intermediate=sbs_keep_intermediate
                )
                deterministic = False
            else:
                sbs_leaves_batch = [None] * len(unfinished_root_nodes)

            for batch_idx, beam_leaves in enumerate(sbs_leaves_batch):
                # Get the best leaf and check if it is better than what we have
                _idx = unfinished_root_indices[batch_idx]
                if beam_leaves is not None:
                    # Obtain the objective function evaluation for all leaves. Also add them to the list of all
                    # leaves that we encountered.
                    self.batch_leaf_evaluation_fn([y.state[1] for y in beam_leaves])
                    all_leaves[_idx].extend(
                        [sbs.BeamLeaf(state=y.state[1], log_probability=y.log_probability, gumbel=0)
                         for y in beam_leaves]
                    )

                    best_leaf = sorted(beam_leaves, key=lambda y: self.leaf_evaluation_fn(y.state[1]), reverse=True)[0]
                    best_leaf_node, best_leaf_client_state = best_leaf.state
                    if best_leaf_batch[_idx] is None or self.leaf_evaluation_fn(
                            best_leaf_client_state) > self.leaf_evaluation_fn(best_leaf_batch[_idx].state):
                        # We have a best leaf. Add the best leaf to the solution list, but remove the trie node from the state so we don't hold on to it.
                        # Also add the corresponding action sequence.
                        best_leaf_batch[_idx] = best_leaf._replace(state=best_leaf_client_state)
                        best_leaf_action_seqs_batch[_idx] = best_leaf_node.get_action_sequence()

                # Now as we have the best leaf (which can be unaltered), we get the root action of it and remove it
                # from the action sequence
                root_action = best_leaf_action_seqs_batch[_idx].pop(0)
                # Mark all sequences as sampled
                if beam_leaves is not None:
                    for beam_leaf in beam_leaves:
                        leaf_node, client_state = beam_leaf.state
                        log_sampled_mass = leaf_node.initial_log_mass_if_not_sampled()
                        leaf_node.mark_mass_sampled(log_sampled_mass)

                # Shift the root nodes
                root_node, root_state = root_nodes[_idx]
                if root_node.sbs_child_state_cache[root_action] is not None:
                    root_state = root_node.sbs_child_state_cache[root_action][0]
                else:
                    # we are in memory aggressive mode: transition again
                    root_state, is_leaf = self.child_transition_fn([(root_state, root_action)])[0]
                    #if is_leaf:
                    #    root_node.mark_leaf()

                root_node = root_node.children[root_action]
                root_node.parent = None
                if not len(root_node.children) or root_node.exhausted():
                    # is a leaf or we have sampled everything from it
                    root_nodes[_idx] = None
                else:
                    root_nodes[_idx] = (root_node, root_state)

        all_leaves_sorted = [
            sorted(x, key=lambda y: self.leaf_evaluation_fn(y.state), reverse=True)
            for x in all_leaves
        ]

        return all_leaves_sorted

    def perform_incremental_sbs(self, beam_width: int, num_rounds: int, nucleus_top_p: float = 1.,
                                sbs_keep_intermediate: bool = False,
                                best_objective: Optional[float] = None) -> List[List[sbs.BeamLeaf]]:
        """
        Performs incremental SBS with the given type of updating the log-probs. Note that the trie and all log-prob updates
        persist, so calling the method multiple times will not reset the trie.

        Parameters:
            beam_width [int]: Beam width for one round of SBS
            num_rounds [int]: Number of SBS rounds, where we update the log-probs after each round.
        """
        leaves_batch: List[List[sbs.BeamLeaf]] = [[] for _ in range(len(self.root_nodes))]
        child_log_probability_fn = self.wrap_child_log_probability_fn(self.child_log_probability_fn, False)
        child_transition_fn = self.wrap_child_transition_fn(self.child_transition_fn, self.memory_aggressive)

        sample_until_better = False
        if isinstance(num_rounds, tuple):
            if best_objective is not None:
                min_num_rounds = num_rounds[0]
                num_rounds = num_rounds[1]
                sample_until_better = True
            else:
                num_rounds = num_rounds[0]

        for round_idx in range(num_rounds):
            # Check for each root node if its exhausted. If so, we won't search it again
            unexhausted_root_idcs = [i for i, exhausted in enumerate(self.root_nodes_exhausted) if not exhausted]
            root_nodes_to_search = [self.root_nodes[i] for i in unexhausted_root_idcs]

            if not len(root_nodes_to_search):
                break

            round_beam_leaves_batch = sbs.stochastic_beam_search(
                child_log_probability_fn=child_log_probability_fn,
                child_transition_fn=child_transition_fn,
                root_states=root_nodes_to_search,
                beam_width=beam_width,
                deterministic=False,
                top_p=nucleus_top_p,
                keep_intermediate=sbs_keep_intermediate
            )

            # Update probabilities and remove _TrieNode parts of the leaves.
            for j, beam_leaves in enumerate(round_beam_leaves_batch):
                batch_idx = unexhausted_root_idcs[j]
                # here we replace the beam leaf state consisting of (_TrieNode, sbs.State) tuples
                # to only contain the sbs.State
                client_beam_leaves = []
                for i, beam_leaf in enumerate(beam_leaves):
                    leaf_node, client_state = beam_leaf.state
                    log_sampled_mass = leaf_node.initial_log_mass_if_not_sampled()
                    leaf_node.mark_mass_sampled(log_sampled_mass)
                    client_beam_leaves.append(beam_leaf._replace(state=client_state))

                leaves_batch[batch_idx].extend(client_beam_leaves)
                self.root_nodes_exhausted[batch_idx] = self.root_nodes[batch_idx][0].exhausted()
                self.batch_leaf_evaluation_fn([y.state for y in client_beam_leaves])
                if sample_until_better:
                    best_leaf_obj = sorted([self.leaf_evaluation_fn(y.state) for y in client_beam_leaves], reverse=True)[0]
                    # check if we found a new best leaf
                    if best_leaf_obj > best_objective + 1e-2 and round_idx >= min_num_rounds - 1:
                        print("Breaking in round", round_idx + 1)
                        self.root_nodes_exhausted[batch_idx] = True  # simply mark it as done

        for i, leaves in enumerate(leaves_batch):
            leaves_batch[i] = sorted(leaves, key=lambda y: self.leaf_evaluation_fn(y.state), reverse=True)

        return leaves_batch

    @staticmethod
    def wrap_child_log_probability_fn(child_log_probability_fn, normalize_advantage_by_visit_count: bool) -> Callable[[List[Tuple[_TrieNode, sbs.State]]], List[np.ndarray]]:
        def wrapper_child_log_probability_fn(node_state_tuples: List[Tuple[_TrieNode, sbs.State]]) -> List[np.ndarray]:
            """Computes child probabilities while updating the trie."""
            results = [None] * len(node_state_tuples)
            unexpanded_client_states = []  # States for which we haven't computed log probs yet
            unexpanded_indices = []  # Corresponding to index in `node_state_tuples`

            for i, (node, client_state) in enumerate(node_state_tuples):
                if node.unsampled_log_masses is None:
                    # We haven't computed log probs for this node yet.
                    unexpanded_client_states.append(client_state)
                    unexpanded_indices.append(i)
                else:
                    # We already have computed log probabilities for this node (and also may have already updated them!)
                    # However, the log probs might not be normalized, so we need to normalize them.
                    # Note that normalizing the log probs is the same as if we would first subtract the parent's log mass to obtain
                    # the conditional log probs and then normalize them again
                    log_unnormalized = node.unsampled_log_masses

                    unnormalized = np.exp(log_unnormalized - np.max(log_unnormalized))
                    with np.errstate(divide='ignore'):
                        results[i] = np.log(unnormalized / np.sum(unnormalized))

            # Use client's child_log_probability_fn to get probabilities for unexpanded states.
            if unexpanded_client_states:
                client_fn_results = child_log_probability_fn(unexpanded_client_states)
                for i, log_probs in zip(unexpanded_indices, client_fn_results):
                    results[i] = log_probs
                    # also set the log probs on the node for which we computed them
                    node = node_state_tuples[i][0]
                    node.unsampled_log_masses = log_probs + node.initial_log_mass_if_not_sampled()

            return typing.cast(List[np.ndarray], results)

        return wrapper_child_log_probability_fn

    @staticmethod
    def wrap_child_transition_fn(child_transition_fn, memory_aggressive: bool) -> Callable[[List[Tuple[Tuple[_TrieNode, sbs.State], int]]],
                                                                  List[Tuple[Tuple[_TrieNode, sbs.State], bool]]]:
        def wrapper_child_transition_fn(node_state_action_index_pairs: List[Tuple[Tuple[_TrieNode, sbs.State], int]]) -> List[Tuple[Tuple[_TrieNode, sbs.State], bool]]:
            """Computes child states while updating the trie."""
            results = [None] * len(node_state_action_index_pairs)
            unexpanded_client_state_index_pairs = []  # States for which we haven't computed the transition yet
            unexpanded_indices = []  # Corresponding to index in `node_state_action_index_pairs`

            for i, ((node, client_state), child_index) in enumerate(node_state_action_index_pairs):
                # Initialize children structures if needed. We can be sure that `unsampled_log_masses` is not None
                if node.children is None:
                    num_children = len(node.unsampled_log_masses)
                    node.children = [None] * num_children
                    node.sbs_child_state_cache = [None] * num_children

                if node.children[child_index] is None or node.sbs_child_state_cache[child_index] is None:
                    # The child has not been created before, or there is no entry for the child in the sbs cache.
                    # The latter only happens in memory_aggressive mode
                    unexpanded_client_state_index_pairs.append((client_state, child_index))
                    unexpanded_indices.append(i)
                else:
                    # There already is a child which we can use.
                    child_client_state, child_is_leaf = node.sbs_child_state_cache[child_index]
                    results[i] = ((node.children[child_index], child_client_state), child_is_leaf)

            # Use client's child_transition_fn to get child client states
            if unexpanded_client_state_index_pairs:
                client_fn_results = child_transition_fn(unexpanded_client_state_index_pairs)
                for i, (child_client_state, child_is_leaf) in zip(unexpanded_indices, client_fn_results):
                    (node, _), child_index = node_state_action_index_pairs[i]
                    child_node = node.children[child_index]
                    if node.children[child_index] is None:
                        # Usual case: the node has not been created before
                        # This condition is only unmet in memory aggressive case where the child node exists but
                        # the client state is no longer there.
                        child_node = _TrieNode(parent=node, index_in_parent=child_index)
                        if child_is_leaf:
                            child_node.mark_leaf()
                        node.children[child_index] = child_node

                    node.sbs_child_state_cache[child_index] = (child_client_state, child_is_leaf)
                    results[i] = ((child_node, child_client_state), child_is_leaf)

            if memory_aggressive:
                # for each node from which we have transitioned, remove its client state from the parent
                for (node, _), _ in node_state_action_index_pairs:
                    if node.parent is not None:
                        node.parent.sbs_child_state_cache[node.index_in_parent] = None

            return typing.cast(List[Tuple[Tuple[_TrieNode, sbs.State], bool]], results)

        return wrapper_child_transition_fn
