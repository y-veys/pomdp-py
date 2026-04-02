"""Fast array-based Value Iteration for constrained BAMDPs.

Exploits the constrained BAMDP structure where each (state, action) has at most
2 transitions: success (→ target) with probability p, failure (→ self-loop)
with probability 1-p.

The graph structure is built once from the transition/reward models and reused.
Only the success probabilities change between calls, passed as a flat array.

Usage:
    fast_vi = FastVI(transition_model, reward_model, policy_model,
                     transition_beliefs, discount_factor=0.95)
    fast_vi.solve(p_success_array)
    action_node_id = fast_vi.get_action_node_id(current_node_id)
"""

import numpy as np

from pomdp_py.framework.satoor import MDPState, MDPAction


cdef class FastVI:
    """Fast Value Iteration using flat arrays — no Python dicts in inner loop.

    Built from standard POMDP model objects. The graph structure is extracted
    once and stored as flat arrays. The solve() method takes a flat array of
    success probabilities for uncertain transitions and runs VI using pure
    C-level array indexing.
    """

    def __init__(self, transition_model, reward_model, policy_model,
                 dict transition_beliefs,
                 double discount_factor=0.95, double epsilon=1e-6,
                 int max_iterations=100):
        """Build FastVI from POMDP model objects.

        Args:
            transition_model: MDPTransitionModel with _adjacency and goal_node_id
            reward_model: MDPRewardModel with _reward_cache
            policy_model: MDPPolicyModel with valid_actions_map
            transition_beliefs: dict {(MDPState, MDPAction, MDPState) -> TransitionBelief}
                Defines which transitions are uncertain. Only the keys are used
                (to build the mapping); belief values are not read.
            discount_factor: Discount factor γ
            epsilon: Convergence threshold
            max_iterations: Max VI iterations
        """
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        adjacency = transition_model._adjacency
        goal_id = transition_model.goal_node_id
        reward_cache = reward_model._reward_cache

        # Build node_id → index mapping (use list() instead of sorted()
        # because node_ids may not support < comparison, e.g. NodeSymbol)
        node_ids = list(adjacency.keys())
        cdef int n = len(node_ids)
        self.num_states = n
        self.idx_to_node_id = node_ids
        self.node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Terminal states
        self.state_is_terminal = np.zeros(n, dtype=np.intc)
        for i, nid in enumerate(node_ids):
            if nid == goal_id:
                self.state_is_terminal[i] = 1

        # Count (state, action) pairs
        cdef int total_sa = 0
        for nid in node_ids:
            if nid == goal_id:
                continue
            neighbors = adjacency.get(nid, set())
            total_sa += len(neighbors)
        self.num_sa_pairs = total_sa

        # Build uncertain transition key → index mapping
        # transition_beliefs keys are (MDPState, MDPAction, MDPState) tuples
        uncertain_keys = list(transition_beliefs.keys())
        self.num_uncertain = len(uncertain_keys)
        self.belief_key_to_idx = {}
        uncertain_by_nodes = {}
        for idx, key in enumerate(uncertain_keys):
            self.belief_key_to_idx[key] = idx
            s_nid = key[0].node_id
            t_nid = key[2].node_id
            uncertain_by_nodes[(s_nid, t_nid)] = idx

        # Allocate arrays
        self.sa_state = np.zeros(total_sa, dtype=np.intc)
        self.sa_target = np.zeros(total_sa, dtype=np.intc)
        self.r_success = np.zeros(total_sa, dtype=np.float64)
        self.r_failure = np.zeros(total_sa, dtype=np.float64)
        self.sa_belief_idx = np.full(total_sa, -1, dtype=np.intc)
        self.state_sa_start = np.zeros(n, dtype=np.intc)
        self.state_sa_count = np.zeros(n, dtype=np.intc)
        self.sa_to_action_node_id = []

        # Fill arrays
        cdef int sa_idx = 0
        for s_idx in range(n):
            nid = node_ids[s_idx]
            self.state_sa_start[s_idx] = sa_idx

            if nid == goal_id:
                self.state_sa_count[s_idx] = 0
                continue

            neighbors = list(adjacency.get(nid, set()))
            self.state_sa_count[s_idx] = len(neighbors)

            for target_nid in neighbors:
                t_idx = self.node_id_to_idx[target_nid]
                self.sa_state[sa_idx] = s_idx
                self.sa_target[sa_idx] = t_idx
                self.sa_to_action_node_id.append(target_nid)

                # Rewards from precomputed cache: (node_id, target_nid, next_node_id)
                r_succ = reward_cache.get((nid, target_nid, target_nid), 0.0)
                r_fail = reward_cache.get((nid, target_nid, nid), 0.0)
                self.r_success[sa_idx] = r_succ
                self.r_failure[sa_idx] = r_fail

                # Check if this transition is uncertain
                ukey = (nid, target_nid)
                if ukey in uncertain_by_nodes:
                    self.sa_belief_idx[sa_idx] = uncertain_by_nodes[ukey]

                sa_idx += 1

        # Allocate solve buffers
        self.V = np.zeros(n, dtype=np.float64)
        self.policy = np.zeros(n, dtype=np.intc)

    cpdef solve(self, double[:] p_success):
        """Solve the MDP with given success probabilities.

        Args:
            p_success: Array of length num_uncertain, indexed by belief_key_to_idx.
                       p_success[i] is the success probability for uncertain transition i.
        """
        cdef int n = self.num_states
        cdef int sa_start, sa_count, sa_idx, bi, best_sa
        cdef double q, max_q, delta, max_delta, p, gamma
        cdef int s, iteration

        gamma = self.discount_factor

        # Initialize V = 0
        for s in range(n):
            self.V[s] = 0.0

        for iteration in range(self.max_iterations):
            max_delta = 0.0

            for s in range(n):
                if self.state_is_terminal[s]:
                    continue

                sa_start = self.state_sa_start[s]
                sa_count = self.state_sa_count[s]

                if sa_count == 0:
                    continue

                max_q = -1e30
                best_sa = sa_start

                for sa_idx in range(sa_start, sa_start + sa_count):
                    bi = self.sa_belief_idx[sa_idx]
                    if bi >= 0:
                        p = p_success[bi]
                    else:
                        p = 1.0

                    q = (p * (self.r_success[sa_idx] + gamma * self.V[self.sa_target[sa_idx]])
                         + (1.0 - p) * (self.r_failure[sa_idx] + gamma * self.V[s]))

                    if q > max_q:
                        max_q = q
                        best_sa = sa_idx

                delta = max_q - self.V[s]
                if delta < 0:
                    delta = -delta
                if delta > max_delta:
                    max_delta = delta

                self.V[s] = max_q
                self.policy[s] = best_sa

            if max_delta < self.epsilon:
                break

    cpdef int get_action_node_id(self, object state_node_id):
        """Get the optimal action (as target node_id) for a given state.

        Args:
            state_node_id: The node_id of the current state

        Returns:
            Target node_id of the optimal action
        """
        cdef int s_idx = self.node_id_to_idx[state_node_id]
        cdef int sa_idx = self.policy[s_idx]
        return self.sa_to_action_node_id[sa_idx]

    def get_policy_dict(self):
        """Return policy as {MDPState -> MDPAction} dict for compatibility."""
        cdef dict policy = {}
        cdef int s, sa_idx
        for s in range(self.num_states):
            if self.state_is_terminal[s]:
                continue
            if self.state_sa_count[s] == 0:
                continue
            sa_idx = self.policy[s]
            nid = self.idx_to_node_id[s]
            target_nid = self.sa_to_action_node_id[sa_idx]
            policy[MDPState(nid)] = MDPAction(target_nid)
        return policy

    def get_value_dict(self):
        """Return value function as {MDPState -> float} dict for compatibility."""
        cdef dict values = {}
        cdef int s
        for s in range(self.num_states):
            nid = self.idx_to_node_id[s]
            values[MDPState(nid)] = self.V[s]
        return values
