# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
"""Deep Sparse Sampling (DSS) for Bayesian Reinforcement Learning.

Implements the DSS algorithm from:
    Grover, Basu, Dimitrakakis. "Bayesian Reinforcement Learning via Deep,
    Sparse Sampling." AISTATS 2020.

DSS plans at the level of K-step policies instead of individual actions,
reducing the effective branching factor from |A| to N (number of sampled
policies). It samples N MDPs from beliefs (Thompson sampling), solves each
with Value Iteration, then evaluates those policies by running them forward
K steps in the BAMDP.

This version uses integer-indexed flat arrays for the simulation loop,
reusing FastVI's graph structure to avoid Python dict lookups and __hash__
calls.
"""

from pomdp_py.framework.basics cimport Action, Agent, State, Observation, \
    TransitionModel, TransitionBelief
from pomdp_py.framework.planner cimport Planner
from pomdp_py.algorithms.fast_vi cimport FastVI

from libc.string cimport memcpy
from libc.stdlib cimport rand, RAND_MAX

import random
import time

import numpy as np

import pomdp_py
from pomdp_py.framework.satoor import MDPState, MDPAction


# ============================================================================
# DSS Planner
# ============================================================================
cdef class DSS(Planner):
    """Deep Sparse Sampling planner for Bayes-Adaptive MDPs.

    Instead of building an MCTS tree over individual actions, DSS branches
    on K-step policies sampled via Thompson sampling + Value Iteration.

    Uses integer-indexed arrays from FastVI for the simulation loop,
    avoiding Python dict lookups and __hash__ calls entirely.

    Parameters:
        max_depth: Total planning horizon T = H * K
        num_policies: N, number of policies sampled per DSS node
        num_samples: M, Monte Carlo samples per policy evaluation
        steps_per_policy: K, steps each policy runs before re-branching
        discount_factor: Discount factor gamma
    """

    def __init__(self, max_depth=20, num_policies=5, num_samples=10,
                 steps_per_policy=5, discount_factor=0.9,
                 show_progress=False, debug=False):
        self._max_depth = max_depth
        self._num_policies = num_policies
        self._num_samples = num_samples
        self._steps_per_policy = steps_per_policy
        self._discount_factor = discount_factor
        self._show_progress = show_progress
        self._debug = debug

        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1
        self._last_action_values = {}

    @property
    def last_num_sims(self):
        return self._last_num_sims

    @property
    def last_planning_time(self):
        return self._last_planning_time

    cpdef public plan(self, Agent agent):
        """Plan and return the best action for the current state.

        Args:
            agent: BayesAdaptiveAgent with transition_beliefs

        Returns:
            Action: Best action for current state
        """
        cdef double start_time, time_taken
        cdef float best_q
        cdef int state_idx, sa_idx, nu, nu2, i
        cdef dict action_values
        cdef TransitionBelief tb

        self._agent = agent
        start_time = time.time()

        # Build FastVI once per planning call (reused across all VI solves)
        if self._fast_vi is None:
            self._fast_vi = FastVI(
                agent.transition_model,
                agent.reward_model,
                agent.policy_model,
                agent.transition_beliefs,
                discount_factor=self._discount_factor,
            )
            nu = self._fast_vi.num_uncertain
            self._p_success_buf = np.zeros(nu, dtype=np.float64)
            # Store ordered belief keys matching FastVI's indexing
            self._belief_keys = [None] * nu
            for key, idx in self._fast_vi.belief_key_to_idx.items():
                self._belief_keys[idx] = key

            # Allocate belief buffers
            self._alpha_buf = np.zeros(nu, dtype=np.float64)
            self._beta_buf = np.zeros(nu, dtype=np.float64)
            self._strength_buf = np.zeros(nu, dtype=np.float64)
            self._alpha_scratch = np.zeros(nu, dtype=np.float64)
            self._beta_scratch = np.zeros(nu, dtype=np.float64)

            # Fill strength buffer (constant across calls)
            for i in range(nu):
                key = self._belief_keys[i]
                belief = <TransitionBelief>agent.transition_beliefs[key]
                self._strength_buf[i] = belief.update_strength

        # Get current state and convert to index
        state = agent.sample_belief()
        state_idx = self._fast_vi.node_id_to_idx[state.node_id]

        if self._debug:
            print(f"\n{'='*70}")
            print(f"DSS: Planning from state {state}")
            print(f"  N={self._num_policies} policies, M={self._num_samples} samples, "
                  f"K={self._steps_per_policy} steps, max_depth={self._max_depth}")
            print(f"{'='*70}")

        # Copy beliefs into flat arrays
        nu2 = self._fast_vi.num_uncertain
        for i in range(nu2):
            key = self._belief_keys[i]
            tb = <TransitionBelief>agent.transition_beliefs[key]
            self._alpha_buf[i] = tb.alpha
            self._beta_buf[i] = tb.beta

        # Run DSS at root
        best_q, best_policy_sa, action_values = self._dss(
            state_idx, self._alpha_buf, self._beta_buf, 0)

        time_taken = time.time() - start_time
        self._last_planning_time = time_taken
        self._last_action_values = action_values

        # Convert best policy's action for current state back to Action object
        best_action = None
        if best_policy_sa is not None:
            sa_idx = best_policy_sa[state_idx]
            # Check that the state actually has actions
            if self._fast_vi.state_sa_count[state_idx] > 0:
                target_nid = self._fast_vi.sa_to_action_node_id[sa_idx]
                best_action = MDPAction(target_nid)

        if best_action is None:
            best_action = agent.policy_model.sample(state)

        if self._debug:
            print(f"\n  Selected action: {best_action} (Q = {best_q:.2f})")
            print(f"  Planning time: {time_taken:.3f}s")

        return best_action

    cdef tuple _dss(self, int state_idx, double[:] alpha, double[:] beta, int depth):
        """Recursive DSS: sample N policies, evaluate each with M samples, return best.

        Args:
            state_idx: Current state index (into FastVI arrays)
            alpha: Current alpha beliefs (flat array, num_uncertain)
            beta: Current beta beliefs (flat array, num_uncertain)
            depth: Current depth in the planning tree

        Returns:
            tuple: (best_q, best_policy_sa, action_values)
                - best_q: float, best Q-value among sampled policies
                - best_policy_sa: int[:] policy array (state→sa_pair), or None at leaf
                - action_values: dict {Action -> float}, best Q per first action
        """
        cdef int i, m, final_depth, final_state_idx, nu
        cdef float best_q, q_value, total_reward, child_q
        cdef dict action_values
        cdef int[:] policy_sa
        cdef int[:] best_policy_sa
        cdef double[:] scratch_a
        cdef double[:] scratch_b

        if depth >= self._max_depth:
            return (0.0, None, {})

        # Check terminal state
        if self._fast_vi.state_is_terminal[state_idx]:
            return (0.0, None, {})

        nu = self._fast_vi.num_uncertain

        # Sample N policies from current beliefs
        # Store as list of int[:] policy arrays
        cdef list policies = []
        for i in range(self._num_policies):
            self._sample_and_solve_mdp(alpha, beta)
            # Copy the policy array (FastVI reuses it)
            policy_sa = np.array(self._fast_vi.policy, dtype=np.intc)
            policies.append(policy_sa)

        indent = "  " * (depth // self._steps_per_policy + 1)
        if self._debug:
            print(f"{indent}[depth={depth}] state_idx={state_idx}, {len(policies)} policies")

        # Evaluate each policy
        best_q = -1e30
        best_policy_sa = None
        action_values = {}

        for i in range(len(policies)):
            policy_sa = policies[i]
            q_value = 0.0

            for m in range(self._num_samples):
                # Copy beliefs into scratch buffers
                scratch_a = self._alpha_scratch
                scratch_b = self._beta_scratch
                memcpy(&scratch_a[0], &alpha[0], nu * sizeof(double))
                memcpy(&scratch_b[0], &beta[0], nu * sizeof(double))

                total_reward, final_state_idx, final_depth = \
                    self._run_policy_k_steps(policy_sa, state_idx,
                                             scratch_a, scratch_b, depth)

                child_q, _, _ = self._dss(final_state_idx, scratch_a, scratch_b, final_depth)
                q_value += total_reward + child_q

            q_value /= self._num_samples

            # Track best Q per first action (convert sa_idx to Action for the dict)
            sa_idx_for_state = policy_sa[state_idx]
            if self._fast_vi.state_sa_count[state_idx] > 0:
                action_nid = self._fast_vi.sa_to_action_node_id[sa_idx_for_state]
                action_key = MDPAction(action_nid)
                if action_key not in action_values or q_value > action_values[action_key]:
                    action_values[action_key] = q_value

            if self._debug:
                print(f"{indent}  Policy {i} (sa_idx: {sa_idx_for_state}): "
                      f"Q = {q_value:.2f}")

            if q_value > best_q:
                best_q = q_value
                best_policy_sa = policy_sa

        return (best_q, best_policy_sa, action_values)

    cdef void _sample_and_solve_mdp(self, double[:] alpha, double[:] beta):
        """Sample an MDP from beliefs via Thompson sampling and solve with FastVI.

        Fills self._p_success_buf from alpha/beta arrays using betavariate,
        then calls fast_vi.solve(). The policy is available as self._fast_vi.policy.
        """
        cdef int i
        cdef int nu = self._fast_vi.num_uncertain
        cdef double a, b

        # Sample success probabilities from Beta beliefs into flat array
        for i in range(nu):
            a = alpha[i]
            b = beta[i]
            self._p_success_buf[i] = random.betavariate(a, b)

        # Solve — policy stored in self._fast_vi.policy
        self._fast_vi.solve(self._p_success_buf)

    cdef tuple _run_policy_k_steps(self, int[:] policy, int s_idx,
                                   double[:] alpha, double[:] beta, int depth):
        """Run a policy for K steps in the BAMDP using integer-indexed arrays.

        _sample_ba_transition and _update_beliefs are inlined here to avoid
        function call overhead.

        Args:
            policy: int[:] array mapping state_idx → sa_pair_idx
            s_idx: Starting state index
            alpha: Alpha beliefs (mutated in place)
            beta: Beta beliefs (mutated in place)
            depth: Current depth

        Returns:
            tuple: (total_discounted_reward, final_state_idx, final_depth)
        """
        cdef double total_reward = 0.0
        cdef double discount = 1.0
        cdef int step, sa_idx, t_idx, bi
        cdef double p_success, r, a, b
        cdef double gamma = self._discount_factor
        cdef bint success

        for step in range(self._steps_per_policy):
            if depth >= self._max_depth:
                break

            # Check terminal
            if self._fast_vi.state_is_terminal[s_idx]:
                break

            # Check state has actions
            if self._fast_vi.state_sa_count[s_idx] == 0:
                break

            # Look up action from policy (C array index)
            sa_idx = policy[s_idx]

            # Get target state
            t_idx = self._fast_vi.sa_target[sa_idx]

            # Get belief index for this (state, action) pair
            bi = self._fast_vi.sa_belief_idx[sa_idx]

            if bi >= 0:
                # Uncertain transition — sample from current beliefs
                a = alpha[bi]
                b = beta[bi]
                p_success = a / (a + b)
                success = random.random() < p_success

                # Inline belief update
                if success:
                    alpha[bi] = a + self._strength_buf[bi]
                else:
                    beta[bi] = b + self._strength_buf[bi]
            else:
                # Deterministic transition
                success = True

            # Compute reward and next state
            if success:
                r = self._fast_vi.r_success[sa_idx]
                total_reward += r * discount
                discount *= gamma
                depth += 1
                s_idx = t_idx
            else:
                r = self._fast_vi.r_failure[sa_idx]
                total_reward += r * discount
                discount *= gamma
                depth += 1
                # s_idx stays the same (self-loop)

        return total_reward, s_idx, depth
