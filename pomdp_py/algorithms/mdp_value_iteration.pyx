"""Infinite-horizon MDP Value Iteration.

This implements standard infinite-horizon value iteration for fully observable MDPs.
Unlike the POMDP value iteration (which builds policy trees), this directly computes
a value function V(s) over states using Bellman backups.

The algorithm iterates:
    V_{k+1}(s) = max_a Σ_{s'} T(s'|s,a)[R(s,a,s') + γV_k(s')]

until convergence: ||V_{k+1} - V_k|| < ε
"""

# Cython imports - these are compile-time imports from other .pxd files
from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Action, State

# Python imports - regular runtime imports
import numpy as np


cdef class MDPValueIteration(Planner):
    """Infinite-horizon MDP Value Iteration.

    Computes optimal value function V*(s) and derives greedy policy π*(s).
    """

    def __init__(self, discount_factor=0.99, epsilon=1e-6, max_iterations=1000):
        """Initialize MDP Value Iteration planner.

        Args:
            discount_factor: Discount factor γ (default: 0.99). Must be in [0, 1).
            epsilon: Convergence threshold (default: 1e-6). Stops when max|V_{k+1}(s) - V_k(s)| < ε.
            max_iterations: Maximum number of iterations (default: 1000). Safety limit.
        """
        # Validate parameters
        if not (0 <= discount_factor < 1):
            raise ValueError("discount_factor must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        # Store parameters in C-level variables
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._max_iterations = max_iterations

        # Initialize value function as empty dict
        self._V = {}

        # Initialize tracking variables
        self._last_num_iterations = 0
        self._last_delta = 0.0

        # Sparse transition cache: dict[(state, action)] -> list[(next_state, prob, reward)]
        # This avoids O(|S|²) loops by only storing reachable transitions
        self._transition_cache = {}

    @property
    def discount_factor(self):
        """Get discount factor γ."""
        return self._discount_factor

    @property
    def epsilon(self):
        """Get convergence threshold ε."""
        return self._epsilon

    @property
    def max_iterations(self):
        """Get maximum iterations limit."""
        return self._max_iterations

    @property
    def num_iterations(self):
        """Get number of iterations performed in last computation."""
        return self._last_num_iterations

    @property
    def convergence_delta(self):
        """Get final convergence delta from last computation."""
        return self._last_delta

    @property
    def value_function(self):
        """Get computed value function V(s) as a dict."""
        return self._V.copy()

    cpdef _build_transition_cache(self, Agent agent):
        """Build sparse transition cache to avoid O(|S|²) loops.

        For each (state, action) pair, cache only reachable next_states with their
        transition probabilities and rewards. This dramatically speeds up VI for
        large sparse MDPs (e.g., grid worlds, graphs).

        Args:
            agent: Agent with transition_model, reward_model, and all_states
        """
        cdef list states, actions
        cdef State state, next_state
        cdef Action action
        cdef float trans_prob, reward
        cdef list transitions
        cdef int total_transitions = 0

        print("    Building sparse transition cache...")

        states = list(agent.all_states)
        self._transition_cache = {}

        for state in states:
            actions = agent.policy_model.get_all_actions(state=state)

            for action in actions:
                transitions = []

                # Query transition probabilities for all states
                for next_state in states:
                    trans_prob = agent.transition_model.probability(
                        next_state, state, action
                    )

                    # Only cache transitions with non-zero probability
                    if trans_prob >= 1e-10:
                        reward = agent.reward_model.sample(state, action, next_state)
                        transitions.append((next_state, trans_prob, reward))
                        total_transitions += 1

                # Store transitions for this (state, action) pair
                if transitions:
                    self._transition_cache[(state, action)] = transitions

        print(f"    Cached {total_transitions} non-zero transitions (avg {total_transitions / len(self._transition_cache):.1f} per (s,a))")

    cpdef _compute_value_function(self, Agent agent):
        """Compute optimal value function V*(s) via Bellman iteration.

        This performs the core value iteration algorithm:
            V_{k+1}(s) = max_a Σ_{s'} T(s'|s,a)[R(s,a,s') + γV_k(s')]

        Uses sparse transition cache for O(|T|) complexity per iteration instead of O(|S|²|A|).

        Args:
            agent: Agent with transition_model, reward_model, and all_states
        """
        # Declare C-level variables for performance
        cdef list states          # All states in the MDP
        cdef State state          # Current state in outer loop
        cdef State next_state     # Next state in transition loop
        cdef list actions         # Available actions for current state
        cdef Action action        # Current action
        cdef float q_value        # Q(s,a) value
        cdef float max_q          # max_a Q(s,a)
        cdef float trans_prob     # T(s'|s,a)
        cdef float reward         # R(s,a,s')
        cdef float delta          # |V_new(s) - V(s)|
        cdef float max_delta      # max_s |V_new(s) - V(s)|
        cdef int iteration        # Iteration counter
        cdef dict V_new           # V_{k+1}
        cdef list transitions     # Cached transitions for (s,a)
        cdef tuple transition     # Single transition tuple

        # Get all states from the agent's transition model
        states = list(agent.all_states)

        # Step 1: Build sparse transition cache (one-time cost)
        if not self._transition_cache:
            self._build_transition_cache(agent)

        # Step 2: Initialize V(s) = 0 for all states
        self._V = {}
        for state in states:
            self._V[state] = 0.0

        # Step 3: Value iteration loop
        for iteration in range(self._max_iterations):
            V_new = {}
            max_delta = 0.0

            # Bellman backup for each state
            for state in states:
                # Get valid actions for this state
                actions = agent.policy_model.get_all_actions(state=state)

                if not actions:
                    # No actions available (e.g., terminal state)
                    V_new[state] = 0.0
                    continue

                # Compute Q(s,a) for each action and take max
                max_q = float('-inf')

                for action in actions:
                    # Q(s,a) = Σ_{s'} T(s'|s,a)[R(s,a,s') + γV(s')]
                    q_value = 0.0

                    # Use cached transitions instead of looping over all states
                    transitions = self._transition_cache.get((state, action), [])

                    for transition in transitions:
                        next_state = transition[0]
                        trans_prob = transition[1]
                        reward = transition[2]

                        # Add contribution to Q-value
                        q_value += trans_prob * (
                            reward + self._discount_factor * self._V[next_state]
                        )

                    # Track maximum Q-value
                    if q_value > max_q:
                        max_q = q_value

                # Update value: V_{k+1}(s) = max_a Q(s,a)
                V_new[state] = max_q

                # Track convergence
                delta = abs(V_new[state] - self._V[state])
                if delta > max_delta:
                    max_delta = delta

            # Update value function
            self._V = V_new
            self._last_num_iterations = iteration + 1
            self._last_delta = max_delta

            # Print progress every iteration
            print(f"    Iteration {iteration + 1}: max_delta={max_delta:.6f}")

            # Check convergence
            if max_delta < self._epsilon:
                print(f"    Converged!")
                break

    cpdef public plan(self, Agent agent):
        """Select greedy action according to current value function.

        This is the main public interface. It returns the action that maximizes
        Q(s,a) = Σ_{s'} T(s'|s,a)[R(s,a,s') + γV(s')] for the current state.

        If the value function hasn't been computed yet, this will compute it first.

        Args:
            agent: Agent with current belief/state

        Returns:
            Action: Greedy action for current state
        """
        # Declare C-level variables
        cdef State state, next_state
        cdef list actions, transitions
        cdef Action action, best_action
        cdef float q_value, best_q
        cdef float trans_prob, reward
        cdef tuple transition

        # Compute value function if not already done
        if not self._V:
            self._compute_value_function(agent)

        # Get current state from agent's belief
        # For MDPs, belief should be point mass on true state
        if hasattr(agent.cur_belief, 'mpe'):
            state = agent.cur_belief.mpe()
        else:
            # Assuming belief is Histogram, get most probable state
            # Find state with highest probability
            best_state = None
            best_prob = float('-inf')
            for s in agent.cur_belief:
                prob = agent.cur_belief[s]
                if prob > best_prob:
                    best_prob = prob
                    best_state = s
            state = best_state

        # Get valid actions
        actions = agent.policy_model.get_all_actions(state=state)

        if not actions:
            raise ValueError(f"No actions available for state {state}")

        # Find action with maximum Q-value
        best_action = None
        best_q = float('-inf')

        for action in actions:
            # Compute Q(s,a) = Σ_{s'} T(s'|s,a)[R(s,a,s') + γV(s')]
            q_value = 0.0

            # Use cached transitions instead of looping over all states
            transitions = self._transition_cache.get((state, action), [])

            for transition in transitions:
                next_state = transition[0]
                trans_prob = transition[1]
                reward = transition[2]

                # Look up V(s') from computed value function
                q_value += trans_prob * (
                    reward + self._discount_factor * self._V.get(next_state, 0.0)
                )

            if q_value > best_q:
                best_q = q_value
                best_action = action

        return best_action
