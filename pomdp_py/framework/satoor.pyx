# cython: profile=True

from pomdp_py.framework.basics cimport State, Action, Observation, \
    TransitionModel, RewardModel, ObservationModel
from pomdp_py.algorithms.ba_po_uct cimport RolloutPolicy, HeuristicFunction, ActionPrior

import random

import networkx as nx
import numpy as np


# ============================================================================
# MDPState
# ============================================================================
cdef class MDPState(State):
    def __init__(self, node_id, position=None, bint is_frontier=False):
        """Initialize MDP state.

        Args:
            node_id: NodeSymbol for any node (known or predicted)
            position: Position array (2D or 3D)
            is_frontier: Whether this is a frontier node
        """
        self.node_id = node_id
        self.position = position
        self.is_frontier = is_frontier
        self._hash = hash(node_id)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, MDPState):
            return self.node_id == (<MDPState>other).node_id
        return False

    def __str__(self):
        return f"State({self.node_id.str(True)})"

    def __repr__(self):
        return f"MDPState({self.node_id.str(True)}, frontier={self.is_frontier})"


# ============================================================================
# MDPAction
# ============================================================================
cdef class MDPAction(Action):
    def __init__(self, target_node_id):
        """Initialize action.

        Args:
            target_node_id: Target NodeSymbol
        """
        self.target_node_id = target_node_id
        self._hash = hash(target_node_id)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, MDPAction):
            return self.target_node_id == (<MDPAction>other).target_node_id
        return False

    def __str__(self):
        return f"goto_{self.target_node_id.str(True)}"

    def __repr__(self):
        return f"MDPAction({self.target_node_id.str(True)})"


# ============================================================================
# MDPObservation
# ============================================================================
cdef class MDPObservation(Observation):
    def __init__(self, node_id):
        """Initialize observation.

        Args:
            node_id: Observed NodeSymbol
        """
        self.node_id = node_id
        self._hash = hash(node_id)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, MDPObservation):
            return self.node_id == (<MDPObservation>other).node_id
        return False

    def __str__(self):
        return f"Obs({self.node_id.str(True)})"

    def __repr__(self):
        return f"MDPObservation({self.node_id.str(True)})"


# ============================================================================
# MDPTransitionModel
# ============================================================================
cdef class MDPTransitionModel(TransitionModel):
    def __init__(self, mdp_graph, goal_node_id):
        """Initialize transition model.

        Args:
            mdp_graph: NetworkX Graph with nodes and weighted edges
            goal_node_id: Goal node ID (NodeSymbol)
        """
        self.mdp = mdp_graph
        self.goal_node_id = goal_node_id
        self._adjacency = self._build_adjacency()
        self._all_states, self._state_cache = self._build_all_states()

    def _build_adjacency(self):
        """Build adjacency map from MDP graph."""
        cdef dict adjacency = {}
        for node_id in self.mdp.nodes():
            neighbors = set(self.mdp.neighbors(node_id))
            adjacency[node_id] = neighbors
        return adjacency

    def _build_all_states(self):
        """Build list of all states and cache from MDP graph."""
        cdef list states = []
        cdef dict state_cache = {}
        for node_id in self.mdp.nodes():
            position = self.mdp.nodes[node_id].get("position", None)
            state = MDPState(node_id, position=position, is_frontier=False)
            states.append(state)
            state_cache[node_id] = state
        return states, state_cache

    def probability(self, next_state, state, action):
        """Return probability of transitioning to next_state."""
        cdef object state_node_id = (<MDPState>state).node_id
        cdef object action_target = (<MDPAction>action).target_node_id

        if state_node_id == self.goal_node_id:
            return 1.0 if (<MDPState>next_state).node_id == state_node_id else 0.0

        if action_target != (<MDPState>next_state).node_id:
            return 0.0

        cdef set neighbors = <set>self._adjacency.get(state_node_id, set())
        if action_target in neighbors:
            return 1.0
        return 0.0

    def sample(self, state, action):
        """Sample next state given state and action."""
        cdef object state_node_id = (<MDPState>state).node_id
        cdef object action_target = (<MDPAction>action).target_node_id

        # Terminal state: stay at goal
        if state_node_id == self.goal_node_id:
            return state

        # Check if action is valid
        cdef set neighbors = <set>self._adjacency.get(state_node_id)
        if neighbors is not None and action_target in neighbors:
            return <MDPState>self._state_cache[action_target]

        return state

    def get_all_states(self):
        """Return all possible states."""
        return self._all_states


# ============================================================================
# MDPRewardModel
# ============================================================================
cdef class MDPRewardModel(RewardModel):
    def __init__(self, mdp_graph, goal_node_id, double goal_reward=100.0,
                 knowledge=None, double navigation_failure_penalty=-1.0,
                 double door_failure_penalty=-1.0):
        """Initialize reward model.

        Args:
            mdp_graph: NetworkX Graph with weighted edges
            goal_node_id: Goal node ID (NodeSymbol)
            goal_reward: Bonus reward for reaching goal
            knowledge: KnowledgeTracker for checking door edges (optional)
            navigation_failure_penalty: Penalty for navigation failure
            door_failure_penalty: Penalty for door failure
        """
        self.mdp = mdp_graph
        self.goal_node_id = goal_node_id
        self.goal_reward = goal_reward
        self.knowledge = knowledge
        self.navigation_failure_penalty = navigation_failure_penalty
        self.door_failure_penalty = door_failure_penalty
        self._reward_cache = self._precompute_rewards()

    def _precompute_rewards(self):
        """Pre-compute rewards for all (state, action, next_state) combinations."""
        cdef dict cache = {}
        cdef double edge_weight, reward

        for node_id in self.mdp.nodes():
            if node_id == self.goal_node_id:
                for neighbor_id in self.mdp.neighbors(node_id):
                    cache[(node_id, neighbor_id, node_id)] = 0.0
                continue

            for neighbor_id in self.mdp.neighbors(node_id):
                edge_weight = self.mdp[node_id][neighbor_id].get("weight", 1.0)
                reward = -edge_weight
                if neighbor_id == self.goal_node_id:
                    reward += self.goal_reward
                cache[(node_id, neighbor_id, neighbor_id)] = reward

                if self.knowledge is not None and self.knowledge.is_door_edge(
                    node_id, neighbor_id
                ):
                    cache[(node_id, neighbor_id, node_id)] = self.door_failure_penalty
                else:
                    cache[(node_id, neighbor_id, node_id)] = (
                        self.navigation_failure_penalty
                    )

        return cache

    def _compute_edge_weight(self, state, next_state):
        """Get edge weight from MDP graph."""
        if self.mdp.has_edge((<MDPState>state).node_id, (<MDPState>next_state).node_id):
            return self.mdp[(<MDPState>state).node_id][(<MDPState>next_state).node_id].get("weight", 1.0)
        return 0.0

    def _reward_func(self, state, action, next_state):
        """Compute reward for transition."""
        cdef object state_node_id = (<MDPState>state).node_id
        cdef object next_state_node_id = (<MDPState>next_state).node_id

        if state_node_id == self.goal_node_id:
            return 0.0

        if state_node_id == next_state_node_id:
            target = (<MDPAction>action).target_node_id
            if self.knowledge is not None and self.knowledge.is_door_edge(
                state_node_id, target
            ):
                return self.door_failure_penalty
            else:
                return self.navigation_failure_penalty

        cdef double edge_weight = self._compute_edge_weight(state, next_state)
        cdef double reward = -edge_weight

        if next_state_node_id == self.goal_node_id:
            reward += self.goal_reward

        return reward

    def sample(self, state, action, next_state):
        """Sample reward (deterministic, uses pre-computed cache)."""
        return self._reward_cache.get(
            ((<MDPState>state).node_id,
             (<MDPAction>action).target_node_id,
             (<MDPState>next_state).node_id),
            0.0,
        )


# ============================================================================
# MDPObservationModel
# ============================================================================
cdef class MDPObservationModel(ObservationModel):
    def __init__(self):
        pass

    def probability(self, observation, next_state, action):
        """Probability of observation given next_state."""
        return 1.0 if (<MDPObservation>observation).node_id == (<MDPState>next_state).node_id else 0.0

    def sample(self, next_state, action):
        """Sample observation (deterministic)."""
        return MDPObservation((<MDPState>next_state).node_id)


# ============================================================================
# MDPHeuristic
# ============================================================================
cdef class MDPHeuristic(HeuristicFunction):
    """Euclidean distance heuristic for MDP navigation."""

    def __init__(self, mdp_graph, goal_node_id):
        """Initialize Euclidean distance heuristic.

        Args:
            mdp_graph: NetworkX graph with node positions
            goal_node_id: ID of the goal node
        """
        self.mdp = mdp_graph
        self.goal_node_id = goal_node_id
        self.goal_position = np.array(
            mdp_graph.nodes[goal_node_id].get("position", [0, 0])
        )
        self._distance_cache = self._precompute_distances()

    def _precompute_distances(self):
        """Pre-compute Euclidean distances from all nodes to goal."""
        cdef dict distances = {}
        for node_id in self.mdp.nodes():
            if node_id == self.goal_node_id:
                distances[node_id] = 0.0
            else:
                pos = np.array(self.mdp.nodes[node_id].get("position", [0, 0]))
                distances[node_id] = float(np.linalg.norm(self.goal_position - pos))
        return distances

    cpdef float value(self, State state):
        """Compute heuristic value (negative Euclidean distance to goal)."""
        if (<MDPState>state).node_id == self.goal_node_id:
            return 0.0
        return -self._distance_cache.get((<MDPState>state).node_id, 0.0)


# ============================================================================
# MDPActionPrior
# ============================================================================
cdef class MDPActionPrior(ActionPrior):
    """Action prior that initializes Q-values using Value Iteration."""

    def __init__(self, vi_values, MDPTransitionModel transition_model,
                 MDPRewardModel reward_model, double discount_factor):
        """Initialize action prior with precomputed VI values.

        Args:
            vi_values: dict mapping MDPState -> V*(s) from Value Iteration
            transition_model: MDPTransitionModel
            reward_model: MDPRewardModel
            discount_factor: Discount factor gamma
        """
        self.vi_values = vi_values
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.discount_factor = discount_factor

    cpdef get_preferred_actions(self, State state, tuple history):
        """Return all valid actions with VI-initialized Q-values."""
        cdef set preferences = set()
        cdef double reward, v_next, value

        for neighbor_id in self.transition_model._adjacency.get((<MDPState>state).node_id, set()):
            action = MDPAction(neighbor_id)
            next_state = self.transition_model.sample(state, action)
            reward = self.reward_model.sample(state, action, next_state)
            v_next = self.vi_values.get(next_state, 0.0)
            value = reward + self.discount_factor * v_next
            preferences.add((action, 1, value))
        return preferences


# ============================================================================
# MDPPolicyModel
# ============================================================================
cdef class MDPPolicyModel(RolloutPolicy):
    """Policy model providing random rollout for MCTS."""

    def __init__(self, MDPTransitionModel transition_model):
        """Initialize policy model.

        Args:
            transition_model: MDPTransitionModel instance
        """
        self.transition_model = transition_model
        self.valid_actions_map = self._build_valid_actions_map()

        cdef set all_actions = set()
        for actions in self.valid_actions_map.values():
            all_actions.update(actions)
        self._all_actions = list(all_actions)

    def _build_valid_actions_map(self):
        """Build map of valid actions for each state."""
        cdef dict valid_actions = {}
        for node_id, neighbors in self.transition_model._adjacency.items():
            actions = [MDPAction(neighbor_id) for neighbor_id in neighbors]
            valid_actions[node_id] = actions
        return valid_actions

    def sample(self, state):
        """Sample a random valid action."""
        cdef list valid_actions = <list>self.valid_actions_map.get(
            (<MDPState>state).node_id, [])
        if not valid_actions:
            return MDPAction((<MDPState>state).node_id)
        return random.choice(valid_actions)

    cpdef Action rollout(self, State state, tuple history):
        """Rollout policy."""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        """Get all valid actions for a state."""
        if state is None:
            return self._all_actions
        return self.valid_actions_map.get((<MDPState>state).node_id, [])


# ============================================================================
# MDPShortestPathPolicyModel
# ============================================================================
cdef class MDPShortestPathPolicyModel(RolloutPolicy):
    """Epsilon-greedy shortest path rollout policy for the MDP graph."""

    def __init__(self, MDPTransitionModel transition_model, goal_node_id,
                 double epsilon=0.1):
        self.transition_model = transition_model
        self.goal_node_id = goal_node_id
        self.epsilon = epsilon
        self.valid_actions_map = self._build_valid_actions_map()

        cdef set all_actions = set()
        for actions in self.valid_actions_map.values():
            all_actions.update(actions)
        self._all_actions = list(all_actions)
        self.next_actions = self._compute_shortest_path_actions()

    def _build_valid_actions_map(self):
        """Build map of valid actions for each state."""
        cdef dict valid_actions = {}
        for node_id, neighbors in self.transition_model._adjacency.items():
            actions = [MDPAction(neighbor_id) for neighbor_id in neighbors]
            valid_actions[node_id] = actions
        return valid_actions

    def _compute_shortest_path_actions(self):
        """Compute next-hop action toward goal for each MDP node using Dijkstra."""
        cdef dict next_actions = {}
        mdp_graph = self.transition_model.mdp

        try:
            _, paths = nx.single_source_dijkstra(
                mdp_graph, self.goal_node_id, weight="weight"
            )
        except Exception:
            return next_actions

        for node_id, path in paths.items():
            if node_id == self.goal_node_id:
                next_actions[node_id] = self.goal_node_id
            elif len(path) >= 2:
                reversed_path = list(reversed(path))
                next_actions[node_id] = (
                    reversed_path[1]
                    if len(reversed_path) >= 2
                    else self.goal_node_id
                )
            else:
                raise ValueError(
                    f"Dijkstra returned path of length {len(path)} for non-goal node {node_id}. "
                    "This should never happen."
                )
        return next_actions

    def sample(self, state):
        """Epsilon-greedy: shortest path with (1-eps), random with eps."""
        cdef object state_node_id = (<MDPState>state).node_id

        if state_node_id == self.goal_node_id:
            return MDPAction(self.goal_node_id)

        cdef list valid_actions = <list>self.valid_actions_map.get(state_node_id, [])
        if not valid_actions:
            return MDPAction(state_node_id)

        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(valid_actions)

        next_node = self.next_actions.get(state_node_id)
        if next_node is not None:
            action = MDPAction(next_node)
            if action in valid_actions:
                return action

        return random.choice(valid_actions)

    cpdef Action rollout(self, State state, tuple history):
        """Rollout policy (epsilon-greedy shortest path)."""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        """Get all valid actions for a state."""
        if state is None:
            return self._all_actions
        return self.valid_actions_map.get((<MDPState>state).node_id, [])
