
import numpy as np
import pomdp_py
from pomdp_py.algorithms.ba_po_uct import POUCT, RandomRollout
import pytest


# ============================================================================
# State, Action, Observation definitions
# ============================================================================

class SimpleState(pomdp_py.State):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"State({self.name})"

    def __eq__(self, other):
        return isinstance(other, SimpleState) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class SimpleAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Action({self.name})"

    def __eq__(self, other):
        return isinstance(other, SimpleAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class SimpleObservation(pomdp_py.Observation):
    """For BAMDP, observations reveal the current state."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Obs({self.name})"

    def __eq__(self, other):
        return isinstance(other, SimpleObservation) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


# ============================================================================
# BAMDP Models
# ============================================================================

class BAMDPTransitionModel(pomdp_py.TransitionModel):
    """
    Transition model for the BAMDP.

    For each (state, action), returns the TARGET state (where we'd go if successful).
    The actual transition (success vs self-loop) is handled by BA-POUCT
    based on the transition beliefs.

    Transition structure:
    - (start, left) -> goal (uncertain)
    - (start, right) -> goal (deterministic)
    - (goal, *) -> goal (terminal state, self-loop)
    """

    def sample(self, state, action):
        """Return the target state for this (state, action) pair."""
        if state.name == "goal":
            return SimpleState("goal")  # Terminal state
        elif state.name == "start":
            if action.name in ["left", "right"]:
                return SimpleState("goal")  # Target is always goal
        return state  # Fallback to self-loop

    def probability(self, next_state, state, action):
        """This is not used directly in BA-POUCT simulation."""
        raise NotImplementedError("Use beliefs for probability in BAMDP")

    def argmax(self, state, action):
        return self.sample(state, action)

    def get_all_states(self):
        return [SimpleState("start"), SimpleState("goal")]

    def is_terminal(self, state):
        """Check if a state is terminal."""
        return state.name == "goal"


class BAMDPObservationModel(pomdp_py.ObservationModel):
    """
    For BAMDP, observations reveal the current state perfectly.

    Uses standard signature: sample(next_state, action)
    """

    def sample(self, next_state, action):
        """Return observation revealing the next state."""
        return SimpleObservation(next_state.name)

    def probability(self, observation, next_state, action):
        """Deterministic observation of next_state."""
        return 1.0 if observation.name == next_state.name else 0.0

    def argmax(self, next_state, action):
        return SimpleObservation(next_state.name)

    def get_all_observations(self):
        return [SimpleObservation("start"), SimpleObservation("goal")]


class BAMDPRewardModel(pomdp_py.RewardModel):
    """
    Reward model for the BAMDP.

    Costs (negative rewards):
    - Self-loop at start: -1.0 (failure cost)
    - (start, left, goal): -2.0 (low cost action)
    - (start, right, goal): -10.0 (high cost but guaranteed)
    - Reaching/staying at goal: 0.0
    """

    def sample(self, state, action, next_state):
        """Return reward for transition."""
        # Terminal state - no cost
        if next_state.name == "goal" and state.name == "goal":
            return 0.0

        # Successful transition to goal
        if state.name == "start" and next_state.name == "goal":
            if action.name == "left":
                return -2.0  # Low cost
            elif action.name == "right":
                return -10.0  # High cost

        # Self-loop (failure)
        if state.name == next_state.name and state.name == "start":
            return -1.0

        return 0.0  # Default

    def argmax(self, state, action, next_state):
        return self.sample(state, action, next_state)

    def probability(self, reward, state, action, next_state):
        """Deterministic rewards."""
        return 1.0 if reward == self.sample(state, action, next_state) else 0.0


# ============================================================================
# Policy Model for BA-POUCT
# ============================================================================

class BAMDPPolicyModel(pomdp_py.PolicyModel):
    """Policy model that defines valid actions for the BAMDP agent."""

    def __init__(self):
        self._actions = [SimpleAction("left"), SimpleAction("right")]

    def sample(self, state, **kwargs):
        """Random action selection."""
        return np.random.choice(self._actions)

    def probability(self, action, state, **kwargs):
        return 1.0 / len(self._actions)

    def argmax(self, state, **kwargs):
        return self._actions[0]

    def get_all_actions(self, **kwargs):
        return self._actions


class BAMDPRolloutPolicy(pomdp_py.algorithms.ba_po_uct.RolloutPolicy):
    """Rollout policy for BA-POUCT planner."""

    def __init__(self):
        super().__init__()
        self._actions = [SimpleAction("left"), SimpleAction("right")]

    def rollout(self, state, history):
        """Random action selection for rollouts."""
        return np.random.choice(self._actions)

    def get_all_actions(self, state=None, history=None):
        """Return all available actions."""
        return self._actions


# ============================================================================
# Bayes-Adaptive Agent
# ============================================================================

class BayesAdaptiveAgent(pomdp_py.Agent):
    """
    A Bayes-Adaptive agent that maintains beliefs over transition probabilities.

    Attributes:
        _transition_beliefs (dict): Maps (state, action, target) -> (alpha, beta)
            representing Beta distribution parameters
    """

    def __init__(self, init_belief, transition_beliefs, policy_model,
                 transition_model, observation_model, reward_model):
        super().__init__(init_belief, policy_model, transition_model,
                        observation_model, reward_model)
        self._transition_beliefs = transition_beliefs

    def transition_beliefs(self):
        """Return current beliefs over transition probabilities."""
        return self._transition_beliefs

    def update_transition_beliefs(self, new_beliefs):
        """Update the agent's transition beliefs."""
        self._transition_beliefs = new_beliefs

    def valid_actions(self, state=None, history=None):
        """Return valid actions for the given state."""
        if state is None:
            state = self.belief.mpe()

        if state.name == "goal":
            # At goal, all actions lead to self-loop
            return [SimpleAction("left"), SimpleAction("right")]
        return [SimpleAction("left"), SimpleAction("right")]


# ============================================================================
# Test Cases
# ============================================================================

def create_test_bamdp():
    """
    Create a simple 2-state, 2-action BAMDP for testing.

    Returns:
        BayesAdaptiveAgent: Configured agent with beliefs
    """
    # Initial belief - agent knows it starts at "start"
    init_belief = pomdp_py.Histogram({
        SimpleState("start"): 1.0,
        SimpleState("goal"): 0.0
    })

    # Transition beliefs: Beta(alpha, beta) over success probabilities
    # Only uncertain transitions are in this dict
    # (start, right, goal) is deterministic (not in beliefs)
    # (start, left, goal) is uncertain with prior Beta(1, 1) = Uniform[0,1]
    transition_beliefs = {
        (SimpleState("start"), SimpleAction("left"), SimpleState("goal")): (1.0, 1.0)
    }

    # Create models
    transition_model = BAMDPTransitionModel()
    observation_model = BAMDPObservationModel()
    reward_model = BAMDPRewardModel()
    policy_model = BAMDPPolicyModel()

    # Create BA agent
    agent = BayesAdaptiveAgent(
        init_belief=init_belief,
        transition_beliefs=transition_beliefs,
        policy_model=policy_model,
        transition_model=transition_model,
        observation_model=observation_model,
        reward_model=reward_model
    )

    return agent


def test_full_episode():
    """
    Run a full episode using BA-POUCT planning.

    This tests the complete pipeline:
    1. Plan with BA-POUCT
    2. Execute action
    3. Update agent belief
    4. Update planner tree
    5. Repeat until goal reached
    """
    np.random.seed(42)
    agent = create_test_bamdp()

    planner = POUCT(
        max_depth=10,
        num_sims=100,
        discount_factor=0.95,
        exploration_const=5.0,
        rollout_policy=BAMDPRolloutPolicy()
    )

    # Create environment
    env = pomdp_py.Environment(
        SimpleState("start"),
        agent.transition_model,
        agent.reward_model
    )

    total_reward = 0.0
    total_discounted_reward = 0.0
    discount = 1.0
    gamma = 0.95
    max_steps = 20

    for step in range(max_steps):
        # Plan
        action = planner.plan(agent)

        # Execute (manually sample transition for BAMDP)
        state = env.state
        target = agent.transition_model.sample(state, action)

        # Sample from beliefs
        start = SimpleState("start")
        goal = SimpleState("goal")

        if state.name == "start" and action.name == "left":
            # Uncertain transition
            alpha, beta = agent.transition_beliefs()[(start, SimpleAction("left"), goal)]
            p_success = np.random.beta(alpha, beta)
            success = np.random.uniform() < p_success
            next_state = target if success else state
        else:
            # Deterministic or terminal
            next_state = target

        # Get observation and reward
        observation = agent.observation_model.sample(next_state, action)
        reward = agent.reward_model.sample(state, action, next_state)

        # Update environment (use apply_transition method instead of direct assignment)
        env.apply_transition(next_state)

        # Update agent belief (for BAMDP, just update to new state)
        agent.set_belief(pomdp_py.Histogram({next_state: 1.0}))

        # Update planner
        planner.update(agent, action, observation)

        # Track rewards
        total_reward += reward
        total_discounted_reward += discount * reward
        discount *= gamma

        print(f"  Step {step}: {state.name} --{action.name}--> {next_state.name} (r={reward:.1f})")

        # Check termination
        if next_state.name == "goal":
            print(f"  Reached goal in {step + 1} steps!")
            break

    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total discounted reward: {total_discounted_reward:.2f}")
    print("✓ Full episode test passed")
