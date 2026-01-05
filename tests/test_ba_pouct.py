"""
Test suite for Bayes-Adaptive POUCT (BA-POUCT) algorithm.

This tests a simple 2-state, 2-action BAMDP where:
- States: ["start", "goal"]
- Actions: ["left", "right"]
- Transition dynamics:
  - (start, right, goal): Deterministic (100% success) - high cost
  - (start, left, goal): Uncertain (unknown probability) - low cost
  - Self-loops occur when transitions fail
- The agent maintains beliefs over transition probabilities using Beta distributions
"""

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


def test_agent_creation():
    """Test that we can create a BA-POUCT agent."""
    agent = create_test_bamdp()

    assert agent is not None
    assert hasattr(agent, 'transition_beliefs')
    assert len(agent.transition_beliefs()) == 1

    # Check belief structure
    start = SimpleState("start")
    left = SimpleAction("left")
    goal = SimpleState("goal")

    alpha, beta = agent.transition_beliefs()[(start, left, goal)]
    assert alpha == 1.0
    assert beta == 1.0
    print("✓ Agent creation test passed")


def test_transition_model():
    """Test that the transition model returns correct target states."""
    agent = create_test_bamdp()
    tm = agent.transition_model

    start = SimpleState("start")
    goal = SimpleState("goal")
    left = SimpleAction("left")
    right = SimpleAction("right")

    # Both actions from start should target goal
    assert tm.sample(start, left) == goal
    assert tm.sample(start, right) == goal

    # Goal is terminal
    assert tm.sample(goal, left) == goal
    assert tm.sample(goal, right) == goal
    assert tm.is_terminal(goal)
    assert not tm.is_terminal(start)

    print("✓ Transition model test passed")


def test_observation_model():
    """Test that observations reveal the state."""
    agent = create_test_bamdp()
    om = agent.observation_model

    start = SimpleState("start")
    goal = SimpleState("goal")
    left = SimpleAction("left")
    right = SimpleAction("right")

    # Observation should match next_state (standard signature: next_state, action)
    obs = om.sample(goal, left)
    assert obs.name == "goal"

    obs = om.sample(start, left)
    assert obs.name == "start"

    obs = om.sample(goal, right)
    assert obs.name == "goal"

    obs = om.sample(start, right)
    assert obs.name == "start"

    print("✓ Observation model test passed")


def test_reward_model():
    """Test that rewards are assigned correctly."""
    agent = create_test_bamdp()
    rm = agent.reward_model

    start = SimpleState("start")
    goal = SimpleState("goal")
    left = SimpleAction("left")
    right = SimpleAction("right")

    # Test costs
    assert rm.sample(start, left, goal) == -2.0  # Low cost
    assert rm.sample(start, right, goal) == -10.0  # High cost
    assert rm.sample(start, left, start) == -1.0  # Self-loop cost
    assert rm.sample(goal, left, goal) == 0.0  # Terminal

    print("✓ Reward model test passed")


def test_ba_pouct_planning():
    """Test that BA-POUCT can plan and select actions."""
    np.random.seed(42)
    agent = create_test_bamdp()

    # Create planner
    planner = POUCT(
        max_depth=10,
        num_sims=100,
        discount_factor=0.95,
        exploration_const=10.0,
        rollout_policy=BAMDPRolloutPolicy()
    )

    # Plan
    action = planner.plan(agent)

    assert action is not None
    assert isinstance(action, SimpleAction)
    assert action.name in ["left", "right"]

    print(f"✓ Planning test passed - selected action: {action.name}")


def test_ba_pouct_simulation():
    """
    Test that BA-POUCT simulation works correctly.

    This tests the core _simulate method by verifying:
    1. Tree structure is correct
    2. Visit counts make sense
    3. Q-values are bounded correctly
    4. UCB exploration works
    """
    np.random.seed(42)
    agent = create_test_bamdp()

    planner = POUCT(
        max_depth=5,
        num_sims=50,
        discount_factor=0.95,
        exploration_const=5.0,
        rollout_policy=BAMDPRolloutPolicy()
    )

    action = planner.plan(agent)

    # Check that tree was built
    assert hasattr(agent, 'tree')
    assert agent.tree is not None
    assert len(agent.tree.children) == 2, "Should have explored both actions"

    # Get Q-nodes
    left = SimpleAction("left")
    right = SimpleAction("right")

    qnode_left = agent.tree[left]
    qnode_right = agent.tree[right]

    print(f"  Action left: visits={qnode_left.num_visits}, value={qnode_left.value:.3f}")
    print(f"  Action right: visits={qnode_right.num_visits}, value={qnode_right.value:.3f}")

    # Verify visit counts
    assert qnode_left.num_visits > 0, "Left action should be visited"
    assert qnode_right.num_visits > 0, "Right action should be visited"

    # Note: Visit counts may not sum exactly to num_sims if some simulations
    # terminate early (e.g., reaching terminal state immediately)
    total_action_visits = qnode_left.num_visits + qnode_right.num_visits
    print(f"  Total action visits: {total_action_visits} (num_sims: 50)")

    # Verify root visit count matches action visits
    assert agent.tree.num_visits == total_action_visits, \
        f"Root visits ({agent.tree.num_visits}) should match sum of action visits ({total_action_visits})"

    # Verify Q-values are bounded by possible rewards
    # Best case: immediate goal with -2.0 cost
    # Worst case: multiple failures at -1.0 each, then eventual success
    assert qnode_left.value >= -10.0, "Left Q-value should be >= -10.0"
    assert qnode_left.value <= 0.0, "Left Q-value should be <= 0.0"

    assert qnode_right.value >= -20.0, "Right Q-value should be >= -20.0"
    assert qnode_right.value <= 0.0, "Right Q-value should be <= 0.0"

    # Verify that 'left' is heavily preferred (cheaper action)
    # With exploration_const=5.0 and 50 sims, left should dominate
    assert qnode_left.num_visits > qnode_right.num_visits, \
        "Left (cheaper) should be visited more than right (expensive)"

    # Verify tree has children (rollouts created VNodes)
    # At least some observations should have been explored
    num_vnodes_left = len([v for v in qnode_left.children.values() if v is not None])
    num_vnodes_right = len([v for v in qnode_right.children.values() if v is not None])

    print(f"  Left action has {num_vnodes_left} observation nodes")
    print(f"  Right action has {num_vnodes_right} observation nodes")

    # The heavily-visited left action should have more observation nodes
    assert num_vnodes_left > 0, "Left action should have at least one observation node"

    print("✓ Simulation test passed")


def test_ba_pouct_belief_update():
    """
    Test that beliefs are updated correctly during simulation.

    This tests the _update_beliefs method directly by simulating
    transitions and verifying Beta parameters update correctly.
    """
    np.random.seed(123)
    agent = create_test_bamdp()

    start = SimpleState("start")
    left = SimpleAction("left")
    goal = SimpleState("goal")

    # Initial belief for (start, left, goal) is Beta(1, 1)
    initial_beliefs = agent.transition_beliefs().copy()
    alpha_init, beta_init = initial_beliefs[(start, left, goal)]
    assert alpha_init == 1.0
    assert beta_init == 1.0

    # Test the _update_beliefs method from the planner
    planner = POUCT(
        max_depth=10,
        num_sims=1,
        discount_factor=0.95,
        exploration_const=5.0,
        rollout_policy=BAMDPRolloutPolicy()
    )

    # Simulate a successful transition: (start, left) -> goal
    # Beta update rule: success -> (α+1, β)
    updated_beliefs_success = planner._update_beliefs(
        initial_beliefs, start, left, goal, success=True
    )
    alpha_success, beta_success = updated_beliefs_success[(start, left, goal)]
    assert alpha_success == 2.0, f"Expected alpha=2.0 after success, got {alpha_success}"
    assert beta_success == 1.0, f"Expected beta=1.0 after success, got {beta_success}"

    # Simulate a failed transition: (start, left) -> start (self-loop)
    # Beta update rule: failure -> (α, β+1)
    # The _update_beliefs method should be called with the TARGET state (goal)
    updated_beliefs_failure = planner._update_beliefs(
        initial_beliefs, start, left, goal, success=False
    )
    alpha_failure, beta_failure = updated_beliefs_failure[(start, left, goal)]
    assert alpha_failure == 1.0, f"Expected alpha=1.0 after failure, got {alpha_failure}"
    assert beta_failure == 2.0, f"Expected beta=2.0 after failure, got {beta_failure}"

    # Verify deterministic transitions are not updated
    # (start, right, goal) is not in beliefs, so update should be no-op
    right = SimpleAction("right")
    updated_beliefs_det = planner._update_beliefs(
        initial_beliefs, start, right, goal, success=True
    )
    assert (start, right, goal) not in updated_beliefs_det
    assert len(updated_beliefs_det) == 1

    print("✓ Belief update test passed")
    print(f"  Success: Beta(1,1) -> Beta({alpha_success},{beta_success})")
    print(f"  Failure: Beta(1,1) -> Beta({alpha_failure},{beta_failure})")


def test_ba_pouct_deterministic_action():
    """
    Test that the planner recognizes deterministic transitions.

    When (start, right, goal) is not in transition_beliefs,
    it should be treated as deterministic (100% success).
    """
    np.random.seed(42)
    agent = create_test_bamdp()

    # Verify right action is not in beliefs (deterministic)
    start = SimpleState("start")
    right = SimpleAction("right")
    goal = SimpleState("goal")

    assert (start, right, goal) not in agent.transition_beliefs()

    planner = POUCT(
        max_depth=10,
        num_sims=100,
        discount_factor=0.95,
        exploration_const=5.0,
        rollout_policy=BAMDPRolloutPolicy()
    )

    action = planner.plan(agent)

    # Both actions should be explored
    assert len(agent.tree.children) == 2

    print("✓ Deterministic action test passed")


def test_ba_pouct_rollout():
    """
    Test that rollouts work correctly during planning.

    This tests rollout behavior by examining the outcomes over many
    simulations rather than calling internal methods directly.
    """
    np.random.seed(42)
    agent = create_test_bamdp()

    # Run planning with many simulations to test rollouts
    planner = POUCT(
        max_depth=10,
        num_sims=200,
        discount_factor=0.95,
        exploration_const=5.0,
        rollout_policy=BAMDPRolloutPolicy()
    )

    action = planner.plan(agent)

    # Verify planning completed
    assert planner.last_num_sims == 200
    assert action is not None

    # Get statistics from tree
    left = SimpleAction("left")
    right = SimpleAction("right")

    qnode_left = agent.tree[left]
    qnode_right = agent.tree[right]

    print(f"  Left: visits={qnode_left.num_visits}, value={qnode_left.value:.3f}")
    print(f"  Right: visits={qnode_right.num_visits}, value={qnode_right.value:.3f}")

    # Verify values are reasonable
    # Left should have value close to -2.0 (direct cost to goal with some failures)
    # Right should have value close to -10.0 (guaranteed high cost)
    assert qnode_left.value > -5.0, "Left should not be too negative"
    assert qnode_right.value < -5.0, "Right should be more negative than left"

    # Verify rollout policy is being used (both actions explored)
    assert qnode_left.num_visits > 10, "Left should be explored significantly"
    assert qnode_right.num_visits >= 1, "Right should be explored at least once"

    # With these parameters, left dominates because it's clearly better
    # This is correct behavior - UCB is working as expected
    assert qnode_left.num_visits > qnode_right.num_visits, \
        "Left (cheaper) should be visited more than right (expensive)"

    print("✓ Rollout test passed")


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


# ============================================================================
# Main test runner
# ============================================================================

def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("Running BA-POUCT Test Suite")
    print("="*70 + "\n")

    tests = [
        ("Agent Creation", test_agent_creation),
        ("Transition Model", test_transition_model),
        ("Observation Model", test_observation_model),
        ("Reward Model", test_reward_model),
        ("Planning", test_ba_pouct_planning),
        ("Simulation", test_ba_pouct_simulation),
        ("Belief Updates", test_ba_pouct_belief_update),
        ("Deterministic Actions", test_ba_pouct_deterministic_action),
        ("Rollouts", test_ba_pouct_rollout),
        ("Full Episode", test_full_episode),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n[Test] {name}")
            print("-" * 70)
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
