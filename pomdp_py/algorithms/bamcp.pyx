"""BAMCP: Bayes-Adaptive Monte Carlo Planning with Root Sampling.

This algorithm implements MCTS for Bayes-Adaptive MDPs (BAMDPs) using root sampling.
At the start of each simulation, all transition probabilities are sampled from their
Beta prior distributions. These sampled probabilities remain fixed throughout the
entire simulation (both tree traversal and rollout phases).

This is equivalent to sampling an MDP from the belief distribution, then running
standard MCTS on that fixed MDP. The Q-values converge to the expected value over
MDPs sampled from the prior.

Key difference from BA-POUCT:
- BA-POUCT: Updates transition beliefs during simulation (path-dependent returns)
- BAMCP: Samples all p_success once at root, no belief updates during simulation

This approach reduces variance in Q-value estimates by ensuring that each simulation
explores a consistent MDP rather than having beliefs change mid-simulation.
"""
# cython: profile=True 

from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel, TransitionBelief
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles

# Import shared tree node classes and rollout policy from ba_po_uct
from pomdp_py.algorithms.ba_po_uct cimport TreeNode, QNode, VNode, RootVNode, \
    ActionPrior, RolloutPolicy, RandomRollout, HeuristicFunction
from pomdp_py.algorithms.ba_po_uct import TreeNode, QNode, VNode, RootVNode, \
    ActionPrior, RolloutPolicy, RandomRollout, HeuristicFunction

import time
import random
import math
from tqdm import tqdm

cdef class BAMCP(Planner):

    """ BAMCP: Bayes-Adaptive Monte Carlo Planning with Root Sampling.

    BAMCP implements MCTS for Bayes-Adaptive MDPs using root sampling:
    - At each simulation, sample all transition probabilities from Beta priors
    - Use these fixed probabilities throughout tree traversal and rollout
    - No belief updates during simulation (unlike BA-POUCT)

    BAMCP only works for problems with action space that can be enumerated.

    __init__(self,
             max_depth=5, planning_time=1., num_sims=-1,
             discount_factor=0.9, exploration_const=math.sqrt(2),
             num_visits_init=1, value_init=0,
             rollout_policy=RandomRollout(),
             action_prior=None, heuristic_fn=None, show_progress=False, pbar_update_interval=5)

    Args:
        max_depth (int): Depth of the MCTS tree. Default: 5.
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
        action_prior (ActionPrior): a prior over preferred actions given state and history.
        heuristic_fn (HeuristicFunction): optional heuristic function for estimating cost-to-go
            when max_depth is reached during rollouts. Default: None.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to run after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.
    """

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=None,
                 action_prior=None, heuristic_fn=None, show_progress=False, pbar_update_interval=5, debug=False):
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._num_visits_init = num_visits_init
        self._value_init = value_init
        self._rollout_policy = rollout_policy
        self._discount_factor = discount_factor
        self._exploration_const = exploration_const
        self._action_prior = action_prior
        self._heuristic_fn = heuristic_fn

        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

        self._debug = debug

    @property
    def updates_agent_belief(self):
        return False

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def num_visits_init(self):
        return self._num_visits_init

    @property
    def discount_factor(self):
        return self._discount_factor

    @property
    def value_init(self):
        return self._value_init

    @property
    def action_prior(self):
        return self._action_prior

    @property
    def rollout_policy(self):
        return self._rollout_policy

    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        if self._rollout_policy is None:
            raise ValueError("rollout_policy unset. Please call set_rollout_policy, "
                             "or pass in a rollout_policy upon initialization")

        # Validate that agent has the required interface for BA-BAMCP
        if not hasattr(agent, 'transition_beliefs'):
            raise TypeError("BA-BAMCP requires an agent with transition_beliefs property. "
                          "The agent must be a Bayes-Adaptive agent that maintains beliefs "
                          "over transition probabilities.")

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action

    cpdef public update(self, Agent agent, Action real_action, Observation real_observation):
        """
        update(self, BayesAdaptiveAgent agent, Action real_action, Observation real_observation)
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        """
        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if real_action not in agent.tree\
           or real_observation not in agent.tree[real_action]:
            agent.tree = None  # replan, if real action or observation differs from all branches
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNode.from_vnode(
                agent.tree[real_action][real_observation],
                agent.history)
        else:
            raise ValueError("Unexpected state; child should not be None")

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    cpdef set_rollout_policy(self, RolloutPolicy rollout_policy):
        """
        set_rollout_policy(self, RolloutPolicy rollout_policy)
        Updates the rollout policy to the given one
        """
        self._rollout_policy = rollout_policy

    cpdef _expand_vnode(self, VNode vnode, tuple history, State state=None):
        cdef Action action
        cdef tuple preference
        cdef int num_visits_init
        cdef float value_init

        for action in self._agent.valid_actions(state=state, history=history):
            if vnode[action] is None:
                history_action_node = QNode(self._num_visits_init,
                                            self._value_init)
                vnode[action] = history_action_node

        if self._action_prior is not None:
            # Using action prior; special values are set;
            for preference in \
                self._action_prior.get_preferred_actions(state, history):
                action, num_visits_init, value_init = preference
                history_action_node = QNode(num_visits_init,
                                            value_init)
                vnode[action] = history_action_node

    cpdef _search(self):
        cdef int sims_count = 0
        cdef double start_time, time_taken
        pbar = self._initialize_progress_bar()
        start_time = time.time()

        # For our Bayes-Adaptive MDP, we assume that the robot's state is known.
        # So, even though we are sampling from a distribution, the distribution
        # should be degenerate. We can sample once outside of the loop.
        state = self._agent.sample_belief()

        if self._debug: 
            print(f"\n{'='*70}")
            print(f"Starting MCTS search from state: {state}")
            print(f"{'='*70}")

        while not self._should_stop(sims_count, start_time):
            if self._debug:
                print(f"\n--- Simulation {sims_count + 1} ---")
            # Shallow copy of dict keys, copy the values (alpha, beta tuples)
            # This avoids issues with unpicklable keys like NodeSymbol
            transition_beliefs = {key: value for key, value in self._agent.transition_beliefs.items()}
            if self._debug:
                print(f"Initial beliefs: {transition_beliefs}")

            # Lazy root sampling: initialize empty dict, sample on-demand in _sample_ba_transition
            # This avoids sampling p_success for transitions we never visit
            sampled_p_success = {}

            self._perform_simulation(state, sampled_p_success)
            sims_count += 1
            self._update_progress(pbar, sims_count, start_time)

        self._finalize_progress_bar(pbar)
        best_action = self._agent.tree.argmax()
        time_taken = time.time() - start_time

        if self._debug:
            print(f"\n{'='*70}")
            print(f"Search complete: {sims_count} simulations")
            print(f"Selected action: {best_action}")
            print(f"{'='*70}\n")

        return best_action, time_taken, sims_count

    cdef _initialize_progress_bar(self):
        if self._show_progress:
            total = self._num_sims if self._num_sims > 0 else self._planning_time
            return tqdm(total=total)

    cpdef _perform_simulation(self, state, sampled_p_success):
        self._simulate(state=state, history=self._agent.history, root=self._agent.tree, parent=None, observation=None, depth=0, sampled_p_success=sampled_p_success)

    cdef bint _should_stop(self, int sims_count, double start_time):
        cdef float time_taken = time.time() - start_time
        if self._num_sims > 0:
            return sims_count >= self._num_sims
        else:
            return time_taken > self._planning_time

    cdef _update_progress(self, pbar, int sims_count, double start_time):
        if self._show_progress:
            pbar.n = sims_count if self._num_sims > 0 else round(time.time() - start_time, 2)
            pbar.refresh()

    cdef _finalize_progress_bar(self, pbar):
        if self._show_progress:
            pbar.close()

    cpdef _simulate(BAMCP self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth, dict sampled_p_success):
        if depth > self._max_depth:
            return 0
        if root is None:
            if self._debug: 
                print(f"  [Depth {depth}] New node at state={state}, expanding...")

            if self._agent.tree is None:
                root = self._VNode(root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()
            if parent is not None:
                parent[observation] = root
            self._expand_vnode(root, history, state=state)

            if self._debug:
                print(f"  [Depth {depth}] Starting rollout from {state}")

            rollout_reward = self._rollout(state, history, root, depth, sampled_p_success)

            if self._debug:
                print(f"  [Depth {depth}] Rollout returned reward: {rollout_reward:.2f}")
            return rollout_reward
        cdef int nsteps
        action = self._ucb(root)
        if self._debug:
            print(f"  [Depth {depth}] At {state}, UCB selected action={action}")
            print(f"  [Depth {depth}] Current Q-values: ", end="")
            for a in root.children:
                print(f"{a}={root[a].value:.2f}(n={root[a].num_visits}) ", end="")
            print()

        next_state, observation, success, reward, nsteps = self._sample_ba_transition(state, action, sampled_p_success)
        if self._debug:
            print(f"  [Depth {depth}] Transition: {state} --{action}--> {next_state} ({'success' if success else 'failure'}, r={reward:.1f})")

        target_state = self._agent.transition_model.sample(state=state, action=action)
                
        if nsteps == 0:
            # This indicates the provided action didn't lead to transition
            # Perhaps the action is not allowed to be performed for the given state
            # (for example, the state is not in the initiation set of the option,
            # or the state is a terminal state)
            return reward

        total_reward = reward + (self._discount_factor**nsteps)*self._simulate(next_state,
                                                                               history + ((action, observation),),
                                                                               root[action][observation],
                                                                               root[action],
                                                                               observation,
                                                                               depth+nsteps,
                                                                               sampled_p_success)
        root.num_visits += 1
        root[action].num_visits += 1
        old_value = root[action].value
        root[action].value = root[action].value + (total_reward - root[action].value) / (root[action].num_visits)
        if self._debug:
            print(f"  [Depth {depth}] Backprop: Q({action}) = {old_value:.2f} -> {root[action].value:.2f} (n={root[action].num_visits}, R={total_reward:.2f})")
        return total_reward

    cpdef _rollout(self, State state, tuple history, VNode root, int depth, dict sampled_p_success):
        cdef Action action
        cdef float discount = 1.0
        cdef float total_discounted_reward = 0
        cdef State next_state
        cdef Observation observation
        cdef float reward
        cdef int nsteps
        cdef float heuristic_value

        cdef int rollout_step = 0

        if self._debug:
            print(f"    [Rollout] Starting from {state}, depth={depth}")

        while depth < self._max_depth:
            # Check if current state is terminal before sampling action
            if hasattr(self._agent.transition_model, 'is_terminal') and \
               self._agent.transition_model.is_terminal(state):
                if self._debug:
                    print(f"    [Rollout step {rollout_step}] Terminal state {state}, stopping")
                break

            action = self._rollout_policy.rollout(state, history)
            next_state, observation, success, reward, nsteps = self._sample_ba_transition(state, action, sampled_p_success)

            target_state = self._agent.transition_model.sample(state=state, action=action)

            # Early termination if terminal state reached (nsteps == 0)
            if nsteps == 0:
                total_discounted_reward += reward * discount
                if self._debug:
                    print(f"    [Rollout step {rollout_step}] nsteps=0, stopping. cumulative={total_discounted_reward:.2f}")
                break

            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
            rollout_step += 1

        # Add heuristic estimate if max depth reached and heuristic function provided
        if depth >= self._max_depth and self._heuristic_fn is not None:
            heuristic_value = self._heuristic_fn.value(state)
            total_discounted_reward += discount * heuristic_value
            if self._debug:
                print(f"    [Rollout] Max depth reached at {state}. Heuristic={heuristic_value:.2f}, "
                      f"total={total_discounted_reward:.2f}")

        if self._debug:
            print(f"    [Rollout] Done. Total discounted reward={total_discounted_reward:.2f}")

        return total_discounted_reward

    cpdef Action _ucb(self, VNode root):
        """UCB1"""
        cdef Action best_action
        cdef float best_value
        best_action, best_value = None, float('-inf')
        for action in root.children:
            if root[action].num_visits == 0:
                val = float('inf')
            else:
                val = root[action].value + \
                    self._exploration_const * math.sqrt(math.log(root.num_visits + 1) / root[action].num_visits)
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    def _VNode(self, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            return RootVNode(self._num_visits_init, self._agent.history)

        else:
            return VNode(self._num_visits_init)

    cpdef tuple _sample_ba_transition(self, State state, Action action, dict sampled_p_success):
        """
        Sample a transition from the Bayes-Adaptive model.

        Args:
            state: Current state
            action: Action to take
            sampled_p_success (dict, optional): Pre-sampled p_success values from root sampling.
                If provided, uses these instead of sampling from beliefs.

        Returns:
            tuple: (next_state, success, reward)
                - next_state: The state we transitioned to
                - success (bool): Whether we successfully reached target (not self-loop)
                - reward (float): Immediate reward for this transition
        """
        cdef State target, next_state
        cdef float alpha, beta, p_success, reward
        cdef bint success

        # The transition model for a BAMDP agent defines the target state
        # given a (state, action) pair IF the agent were to transition successfully.
        target = self._agent.transition_model.sample(state=state, action=action)

        # Check if this transition has uncertain probability (in transition_beliefs)
        if (state, action, target) in self._agent.transition_beliefs:
            # Lazy root sampling: sample if not already sampled this simulation
            if (state, action, target) not in sampled_p_success:
                belief = self._agent.transition_beliefs[(state, action, target)]
                sampled_p_success[(state, action, target)] = random.betavariate(belief.alpha, belief.beta)

            p_success = sampled_p_success[(state, action, target)]
            success = random.random() < p_success

            if self._debug:
                print(f"      [BA-Sample] ({state}, {action}, {target}): "
                      f"p_success={p_success:.4f} -> {'SUCCESS' if success else 'FAILURE'}")
        else:
            # Deterministic transition (not in transition_beliefs)
            success = True

        if success:
            next_state = target
        else:
            next_state = state

        observation = self._agent.observation_model.sample(next_state, action)
        reward = self._agent.reward_model.sample(state, action, next_state)

        return next_state, observation, success, reward, 1
