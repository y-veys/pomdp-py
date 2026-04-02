from pomdp_py.framework.basics cimport State, Action, Observation, \
    TransitionModel, RewardModel, ObservationModel
from pomdp_py.algorithms.ba_po_uct cimport RolloutPolicy, HeuristicFunction, ActionPrior


cdef class MDPState(State):
    cdef public object node_id
    cdef public object position
    cdef public bint is_frontier
    cdef Py_hash_t _hash


cdef class MDPAction(Action):
    cdef public object target_node_id
    cdef Py_hash_t _hash


cdef class MDPObservation(Observation):
    cdef public object node_id
    cdef Py_hash_t _hash


cdef class MDPTransitionModel(TransitionModel):
    cdef public object mdp
    cdef public object goal_node_id
    cdef public dict _adjacency
    cdef public list _all_states
    cdef public dict _state_cache


cdef class MDPRewardModel(RewardModel):
    cdef public object mdp
    cdef public object goal_node_id
    cdef public double goal_reward
    cdef public object knowledge
    cdef public double navigation_failure_penalty
    cdef public double door_failure_penalty
    cdef public dict _reward_cache


cdef class MDPObservationModel(ObservationModel):
    pass


cdef class MDPHeuristic(HeuristicFunction):
    cdef public object mdp
    cdef public object goal_node_id
    cdef public object goal_position
    cdef dict _distance_cache


cdef class MDPFallbackHeuristic(HeuristicFunction):
    cdef public object goal_node_id
    cdef public float _goal_reward
    cdef public float _fallback_cost
    cdef dict _cost_to_start


cdef class MDPActionPrior(ActionPrior):
    cdef public dict vi_values
    cdef public MDPTransitionModel transition_model
    cdef public MDPRewardModel reward_model
    cdef public double discount_factor


cdef class MDPPolicyModel(RolloutPolicy):
    cdef public MDPTransitionModel transition_model
    cdef public dict valid_actions_map
    cdef public list _all_actions


cdef class MDPShortestPathPolicyModel(RolloutPolicy):
    cdef public MDPTransitionModel transition_model
    cdef public object goal_node_id
    cdef public double epsilon
    cdef public dict valid_actions_map
    cdef public list _all_actions
    cdef public dict next_actions
