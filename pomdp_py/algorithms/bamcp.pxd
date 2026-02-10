from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, PolicyModel, Action, State, Observation

# Import shared classes from ba_po_uct
from pomdp_py.algorithms.ba_po_uct cimport TreeNode, QNode, VNode, RootVNode, \
    ActionPrior, RolloutPolicy, RandomRollout, HeuristicFunction

cdef class BAMCP(Planner):
    cdef int _max_depth
    cdef float _planning_time
    cdef int _num_sims
    cdef int _num_visits_init
    cdef float _value_init
    cdef float _discount_factor
    cdef float _exploration_const
    cdef ActionPrior _action_prior
    cdef RolloutPolicy _rollout_policy
    cdef HeuristicFunction _heuristic_fn
    cdef Agent _agent
    cdef int _last_num_sims
    cdef float _last_planning_time
    cdef bint _show_progress
    cdef int _pbar_update_interval
    cdef bint _debug

    cpdef _search(self)
    cdef _initialize_progress_bar(self)
    cpdef _perform_simulation(self, state, sampled_p_success)
    cdef bint _should_stop(self, int sims_count, double start_time)
    cdef _update_progress(self, pbar, int sims_count, double start_time)
    cdef _finalize_progress_bar(self, pbar)

    cpdef _simulate(BAMCP self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth, dict sampled_p_success)

    cpdef _expand_vnode(self, VNode vnode, tuple history, State state=*)
    cpdef _rollout(self, State state, tuple history, VNode root, int depth, dict sampled_p_success)
    cpdef Action _ucb(self, VNode root)
    cpdef set_rollout_policy(self, RolloutPolicy rollout_policy)

    cpdef tuple _sample_ba_transition(self, State state, Action action, dict sampled_p_success)
