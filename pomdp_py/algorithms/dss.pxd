from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Action, State, Observation, \
    TransitionModel, TransitionBelief
from pomdp_py.algorithms.fast_vi cimport FastVI

cdef class DSS(Planner):
    cdef int _max_depth
    cdef int _num_policies
    cdef int _num_samples
    cdef int _steps_per_policy
    cdef float _discount_factor
    cdef Agent _agent
    cdef int _last_num_sims
    cdef float _last_planning_time
    cdef bint _show_progress
    cdef bint _debug
    cdef public dict _last_action_values
    cdef FastVI _fast_vi
    cdef double[:] _p_success_buf
    cdef list _belief_keys

    # Belief arrays (sized num_uncertain, reused across calls)
    cdef double[:] _alpha_buf
    cdef double[:] _beta_buf
    cdef double[:] _strength_buf

    # Scratch copies for per-sample simulation
    cdef double[:] _alpha_scratch
    cdef double[:] _beta_scratch

    cpdef public plan(self, Agent agent)
    cdef tuple _dss(self, int state_idx, double[:] alpha, double[:] beta, int depth)
    cdef tuple _run_policy_k_steps(self, int[:] policy, int s_idx,
                                   double[:] alpha, double[:] beta, int depth)
    cdef void _sample_and_solve_mdp(self, double[:] alpha, double[:] beta)
