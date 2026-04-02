cdef class FastVI:
    # Structure arrays (built once from graph)
    cdef public int num_states
    cdef public int num_sa_pairs
    cdef public int num_uncertain
    cdef int[:] sa_state           # sa_pair → state index
    cdef int[:] sa_target          # sa_pair → target state index
    cdef double[:] r_success       # sa_pair → reward if transition succeeds
    cdef double[:] r_failure       # sa_pair → reward if transition fails (self-loop)
    cdef int[:] state_sa_start     # state → first sa_pair index
    cdef int[:] state_sa_count     # state → number of sa_pairs for this state
    cdef int[:] sa_belief_idx      # sa_pair → index into p_success array (-1 if deterministic)
    cdef bint[:] state_is_terminal # state → is terminal

    # Solve parameters
    cdef public double discount_factor
    cdef double epsilon
    cdef int max_iterations

    # Solve output (reused across calls)
    cdef double[:] V
    cdef int[:] policy             # policy[state] → sa_pair index of best action

    # Mapping back to objects
    cdef public list idx_to_node_id       # state index → node_id
    cdef public dict node_id_to_idx       # node_id → state index
    cdef public list sa_to_action_node_id # sa_pair → target node_id (the action)
    cdef public dict belief_key_to_idx    # (MDPState, MDPAction, MDPState) → uncertain idx

    cpdef solve(self, double[:] p_success)
    cpdef int get_action_node_id(self, object state_node_id)
