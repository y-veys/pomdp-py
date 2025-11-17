from pomdp_py.framework.planner cimport Planner
from pomdp_py.framework.basics cimport Agent, Action

cdef class MDPValueIteration(Planner):
    cdef float _discount_factor
    cdef float _epsilon
    cdef int _max_iterations
    cdef public dict _V
    cdef int _last_num_iterations
    cdef float _last_delta
    cdef dict _transition_cache

    cpdef _build_transition_cache(self, Agent agent)
    cpdef _compute_value_function(self, Agent agent)
