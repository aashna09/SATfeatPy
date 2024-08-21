# cython: language_level=3

import math
import random
cimport cython
import statistics
from feature_computation.enums cimport VarState, ClauseState
from feature_computation.stopwatch cimport Stopwatch
from sat_instance.sat_instance cimport SATInstance
from libc.time cimport clock, CLOCKS_PER_SEC
import numpy as np
cimport numpy as np


cdef class DPLLProbing:
    """
    So the values(states) of the variables are stored in varStates
    The clauses themselves are not changed (I assume for storage reasons)

    So assignments of the variables are stored in varstate, if they are true, false, etc,
    and then clauses are iterated through and checked if they are unassigned, so that is the next unit clause

    lengths of clauses kept track of in clause_lengths


    First check is for unit clauses -> if there are unit clauses
    Within these clauses, the unassigned variable is found
    This variable is set to the value of the literal (positive or negative)
    and the clauses that contain this variable are reduced

    reduction
    first remove all of the instances of the negative variable...
    so reduce the size of clauses with this variable,
    if the new size is 0, then we have a problem (inconsistent)

    Then satisfy the consistent clauses (clauses that contain that literal)

    Code adapted from SATzilla implementation

    """
    def __init__(self, sat_instance):
        self.sat_instance = sat_instance
        self.verbose = sat_instance.verbose
        self.num_vars_to_try = 10
        self.num_probes = 5

        # still not sure what lob is
        self.num_lob_probe = 30000
        # not sure what 2 is in this case(perhaps 2 seconds?)
        self.time_limit = 0.5

        # two stacks
        self.num_reduced_clauses = []
        self.num_reduced_vars = []

        # stack of variables that have been reduced
        self.reduced_vars = []
        self.reduced_clauses = []
        self.unit_props_log_nodes_dict = {}
        self.search_space_measures_dict = {}
        self.left_subtree_size = []
        cdef int i

        for i in range(self.sat_instance.v + 1):
            self.left_subtree_size.append(0)
        self.branch_lengths = []
        self.branch_probabilities = []
        self.depths_knuth = []

        self.probing_stopwatch = Stopwatch()

    # unit propagation


# num_bin_clauses_with_var, int array containing the number of binary clauses with a certain variable (index),

# this should change as the propagation happens

    def search_space_probe(self, bint halt_on_assignment=False):
        """
        Randomly choose a variable, propagate, and see how deep you get. This generates the mean depth to contradiction
        and an estimate on the number of nodes.
        Timeout of 2 seconds (can be changed, more runs means a more accurate result, as the algorithm is stochastic)
        :param halt_on_assignment:
        :return:
        """
        # lobjois probe in satzilla
        cdef int var, randnum, stepsleft, probe_num
        cdef bint val
        cdef list depths
        cdef double mean_depth_to_con_over_vars, res, lobjois
        cdef Stopwatch sw
        cdef int orig_num_active_vars, max_depth

        if self.verbose:
            print("search space estimate probe")

        sw = Stopwatch()
        sw.start()

        depths = []

        orig_num_active_vars = self.sat_instance.num_active_vars

        probe_num = 0

        # while probe_num < self.num_lob_probe and stopwatch.lap() < self.lobjois_tim_limit:
        while probe_num < self.num_lob_probe and sw.lap() < self.time_limit:

            # randomly choose an unassigned variable and a value propagate this, and continue until a contradiction is reached. This result is the depth to contradiction.
            # Do this while within the time limit, and within the number of probes

            var = 1
            val = False

            # another do while block
            while True:

                if self.sat_instance.num_active_vars == 0:
                    if self.sat_instance.num_active_clauses == 0 and halt_on_assignment:
                        print("finished")
                        return
                    else:
                        break

                # chooses a random unassigned variable, should be between 1 and v
                var = 0
                randnum = random.randint(1, self.sat_instance.num_active_vars)
                for stepsleft in range(randnum, 0, -1):
                    var += 1

                    while self.sat_instance.var_states[var] != VarState.UNASSIGNED:
                        var += 1
                        # This upper limit should be v, as the actual array of variables does not change, but the just
                        # states of the variables
                        if var == self.sat_instance.v:
                            var = 1

                # Choose a random value to propagate
                val = random.random() < 0.5

                if not self.set_var_and_prop(var, val):
                    break

            # print("reached bottom")
            depths.append(orig_num_active_vars - self.sat_instance.num_active_vars)

            # reset the problem
            while self.sat_instance.num_active_vars != orig_num_active_vars:
                self.backtrack()

            probe_num += 1

        mean_depth_to_con_over_vars = statistics.mean(depths)/self.sat_instance.v
        # print("mean depth over variables: ", mean_depth_to_contradiction_over_vars)

        self.unit_props_log_nodes_dict["mean_depth_to_contradiction_over_vars"] = mean_depth_to_con_over_vars

        max_depth = max(depths)

        res = 0.0
        for i in range(probe_num):
            res += math.pow(2, (depths[i] - max_depth))

        lobjois = max_depth + (math.log(res/probe_num) / math.log(2.0))
        if probe_num == 0:
            res = 0
        else:
            res = lobjois/self.sat_instance.v

        self.unit_props_log_nodes_dict["estimate_log_number_nodes_over_vars"] = res
        # print("lobjois log num nodes over vars", lobjois/self.sat_instance.v)

        # also should be timed
        if self.verbose:
            print("total time:", sw.lap())

    def unit_propagation_probe(self, bint haltOnAssignment=False):
        """
        Method to calculate the dpll probing features
        """
        cdef int current_depth, orig_num_active_vars, probe_num, next_probe_depth
        cdef bint reached_bottom
        cdef Stopwatch sw
        cdef int array_size, var, j, k, max_props_var, num_props, var_num
        cdef list vars_in_most_bin_clauses, num_bin
        cdef bint max_props_val, con, value, max_props
        cdef str unit_props_str
        cdef double unit_props_res

        if self.verbose:
            print("unit propagation probe")


        # the depths are manually set, multiples of 4 each time
        current_depth = 0
        orig_num_active_vars = self.sat_instance.num_active_vars
        reached_bottom = False

        sw = Stopwatch()
        sw.start()

        for probe_num in range(self.num_probes):
            next_probe_depth = 1
            for j in range(probe_num):
                next_probe_depth *= 4

            if self.verbose:
                print("searching depth: ", next_probe_depth)

            while current_depth < next_probe_depth and not reached_bottom and sw.lap() < self.time_limit:
                vars_in_most_bin_clauses = [0] * self.num_vars_to_try
                num_bin = [0] * self.num_vars_to_try

                array_size = 0
                for var in range(1, self.sat_instance.v + 1):
                    if self.sat_instance.var_states[var] != VarState.UNASSIGNED:
                        continue

                    if array_size < self.num_vars_to_try:
                        array_size += 1

                    j = 0
                    while j < array_size - 1 and self.sat_instance.num_bin_clauses_with_var[var] < num_bin[j]:
                        j += 1

                    for k in range(array_size - 1, j, -1):
                        vars_in_most_bin_clauses[k] = vars_in_most_bin_clauses[k - 1]
                        num_bin[k] = num_bin[k - 1]

                    vars_in_most_bin_clauses[j] = var
                    num_bin[j] = self.sat_instance.num_bin_clauses_with_var[var]

                max_props_var = 0
                max_props_val = False

                if array_size == 0:
                    max_props_var = 1
                    while self.sat_instance.var_states[max_props_var] != VarState.UNASSIGNED and max_props_var < self.sat_instance.v:
                        max_props_var += 1

                    max_props_val = True

                else:
                    max_props = -1

                    for var_num in range(array_size):
                        value = True
                        while True:
                            con = self.set_var_and_prop(vars_in_most_bin_clauses[var_num], value)
                            if con and self.sat_instance.num_active_vars <= 0:
                                if haltOnAssignment:
                                    print("solved")
                                    print(self.sat_instance.num_active_clauses)
                                    print(self.sat_instance.num_active_vars)
                                    return

                            num_props = orig_num_active_vars - self.sat_instance.num_active_vars - current_depth

                            if num_props > max_props:
                                max_props_var = vars_in_most_bin_clauses[var_num]
                                max_props_val = value

                            self.backtrack()

                            value = not value
                            if value:
                                break

                assert max_props_var != 0

                if not self.set_var_and_prop(max_props_var, max_props_val):
                    reached_bottom = True
                elif self.sat_instance.num_active_clauses == 0:
                    if haltOnAssignment:
                        print(self.sat_instance.num_active_clauses)
                        print(self.sat_instance.num_active_vars)
                        return
                    reached_bottom = True

                current_depth += 1

            unit_props_str = f"unit_props_at_depth_{next_probe_depth}"
            unit_props_res = (orig_num_active_vars - self.sat_instance.num_active_vars - current_depth) / self.sat_instance.v
            self.unit_props_log_nodes_dict[unit_props_str] = unit_props_res

        while self.sat_instance.num_active_vars != orig_num_active_vars:
            self.backtrack()

    def combined_probing(self):
        self.start_probing()
        self.reset_estimators_data()

        while self.probing_stopwatch.lap() < self.time_limit:
            if not self.make_decision_and_propagate():
                self.update_estimators_data()
                self.backtrack()

        weighted_backtrack_estimate = self.calculate_weighted_backtrack_estimate()
        recursive_estimate = self.estimate_tree_size(0)
        knuths_estimate = self.calculate_knuths_estimate()

        self.search_space_measures_dict["knuths_estimate"] = knuths_estimate
        self.search_space_measures_dict["weighted_backtrack_estimate"] = weighted_backtrack_estimate
        self.search_space_measures_dict["recursive_estimate"] = recursive_estimate

    def reset_estimators_data(self):
        self.left_subtree_size = []
        cdef int i
        for i in range(self.sat_instance.v + 1):
            self.left_subtree_size.append(0)
        self.branch_lengths = []
        self.branch_probabilities = []
        self.depths_knuth = []

    def update_estimators_data(self):
        cdef int current_depth = len(self.reduced_vars)
        self.left_subtree_size[current_depth - 1 if current_depth > 0 else 0] += 1
        self.branch_lengths.append(current_depth)
        self.branch_probabilities.append(2 ** -current_depth)
        self.depths_knuth.append(current_depth)

    def make_decision_and_propagate(self):
        cdef int var = self.select_unassigned_variable()
        cdef bint success
        if var is None:
            return True  # No unassigned variables left

        cdef bint value = random.choice([True, False])
        success = self.set_var_and_prop(var, value)
        if success:
            self.update_estimators_data()
        return success

    def select_unassigned_variable(self):
        cdef int var
        for var in range(1, self.sat_instance.v + 1):
            if self.sat_instance.var_states[var] == VarState.UNASSIGNED:
                return var
        return None

    def calculate_weighted_backtrack_estimate(self):
        if not self.branch_lengths:
            return 0.0

        cdef double epsilon = 1e-10
        cdef int n = len(self.branch_probabilities)
        cdef double log_weighted_sum = 0.0
        cdef double total_log_prob = 0.0

        cdef double prob
        cdef double adjusted_prob

        for i in range(n):
            prob = self.branch_probabilities[i]
            adjusted_prob = prob if prob > 0 else epsilon
            log_weighted_sum += math.log(adjusted_prob) + (self.branch_lengths[i] + 1) * math.log(2)
            total_log_prob += math.log(adjusted_prob)

        cdef double weighted_backtrack_estimate = log_weighted_sum - total_log_prob
        cdef double log_weighted_backtrack_estimate = weighted_backtrack_estimate / self.sat_instance.v

        return log_weighted_backtrack_estimate

    def estimate_branching_factor_reduction(self, int depth):
        cdef double base_activity = 1.0
        cdef double decay_factor = 0.95
        cdef double adjusted_activity = base_activity * (decay_factor ** depth)
        cdef double reduction = min(1.0, adjusted_activity / (1 + depth * 0.1))

        return reduction

    def estimate_tree_size(self, int depth, double branching_factor=2):
        if depth >= self.sat_instance.v:
            return 0.0

        cdef double reduction = self.estimate_branching_factor_reduction(depth)
        cdef double adjusted_branching_factor = max(1, branching_factor - reduction)
        cdef double left_size = self.left_subtree_size[depth] * adjusted_branching_factor
        cdef double right_size = left_size
        cdef double tree_size = 1 + left_size + right_size
        cdef double log_tree_size = math.log(tree_size + 1)
        
        return log_tree_size / self.sat_instance.v

    def calculate_knuths_estimate(self):
        if not self.depths_knuth:
            return 0.0

        cdef double average_depth = sum(self.depths_knuth) / len(self.depths_knuth)
        cdef double log_knuths_estimate = (average_depth + 1) * math.log(2)
        cdef double normalized_log_knuths_estimate = log_knuths_estimate / self.sat_instance.v

        return normalized_log_knuths_estimate

    def start_probing(self):
        self.probing_stopwatch.start()

    def set_var_and_prop(self, int var, bint value):
        cdef int num_clauses_reduced = 0
        cdef int num_vars_reduced = 1
        cdef int literal
        cdef bint consistent

        assert self.sat_instance.var_states[var] == VarState.UNASSIGNED

        if value:
            self.sat_instance.var_states[var] = VarState.TRUE_VAL
            literal = var
        else:
            self.sat_instance.var_states[var] = VarState.FALSE_VAL
            literal = -var

        self.reduced_vars.append(var)
        self.sat_instance.num_active_vars -= 1

        consistent, num_clauses_reduced, num_vars_reduced = self.reduce_clauses(literal, num_clauses_reduced, num_vars_reduced)

        if consistent:
            consistent, num_clauses_reduced, num_vars_reduced = self.unit_prop(num_clauses_reduced, num_vars_reduced)

        self.num_reduced_clauses.append(num_clauses_reduced)
        self.num_reduced_vars.append(num_vars_reduced)

        return consistent

    def reduce_clauses(self, int orig_literal, int num_clauses_reduced, int num_vars_reduced):
        cdef int clause_num, literal, curr_var, i, j

        for clause_num in self.sat_instance.clauses_with_literal(-orig_literal):
            # iterate through all of the clauses that contain this literal
            # if it is active
            if self.sat_instance.clause_states[clause_num] == ClauseState.ACTIVE:
                self.reduced_clauses.append(clause_num)
                num_clauses_reduced += 1

                # decrease the size (this length actually represents the number of yet to be assigned variables within that clause)
                self.sat_instance.clause_lengths[clause_num] -= 1

                if self.sat_instance.clause_lengths[clause_num] == 2:
                    for literal in self.sat_instance.clauses[clause_num]:
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] += 1

                elif self.sat_instance.clause_lengths[clause_num] == 1:
                    for literal in self.sat_instance.clauses[clause_num]:
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] -= 1
                    self.sat_instance.unit_clauses.append(clause_num)

                elif self.sat_instance.clause_lengths[clause_num] == 0:
                    return False, num_clauses_reduced, num_vars_reduced

        for i in range(len(self.sat_instance.clauses_with_literal(orig_literal))):
            clause_num = self.sat_instance.clauses_with_literal(orig_literal)[i]
            if self.sat_instance.clause_states[clause_num] == ClauseState.ACTIVE:
                self.sat_instance.clause_states[clause_num] = ClauseState.PASSIVE
                self.reduced_clauses.append(clause_num)
                self.sat_instance.num_active_clauses -= 1

                for j in range(len(self.sat_instance.clauses[clause_num])):
                    curr_var = abs(self.sat_instance.clauses[clause_num][j])
                    self.sat_instance.num_active_clauses_with_var[curr_var] -= 1

                    if self.sat_instance.clause_lengths[clause_num] == 2:
                        self.sat_instance.num_bin_clauses_with_var[curr_var] -= 1

                    if self.sat_instance.num_active_clauses_with_var[curr_var] == 0 and self.sat_instance.var_states[curr_var] == VarState.UNASSIGNED:
                        self.sat_instance.var_states[curr_var] = VarState.IRRELEVANT
                        self.reduced_vars.append(curr_var)
                        self.sat_instance.num_active_vars -= 1
                        num_vars_reduced += 1

                num_clauses_reduced += 1

        return True, num_clauses_reduced, num_vars_reduced

    def unit_prop(self, int num_clauses_reduced, int num_vars_reduced):
        cdef bint consistent = True
        cdef int clause_number, lit_num, literal
        cdef bint r_consistent

        while (len(self.sat_instance.unit_clauses) > 0) and consistent:
            clause_number = self.sat_instance.unit_clauses.pop()

            if self.sat_instance.clause_states[clause_number] != ClauseState.ACTIVE:
                continue

            lit_num = 0
            while self.sat_instance.var_states[abs(self.sat_instance.clauses[clause_number][lit_num])] != VarState.UNASSIGNED:
                lit_num += 1

            assert self.sat_instance.clause_lengths[clause_number] == 1
            literal = self.sat_instance.clauses[clause_number][lit_num]

            if literal > 0:
                self.sat_instance.var_states[abs(literal)] = VarState.TRUE_VAL
            else:
                self.sat_instance.var_states[abs(literal)] = VarState.FALSE_VAL

            self.reduced_vars.append(abs(literal))
            self.sat_instance.num_active_vars -= 1
            num_vars_reduced += 1

            r_consistent, num_clauses_reduced, num_vars_reduced = self.reduce_clauses(literal, num_clauses_reduced, num_vars_reduced)
            consistent = consistent and r_consistent

        return consistent, num_clauses_reduced, num_vars_reduced

    def backtrack(self):
        """
        Should undo one call of setVar or unitprop
        :return:
        """
        cdef int num_vars_reduced, num_clauses_reduced, var, clause_num, j, literal, i

        num_vars_reduced = self.num_reduced_vars.pop()

        for i in range(num_vars_reduced):
            var = self.reduced_vars.pop()
            self.sat_instance.var_states[var] = VarState.UNASSIGNED
            self.sat_instance.num_active_vars += 1

        num_clauses_reduced = self.num_reduced_clauses.pop()
        for i in range(num_clauses_reduced):
            clause_num = self.reduced_clauses.pop()

            if self.sat_instance.clause_states[clause_num] != ClauseState.ACTIVE:
                self.sat_instance.num_active_clauses += 1
                self.sat_instance.clause_states[clause_num] = ClauseState.ACTIVE

                if self.sat_instance.clause_lengths[clause_num] == 2:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_active_clauses_with_var[abs(literal)] += 1
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] += 1
                else:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_active_clauses_with_var[abs(literal)] += 1
            else:
                self.sat_instance.clause_lengths[clause_num] += 1

                if self.sat_instance.clause_lengths[clause_num] == 2:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] += 1

                elif self.sat_instance.clause_lengths[clause_num] == 3:
                    for j in range(len(self.sat_instance.clauses[clause_num])):
                        literal = self.sat_instance.clauses[clause_num][j]
                        self.sat_instance.num_bin_clauses_with_var[abs(literal)] -= 1

        self.sat_instance.unit_clauses = []
