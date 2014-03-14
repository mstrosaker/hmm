#!/usr/bin/env python
#
# Copyright (c) 2014 Michael Strosaker

import itertools, math

class state:
    def __init__(self, name, p_initial, p_emission, p_transition,
                 p_termination=0.0):
        self.name = name			# string
        self.p_initial = p_initial		# float (between 0.0 and 1.0)
        self.p_emission = p_emission		# dictionary (string: float)
        self.p_transition = p_transition	# dictionary (string: float)
        self.p_termination = p_termination	# float (between 0.0 and 1.0)

class hmm:
    def __init__(self, alphabet, states):
        """
        Creates a new hidden Markov model object.

        Parameters:
          - alphabet: a list of strings, where each string represents a
            symbol that can possibly be emitted by any of the states in
            the model (for example, ['A', 'C', 'G', 'T'] for genomic data)
          - states: a list of state objects

        The initial distribution vector (usually referred to as pi) is
        described by the states with non-zero values for p_initial.

        If at least one of the states in the states parameter includes a
        non-zero p_termination, then there is an implied "end" state.
        In that case, only those states with a non-zero p_termination
        can be the last state in a valid sequence of states.

        TODO: Data validation:
          - ensure that all of the p_initial probabilities add up to
            (about) 1,0
          - ensure that the probabilities in the p_emission dictionaries
            add up to (about) 1.0 for each state
          - ensure that the probabilities in the p_transition dictionaries
            (plus p_termination, if non-zero) add up to (about) 1.0 for
            each state
          - ensure that there are no duplicate state names
          - ensure that no states will attempt to transition to non-
            existent states
          - ensure that no states will emit values that are not in the
            specified alphabet
          - ensure that all states are reachable
        """
        self.alphabet = alphabet
        self.terminal_state = False
        self.initial_states = []
        self.terminating_states = []
        self.states = {}
        for state in states:
            self.states[state.name] = state
            if state.p_initial > 0.0:
                self.initial_states.append(state.name)
            if state.p_termination > 0.0:
                self.terminal_state = True
                self.terminating_states.append(state.name)

    def score(self, seq_state, seq_observed):
        """
        Calculates the log (base 10) of the probability of observing this
        sequence of states and observations.

        Parameters:
          - seq_state: a list of strings, representing an ordered sequence
            of states in the HMM
          - seq_observed: a list of strings, representing an ordered sequence
            of symbols that were observed
        Returns:
          - a float, representing the log (base 10) of the probability
            of the state sequence being observed
          - None if the specified state sequence is invalid because:
           - the first state in the sequence cannot be an initial state (has
             a p_initial of 0)
           - there is no edge between two consecutive states in the state
             sequence (i.e., the probability of transitioning from state[i] to
             state[i+1] is 0 for some i)
           - a state cannot emit the corresponding symbol in the sequence of
             observations (i.e., seq_state[i] cannot emit seq_observed[i] for
             some i)
           - the model has a terminating state, and the last state in the
             sequence does not have an edge to the terminating state

        TODO: Data validation:
          - ensure that the specified sequence of observations only
            includes symbols present in the alphabet
        """

        # if there is an implied terminal state, make sure the last state
        # in the specified sequence has an edge to it
        if self.terminal_state:
            if seq_state[-1] not in self.terminating_states:
                return None

        p = 0.0
        state_prev = None
        for i in range(len(seq_state)):
            state_cur = self.states[seq_state[i]]
            if i == 0:
                # if the initial probability for this state is 0, then
                # this is not a valid sequence of states
                if state_cur.p_initial == 0.0:
                    return None

                p += math.log10(state_cur.p_initial)

            else:
                # check that the prior state can transition to this one
                if seq_state[i] not in state_prev.p_transition.keys():
                    return None

                p += math.log10(state_prev.p_transition[seq_state[i]])

            # check that this state can emit the observed alphabet symbol
            if seq_observed[i] not in state_cur.p_emission.keys():
                return None

            if state_cur.p_emission[seq_observed[i]] == 0.0:
                return None

            p += math.log10(state_cur.p_emission[seq_observed[i]])

            state_prev = state_cur

        return p

    def enumerate(self, observed):
        """
        Enumerates and prints every possible path of states that can
        explain an observed sequence of symbols, along with the probability
        associated with each of the state sequences.

        *** IMPORTANT NOTE: ***
        Enumerating all possible state sequences is an expensive operation
        for models with many states and for long observations.  The number
        of state sequences in the overall enumeration is:
            (#states)^(len(observation))

        Parameters:
          - seq_observed: a list of strings, representing an ordered sequence
            of symbols that were observed
        Returns: (nothing)

        TODO: Data validation:
          - ensure that the specified sequence of observations only
            includes symbols present in the alphabet
        """
        best_seq = None
        best_score = None
        for seq in itertools.product(self.states.keys(), repeat=len(observed)):
            s = self.score(seq, observed)
            if s is not None:
                print '%s: %f' % (seq, s)
                if best_score is None:
                    best_seq = seq
                    best_score = s
                elif s > best_score:
                    best_seq = seq
                    best_score = s

        print 'BEST: %s: %f' % (best_seq, best_score)

    def _p_emit(self, state, observation):
        """
        Retrieves the probability of a state emitting a given symbol.
        """
        if state not in self.states.keys():
            return None
        if observation not in self.states[state].p_emission.keys():
            return 0.0
        return self.states[state].p_emission[observation]

    def _p_transition(self, from_state, to_state):
        """
        Retrieves the probability of a state transitioning to a given state.
        """
        if from_state not in self.states.keys():
            return None
        if to_state not in self.states[from_state].p_transition.keys():
            return 0.0
        return self.states[from_state].p_transition[to_state]

    def _connected(self, from_state, to_state):
        """
        Establishes whether there is an edge between two given states.

        Parameters:
          - from_state: a state in the states member of this object
          - to_state: a state in the states member of this object
        Returns:
          - True if there is an edge from from_state to to_state
          - False if there is no such edge, or if the from_state does not
            exist
        """
        if from_state not in self.states.keys():
            return False
        if to_state not in self.states[from_state].p_transition.keys():
            return False
        if self.states[from_state].p_transition[to_state] > 0.0:
            return True
        return False

    def trellis(self, observed):
        """
        Builds a trellis of the probabilities of the possible paths,
        given a sequence of observed symbols.

        Parameters:
          - observed: a sequence of observed symbols
        Returns:
          - a list of dictionaries, one dictionary per symbol in the
            observations; each dictionary represents a column of the trellis

        TODO: Data validation:
          - ensure that the specified sequence of observations only
            includes symbols present in the alphabet
        """
        state_names = self.states.keys()
        trellis = []
        prior_probs = None

        for i in range(len(observed)):
            probs = {}

            if i == 0:
                # first state; only those states with initial probabilities
                # can have non-None values in this column
                for state in state_names:
                    p_init = self.states[state].p_initial
                    p_emit = self._p_emit(state, observed[i])
                    if p_init == 0.0 or p_emit == 0.0:
                        probs[state] = None
                    else:
                        probs[state] = math.log10(p_init) + \
                                       math.log10(p_emit)

            else:
                for state in state_names:
                    p_emit = self._p_emit(state, observed[i])
                    if p_emit == 0.0:
                        probs[state] = None
                    else:
                        best = None
                        for prev_state, prev_prob in prior_probs.iteritems():
                            if prev_prob is None:
                                continue
                            if self._connected(prev_state, state):
                                p_tran = self._p_transition(prev_state, state)
                                s = prev_prob + math.log10(p_emit) + \
                                      math.log10(p_tran)
                                if best == None:
                                    best = s
                                elif s > best:
                                    best = s
                        probs[state] = best

            # the last column of the trellis can only include those states
            # that can transition to the implied terminal state, if one exists
            if i == (len(observed)-1):
                if self.terminal_state:
                    for s in state_names:
                        if s not in self.terminating_states:
                            probs[s] = None
                        else:
                            probs[s] += math.log10(self.states[s].p_termination)

            trellis.append(probs)
            prior_probs = probs

        return trellis

    def viterbi_path(self, observed):
        """
        Establish the most probable path of states that explains a sequence
        of observations, along with the probability of that path being
        observed.

        Parameters:
          - seq_observed: a list of strings, representing an ordered sequence
            of symbols that were observed
        Returns:
          - a tuple of two values:
            - a list of state names that explains the observations
            - a float, representing the log (base 10) of the probability
              of the sequence being observed
        """
        trellis = self.trellis(observed)

        # start with the best (largest) value in the last column of the
        # trellis; we will work backwards from here
        probs = trellis[-1]
        next_state = max(probs, key=probs.get)
        state_seq = [next_state]
        p_overall = probs[next_state]

        for i in reversed(range(len(trellis)-1)):
            probs = trellis[i]
            states = probs.keys()
            # eliminate the states that cannot transition to the (known)
            # next state
            for state in states:
                if probs[state] == None:
                    del probs[state]
                elif not self._connected(state, next_state):
                    del probs[state]
            next_state = max(probs, key=probs.get)
            state_seq.append(next_state)

        state_seq.reverse()   # because the list of states was built backwards
        return (state_seq, p_overall)


if __name__ == '__main__':

    # The simple HMM shown in section 12.1 of Ewens and Grant, "Statistical
    # Methods in Bioinformatics: An Introduction, 2nd Ed.", 2005
    s1 = state(
            'S1',		# name of the state
            0.5,		# probability of being the initial state
            { '1': 0.5,		# probability of emitting a '1' at each visit
              '2': 0.5 },	# probability of emitting a '2' at each visit
            { 'S1': 0.9,	# probability of transitioning to itself
              'S2': 0.1 })	# probability of transitioning to state 'S2'
    s2 = state('S2', 0.5, {'1': 0.25, '2': 0.75}, {'S1': 0.8, 'S2': 0.2})
    h = hmm(['1', '2'],		# all symbols that can be emitted
            [s1, s2])		# all of the states in this HMM

    print "All possible sequences of states that can explain the sequence"
    print "of observed states: ['2', '2', '2']"
    h.enumerate(['2', '2', '2'])

    print
    print "The most likely sequence of states that can explain the sequence"
    print "of observed states ['2', '2', '2'], as established by the"
    print "Viterbi algorithm"
    print h.viterbi_path('222')

    # The HMM shown in figure 1 of:
    # Eddy SR.  2004.  What is a hidden Markov model?
    # Nature Biotechnology 22(10): 1315-1316.
    s1 = state('E', 1.0,
               {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
               {'E': 0.9, '5': 0.1})
    s2 = state('5', 0.0,
               {'A': 0.05, 'C': 0.0, 'G': 0.95, 'T': 0.0},
               {'I': 1.0})
    s3 = state('I', 0.0,
               {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4},
               {'I': 0.9},
               0.1)
    h = hmm(['A', 'C', 'G', 'T'], [s1, s2, s3])

    print
    print "The most likely sequence of states that can explain the sequence"
    print "of observed states 'CTTCATGTGAAAGCAGACGTAAGTCA', as established"
    print "by the Viterbi algorithm"
    print h.viterbi_path('CTTCATGTGAAAGCAGACGTAAGTCA')

    # A simple HMM to differentiate between GC-rich and GC-poor regions of
    # a genome.  Gs and Cs are represented by the symbol 'b', and As and Ts
    # are represented by the symbol 'a'.
    s1 = state('GC-poor', 0.5,
               {'a': 0.6, 'b': 0.4},
               {'GC-poor': 0.75, 'GC-rich': 0.25})
    s2 = state('GC-rich', 0.5,
               {'a': 0.35, 'b': 0.65},
               {'GC-poor': 0.25, 'GC-rich': 0.75})
    h = hmm(['a', 'b'], [s1, s2])

    print
    print "The most likely position of GC-rich and GC-poor regions in the"
    print "following genome: 'bbababbbaabbababbabbbbaabbbabaababaaabbaababaaa"
    print "aaabbaaaababbababbbaabbababbabbbbaabbbab', where an 'a' symbol"
    print "represents an A or T, and a 'b' symbol represents a G or C"
    print h.viterbi_path('bbababbbaabbababbabbbbaabbbabaababaaabbaababaaaaaabbaaaababbababbbaabbababbabbbbaabbbab')

