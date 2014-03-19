#!/usr/bin/env python

# Copyright (c) 2014 Michael Strosaker
# MIT License
# http://opensource.org/licenses/MIT

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

sys.path.insert(0, '..')
import hmm

class TestHMMConstructor(unittest.TestCase):

    def test_simple_hmm(self):
        # The simple HMM shown in section 12.1 of Ewens and Grant, "Statistical
        # Methods in Bioinformatics: An Introduction, 2nd Ed.", 2005
        s1 = hmm.state(
                'S1',            # name of the state
                0.5,             # probability of being the initial state
                { '1': 0.5,      # probability of emitting a '1' at each visit
                  '2': 0.5 },    # probability of emitting a '2' at each visit
                { 'S1': 0.9,     # probability of transitioning to itself
                  'S2': 0.1 })   # probability of transitioning to state 'S2'
        s2 = hmm.state('S2', 0.5,
                { '1': 0.25, '2': 0.75 },
                { 'S1': 0.8, 'S2': 0.2 })
        model = hmm.hmm(['1', '2'],  # all symbols that can be emitted
                        [s1, s2])    # all of the states in this HMM

        self.assertEqual(model.states['S1'].p_initial, 0.5)
        self.assertEqual(model.states['S2'].p_transition['S1'], 0.8)
        self.assertEqual(len(model.initial_states), 2)
        self.assertFalse(model.terminal_state)
        self.assertEqual(len(model.terminating_states), 0)

    def test_implied_terminal_state(self):
        # The HMM shown in figure 1 of:
        # Eddy SR.  2004.  What is a hidden Markov model?
        # Nature Biotechnology 22(10): 1315-1316.
        s1 = hmm.state('E', 1.0,
                       {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
                       {'E': 0.9, '5': 0.1})
        s2 = hmm.state('5', 0.0,
                       {'A': 0.05, 'C': 0.0, 'G': 0.95, 'T': 0.0},
                       {'I': 1.0})
        s3 = hmm.state('I', 0.0,
                       {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4},
                       {'I': 0.9},
                       0.1)
        model = hmm.hmm(['A', 'C', 'G', 'T'], [s1, s2, s3])

        self.assertEqual(len(model.initial_states), 1)
        self.assertTrue(model.terminal_state)
        self.assertEqual(len(model.terminating_states), 1)

class TestViterbiPath(unittest.TestCase):

    def test_simple_hmm(self):
        s1 = hmm.state('S1', 0.5,
                       { '1': 0.5, '2': 0.5 },
                       { 'S1': 0.9, 'S2': 0.1 })
        s2 = hmm.state('S2', 0.5,
                       { '1': 0.25, '2': 0.75 },
                       { 'S1': 0.8, 'S2': 0.2 })
        model = hmm.hmm(['1', '2'], [s1, s2])

        state_path, prob = model.viterbi_path('222')

        self.assertEqual(state_path, ['S2', 'S1', 'S1'])
        self.assertEqual(round(prob, 6), -1.170696)

    def test_implied_terminal_state(self):
        s1 = hmm.state('E', 1.0,
                       {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
                       {'E': 0.9, '5': 0.1})
        s2 = hmm.state('5', 0.0,
                       {'A': 0.05, 'C': 0.0, 'G': 0.95, 'T': 0.0},
                       {'I': 1.0})
        s3 = hmm.state('I', 0.0,
                       {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4},
                       {'I': 0.9},
                       0.1)
        model = hmm.hmm(['A', 'C', 'G', 'T'], [s1, s2, s3])

        state_path, prob = model.viterbi_path('CTTCATGTGAAAGCAGACGTAAGTCA')

        self.assertEqual(state_path, ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
                         'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', '5',
                         'I', 'I', 'I', 'I', 'I', 'I', 'I'])
        self.assertEqual(round(prob, 10), -17.9014785649)

class TestHMMRepr(unittest.TestCase):

    def test_repr(self):
        s1 = hmm.state('E', 1.0,
                       {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
                       {'E': 0.9, '5': 0.1})
        s2 = hmm.state('5', 0.0,
                       {'A': 0.05, 'C': 0.0, 'G': 0.95, 'T': 0.0},
                       {'I': 1.0})
        s3 = hmm.state('I', 0.0,
                       {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4},
                       {'I': 0.9},
                       0.1)
        model = hmm.hmm(['A', 'C', 'G', 'T'], [s1, s2, s3])

        model2 = eval(repr(model))

        self.assertEqual(model.alphabet, model2.alphabet)
        self.assertEqual(model.states['E'].p_transition['5'],
                         model2.states['E'].p_transition['5'])
        self.assertEqual(model.states['5'].p_emission['T'],
                         model2.states['5'].p_emission['T'])
        self.assertEqual(model.states['I'].p_termination,
                         model2.states['I'].p_termination)

class TestHMMTraining(unittest.TestCase):

    def test_implied_terminal_state(self):
        training_data = [('CTTCATGTGAAAGCAGACGTAAGTCA',
                          'EEEEEEEEEEEEEEEEEE5IIIIIII'),
                         ('CTTCATGTGAAAGCAGACATAAGTCA',
                          'EEEEEEEEEEEEEEEEEE5IIIIIII')]
        model = hmm.train_hmm(training_data, True)

        self.assertIn('A', model.alphabet)
        self.assertIn('C', model.alphabet)
        self.assertIn('G', model.alphabet)
        self.assertIn('T', model.alphabet)
        self.assertNotIn('B', model.alphabet)
        self.assertIn('E', model.states.keys())
        self.assertIn('5', model.states.keys())
        self.assertIn('I', model.states.keys())
        self.assertEqual(model.states['5'].p_emission['A'], 0.5)
        self.assertEqual(model.states['5'].p_transition['I'], 1.0)
        self.assertEqual(model.initial_states, ['E'])
        self.assertTrue(model.terminal_state)
        self.assertEqual(model.terminating_states, ['I'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

