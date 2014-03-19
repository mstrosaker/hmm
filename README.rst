Introduction: what is hmm?
--------------------------

**hmm** is a pure-Python module for constructing hidden Markov models.
It provides the ability to create arbitrary HMMs of a specified topology,
and to calculate the most probable path of states that explains a given
sequence of observations using the Viterbi algorithm, or by enumerating
every possible path (for small models and short observations).  It also
provides the ability to construct HMMs empirically, based on annotated
observations.

Prerequisites
-------------

**hmm** has been tested with Python version 2.6.

Installing
----------

The source distribution for the most recent version can be obtained from
the `hclust project page <https://github.com/mstrosaker/hmm>`_  by
clicking on the Download ZIP button.  The module can be installed with::

    > python setup.py install

Since **hmm** is a work in progress, it's recommended to have the most
recent version of the code.

How to use it?
--------------

**hmm** is a regular Python module; you import and invoke it from your
own code.  For a detailed usage guide and examples, please consult the
`user's guide <htps://github.com/mstrosaker/hmm/wiki/User's-guide>`_.

License
-------

Copyright (c) 2014 Michael Strosaker.  See the LICENSE file for license
rights and limitations (MIT).

