"""Module for analytically solving for loss of load probability and
frequency for the system described in ECEN643_Project_2019.pdf

Author: Brandon Thayer
"""

import numpy as np

########################################################################
# HELPER FUNCTIONS
########################################################################


def fill_r_diag(r):
    """Fill diagonal elements of transition rate matrix.

    :param r: square, 2D numpy.ndarray instance.
    """
    # Make sure array is square.
    assert r.shape[0] == r.shape[1]

    # Loop, and fill diagonals with negative sum of rows.
    for i in range(r.shape[0]):
        r[i, i] = -r[i, :].sum()

    return r


def get_two_state_r(fr, rr):
    """Get a transition rate matrix for a two state component.

    :param fr: failure rate
    :param rr: repair rate
    """
    return np.array([
        [-fr, fr],
        [rr, -rr]
    ])


def p_from_r(r):
    """Compute steady state probabilities given transition rate matrix.

    :param r: 2D, square, np.ndarray instance. Transition rate matrix.
    """
    # Initialize B matrix to transposed R matrix.
    b = np.copy(r.T)

    # Replace the first row with 1's.
    b[0, :] = 1

    # Initialize column vector C.
    c = np.zeros([b.shape[0], 1])

    # Replace the first element with a 1.
    c[0] = 1

    # Solve for steady state probabilities.
    return np.linalg.solve(b, c)


########################################################################
# GENERATORS
########################################################################
# Set failure and repair rates.
# (0.1/day) * (1 day / 24 hours)
g_fr = 0.1 / 24

# (1/8hr)
g_rr = 1 / 8

# G1 and G2 have four possible states.
# 0: 1U, 2U
# 1: 1D, 2U
# 2: 1U, 2D
# 3: 1D, 2D
# Initialize transition rate matrix for G1 and G2.
g12_r = np.zeros([4, 4])
# Fill in matrix.
# 0 --> 1 (failure)
g12_r[0, 1] = g_fr
# 0 --> 2 (failure)
g12_r[0, 2] = g_fr
# 1 --> 0 (repair)
g12_r[1, 0] = g_rr
# 1 --> 3 (failure)
g12_r[1, 3] = g_fr
# 2 --> 0 (repair)
g12_r[2, 0] = g_rr
# 2 --> 3 (failure)
g12_r[2, 3] = g_fr
# 3 --> 2 (repair)
g12_r[3, 2] = g_rr
# 3 --> 1 (repair)
g12_r[3, 1] = g_rr
# Fill diagonals with row sums.
g12_r = fill_r_diag(g12_r)

# Compute steady state probabilities.
g12_p = p_from_r(g12_r)

# Get the two state transition matrix for generator 3.
g3_r = get_two_state_r(fr=g_fr, rr=g_rr)

# Compute steady state probabilities for generator 3.
g3_p = p_from_r(g3_r)

########################################################################
# TRANSMISSION + WEATHER
########################################################################
# Transmission failure and repair rates for normal and adverse weather:
# 10 / year * 1 year / 8760 hours
t_fr_nw = 10 / 8760
# 100 / year * 1 year / 8760 hours
t_fr_aw = 100 / 8760
# Repair rate is 1/8hrs
t_rr = 1 / 8

# Weather rates (consider transition from normal to averse a "failure"):
w_fr = 1/200
w_rr = 1/20

# The transmission lines themselves form a four state system. Combined
# with the weather, it makes an eight state system.
# State Num.    Component States    Weather
# 0:            1U, 2U              N
# 1:            1D, 2U              N
# 2:            1U, 2D              N
# 3:            1D, 2D              N
# 4:            1U, 2U              A
# 5:            1D, 2U              A
# 6:            1U, 2D              A
# 7:            1D, 2D              A
#
# Initialize the 8x8.
tw_r = np.zeros([8, 8])
# Fill the matrix by going state by state and placing rates OUT of the
# state in question. Read diagram left to right, top to bottom.
# TODO: Put diagram in repo?
# First row of diagram:
# 2
tw_r[2, 0] = t_rr
tw_r[2, 3] = t_fr_nw
tw_r[2, 6] = w_fr
# 0
tw_r[0, 1] = t_fr_nw
tw_r[0, 2] = t_fr_nw
tw_r[0, 4] = w_fr
# 1
tw_r[1, 0] = t_rr
tw_r[1, 3] = t_fr_nw
tw_r[1, 5] = w_fr
# 3
tw_r[3, 1] = t_rr
tw_r[3, 2] = t_rr
tw_r[3, 7] = w_fr
# Second row of diagram:
# 6
tw_r[6, 2] = w_rr
tw_r[6, 4] = t_rr
tw_r[6, 7] = t_fr_aw
# 4
tw_r[4, 0] = w_rr
tw_r[4, 5] = t_fr_aw
tw_r[4, 6] = t_fr_aw
# 5
tw_r[5, 1] = w_rr
tw_r[5, 4] = t_rr
tw_r[5, 7] = t_fr_aw
# 7
tw_r[7, 3] = w_rr
tw_r[7, 5] = t_rr
tw_r[7, 6] = t_rr
# Fill diagonals.
tw_r = fill_r_diag(tw_r)

# Compute steady state probabilities for transmission states.
tw_p = p_from_r(tw_r)

########################################################################
# DEMAND (LOAD)
########################################################################
# Model demand as 4 states:
# 0: 60 MW, duration: 12 hours
# 1: 105 MW, duration: 4 hours
# 2: 205 MW, duration: 4 hours
# 3: 105 MW, duration: 4 hours
#
# Note that I'm not combining the 105MW states to make it easy to
# build the transition rate matrix.
#
# Note we're using d for demand instead of l for load, as it's easier to
# read.

# Initialize transition rate matrix for demand.
d_r = np.zeros([4, 4])
# 0 --> 1:
d_r[0, 1] = 1 / 12
# 1 --> 2:
d_r[1, 2] = 1 / 4
# 2 --> 3:
d_r[2, 3] = 1 / 4
# 3 --> 0:
d_r[3, 0] = 1 / 4
# Fill diagonals.
d_r = fill_r_diag(d_r)

# Compute steady state probabilities.
d_p = p_from_r(d_r)
pass
