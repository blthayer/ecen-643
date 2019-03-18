"""Module for analytically solving for loss of load probability and
frequency for the system described in ECEN643_Project_2019.pdf

Author: Brandon Thayer
"""

import numpy as np
import pandas as pd

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
    c[0, 0] = 1

    # Solve for steady state probabilities.
    return np.linalg.solve(b, c)


def compute_capacity(g12_cap, g3_cap, t_cap):
    """Helper to compute system capacity to deliver load."""
    return min(g12_cap, t_cap) + g3_cap


def main():
    ####################################################################
    # GENERATORS
    ####################################################################
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

    # Manually assign capacities to each state.
    g12_c = np.array([150, 75, 75, 0])

    # Get the two state transition matrix for generator 3.
    g3_r = get_two_state_r(fr=g_fr, rr=g_rr)

    # Compute steady state probabilities for generator 3.
    g3_p = p_from_r(g3_r)

    # Manually assign capacities to each state.
    g3_c = np.array([75, 0])

    ####################################################################
    # TRANSMISSION + WEATHER
    ####################################################################
    # Transmission failure and repair rates for normal and adverse
    # weather:
    # 10 / year * 1 year / 8760 hours
    t_fr_nw = 10 / 8760
    # 100 / year * 1 year / 8760 hours
    t_fr_aw = 100 / 8760
    # Repair rate is 1/8hrs
    t_rr = 1 / 8

    # Weather rates (consider transition from normal to averse a
    # "failure"):
    w_fr = 1/200
    w_rr = 1/20

    # The transmission lines themselves form a four state system.
    # Combined with the weather, it makes an eight state system.
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
    # Fill the matrix by going state by state and placing rates OUT of
    # the state in question. Read diagram left to right, top to bottom.
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

    # Manually assign capacities to each state.
    tw_c = np.array([200, 100, 100, 0, 200, 100, 100, 0])

    ####################################################################
    # DEMAND (LOAD)
    ####################################################################
    # Model demand as 4 states:
    # 0: 60 MW, duration: 12 hours
    # 1: 105 MW, duration: 4 hours
    # 2: 205 MW, duration: 4 hours
    # 3: 105 MW, duration: 4 hours
    #
    # Note that I'm not combining the 105MW states to make it easy to
    # build the transition rate matrix.
    #
    # Note we're using d for demand instead of l for load, as it's
    # easier to read.

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

    # Manually assign demand for each state.
    d_d = np.array([60, 105, 205, 105])

    ####################################################################
    # COMPUTE STATE PROBABILITIES AND DETERMINE IF DEMAND IS SERVED
    ####################################################################
    # Do some nasty, hard-coded nested for-loops to get all possible
    # system states, their probabilities, and whether or not demand
    # is served.

    # Initialize list which will track states, probabilities,
    # loss of load, etc.
    data = []

    # Loop over g12.
    for g12_idx in range(len(g12_c)):

        # Loop over g3.
        for g3_idx in range(len(g3_c)):

            # Loop over transmission + weather.
            for tw_idx in range(len(tw_c)):

                # Compute system capacity to serve demand.
                system_capacity = \
                    compute_capacity(g12_cap=g12_c[g12_idx],
                                     g3_cap=g3_c[g3_idx],
                                     t_cap=tw_c[tw_idx])

                # Loop over demand.
                for d_idx in range(len(d_d)):

                    # Determine whether or not demand is met.
                    demand_met = system_capacity >= d_d[d_idx]

                    # Compute the probability of being in this
                    # system state.
                    p = (g12_p[g12_idx, 0] * g3_p[g3_idx, 0]
                         * tw_p[tw_idx, 0] * d_p[d_idx, 0])

                    # Create data entry. BE CAREFUL! The order of this
                    # array needs to match the order of the "columns"
                    # argument given to the DataFrame constructor.
                    data.append([g12_idx, g3_idx, tw_idx, d_idx, demand_met,
                                 p])

    # Create DataFrame with all system states.
    df = pd.DataFrame(data, columns=['g12_state', 'g3_state', 'tw_state',
                                     'd_state', 'demand_met', 'probability'])

    ####################################################################
    # COMPUTE LOSS OF LOAD PROBABILITY
    ####################################################################
    # Grab and sum the probabilities of states which don't meet demand.
    lolp = df[~df['demand_met']]['probability'].sum()

    print('Loss of Load Probability: {:.4f}'.format(round(lolp, 4)))

    ####################################################################
    # COMPUTE FREQUENCY OF LOSS OF LOAD
    ####################################################################
    # I'm not happy with the efficiency/optimality of this section,
    # but hey, it gets the job done and I don't have time to
    # over-optimize.
    #
    # I'm going to use the matrix approach.
    # f_f = U * A^bar * Q, where:
    #
    # f_f: frequency of failure
    # U: row vector, u_i = 1 if i within failed states, else u_i = 0
    # A^bar: transpose of transition rate matrix, but with 0's on the
    #        diagonal.
    # Q: column vector of state probabilities, except p_i = 0 if i is
    #    within the failed states.

    # Initialize the transition matrix for the entire system.
    # NOTE: We really just want A^bar, so we won't actually fill in the
    # diagonals at the end.
    r_system = np.zeros([df.shape[0], df.shape[0]])

    # Define columns of states.
    state_cols = ['g12_state', 'g3_state', 'tw_state', 'd_state']

    # Map the state columns to the transition rate matrices.
    r_dict = {'g12_state': g12_r, 'g3_state': g3_r, 'tw_state': tw_r,
              'd_state': d_r}

    # Initialize a DataFrame that will shrink as we iterate, just to
    # show I have a clue about efficiency despite the fact I haven't
    # put much efficiency effort in so far. However, note that the
    # overhead required to drop DataFrame rows as we go may be greater
    # than just doing excessive iteration.
    shrinking_df = df.copy(deep=True)

    # Loop over each row of the DataFrame. I'm not claiming this is
    # efficient, just trying to get the job done.
    for i in df.index:
        # Extract this row.
        row = df.loc[i]

        shrinking_df.drop(labels=i, axis=0, inplace=True)

        # Take the absolute difference in states with remaining rows.
        state_diff = (row[state_cols] - shrinking_df[state_cols]).abs()

        # Get a mask so we pull rows that only change in one column.
        mask = (state_diff > 0).sum(axis=1) == 1

        # Extract the rows which only changed in one column.
        one_col_change = shrinking_df[mask]

        # Find the column which changed.
        col_idx = state_diff[mask].idxmax(axis=1)

        # Loop over rows in one_col_change and update r_system.
        for j in one_col_change.index:
            # Grab this column.
            this_col = col_idx[j]

            # Grab the appropriate transition rate matrix.
            this_r = r_dict[this_col]

            # Extract state indices for i and j.
            s_i = row[this_col]
            s_j = one_col_change.loc[j, this_col]

            # Update r_system
            r_system[i, j] = this_r[s_i, s_j]

            # Update [j, i]
            r_system[j, i] = this_r[s_j, s_i]

    # We now have r_system, but with zero's on the diagonal.
    a_bar = r_system.T

    # Create the U vector.
    u = np.zeros(df.shape[0])
    u[~df['demand_met']] = 1

    # Create the q vector.
    q = np.copy(df['probability'].values)
    q[~df['demand_met']] = 0

    # Compute frequency of failure. NOTE: This will be in units of 1/hr.
    f_f = np.matmul(np.matmul(u.T, a_bar), q)

    print('Frequency of Load Loss: {:.4f}/year'.format(f_f * 8760))
    pass


if __name__ == '__main__':
    main()
