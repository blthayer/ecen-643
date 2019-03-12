"""
Module for the ECEN-643 project, Spring 2019.
Author: Brandon Thayer

Note references to "the textbook" refer to Electric Power Grid
Reliability Evaluation - Models and Methods by Singh, Jirutitijaroen,
and Mitra 2019.

The file ECEN643_Project_2019.pdf describes what we're trying to
accomplish here.
"""
import numpy as np

# For reproducible results, seed the generator.
SEED = 42


def time_to_event(random_state, rho):
    """Helper function to compute the time to next event given a
    RandomState and mean^-1 of the exponential distribution.

    See equation 6.11 on page 173 of the textbook.

    :param random_state: numpy.random.RandomState instance.
    :param rho: 1/rho is the mean of the exponential distribution to
           draw from.

    :returns Time to next event in units of rho.

    Example from textbook on page 174: If random number is 0.946,
    and rho is 0.01/hr (failure rate), the time to next event (failure)
    will be ~5 hours
    """
    # x = -ln(z) / rho
    return -np.log(random_state.rand()) / rho


def get_load_level(time):
    """Determine the current load level given the time in hours.

    :param time: float representing current time in hours.

    :returns load: Current system load in MW.

    NOTE: If this weren't a simple class project, this hard-coded
    function with if/else statements would be terrible.
    """
    r = time % 24

    if 0 <= r < 4:
        return 60
    elif 4 <= r < 8:
        return 105
    elif 8 <= r < 12:
        return 205
    elif 12 <= r < 16:
        return 105
    elif 16 <= r < 24:
        return 60
    else:
        raise ValueError('Something is wrong with your logic.')


class TwoStateComponent:
    """Class for basic two-state components."""

    def __init__(self, failure_rate, repair_rate, random_state):
        """ Simply assign parameters.

        :param failure_rate: Exponentially distributed failure rate,
               units are 1/time.
        :param repair_rate: Exponentially distributed repair rate,
               units are 1/time.
        :param random_state: numpy.random.RandomState instance.
        """
        self.failure_rate = failure_rate
        self.repair_rate = repair_rate
        self.random_state = random_state

        # Whether component is up or down.
        self.up = True

    def time_to_event(self):
        """Compute the time to the next event, depending on whether or
        not the component is up.
        """
        if self.up:
            return time_to_event(random_state=self.random_state,
                                 rho=self.failure_rate)
        else:
            return time_to_event(random_state=self.random_state,
                                 rho=self.repair_rate)


def main():
    """Main method for performing Monte Carlo analysis.

    All rates and times will be in per hour.
    """
    # Create a random instance.
    r = np.random.RandomState(SEED)

    # Create two state components for each generator.
    # (0.1/day) * (1 day / 24 hours)
    gen_failure_rate = 0.1 / 24
    # (1/8hr)
    gen_repair_rate = 1 / 8
    g1 = TwoStateComponent(failure_rate=gen_failure_rate,
                           repair_rate=gen_repair_rate, random_state=r)
    g2 = TwoStateComponent(failure_rate=gen_failure_rate,
                           repair_rate=gen_repair_rate, random_state=r)
    g3 = TwoStateComponent(failure_rate=gen_failure_rate,
                           repair_rate=gen_repair_rate, random_state=r)

    # Create list of generators.
    gens = [g1, g2, g3]

    # Create list of all components.
    components = [*gens]

    # Draw time to event for each component.
    times = np.array([x.time_to_event() for x in components])

    # Initialize the time to 0.
    time = 0
    it_count = 0

    # TODO: Update for convergence criteria.
    while time < 1 and it_count < 5:
        # Figure out which component will be changing state next.
        min_time_idx = np.argmin(times)

        # Update the time.
        time += times[min_time_idx]

        # Toggle the state of the component.
        # noinspection PyTypeChecker
        this_component = components[min_time_idx]
        this_component.up = not this_component.up

        # Redraw time for this component.
        times[min_time_idx] = this_component.time_to_event()

        # Get the current load level for this time.
        load = get_load_level(time)

        # Determine if we're able to meet load.

        # Update iteration counter.
        it_count += 1

    pass


if __name__ == '__main__':
    main()

    print('hooray')
