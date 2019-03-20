"""
Module for the Monte Carlo portion of the ECEN-643 project, Spring 2019.
Author: Brandon Thayer

Note references to "the textbook" refer to Electric Power Grid
Reliability Evaluation - Models and Methods by Singh, Jirutitijaroen,
and Mitra, 2019.

The file ECEN643_Project_2019.pdf describes what we're trying to
accomplish here.

This module has plenty of hard-coding in it. I'm actively fighting my
urge to make everything general, and accepting that this is a project
with a static specification, and life will be better if I hard-code.
"""
import numpy as np
import logging as log

# For reproducible results, seed the random number generator.
SEED = 42

# Set log level and create module level logger.
LOG_LEVEL = log.INFO
log.basicConfig(level=LOG_LEVEL)


def time_to_event(*, rho, rn):
    """Helper function to compute the time to next event given a
    RandomState and mean^-1 of the exponential distribution.

    See equation 6.11 on page 173 of the textbook.

    :param rho: 1/rho is the mean of the exponential distribution to
           draw from.
    :param rn: Random number on interval [0, 1)

    :returns Time to next event in units of rho.

    Example from textbook on page 174: If random number is 0.946,
    and rho is 0.01/hr (failure rate), the time to next event (failure)
    will be ~5 hours
    """

    # x = -ln(z) / rho
    return -np.log(rn) / rho


def load_served(gens, t_lines, load):
    """Determine whether or not load can be served depending on the
    state of the generators, transmission lines, and system loading.

    :param gens: List of the three generators. Each generator is
           represented by a TwoStateComponent object. gens[0] is G1,
           gens[1] is G2, and gens[2] is G3.
    :param t_lines: List of the two transmission lines. Each line is
           represented by a TransmissionLine object.
    :param load: Current system loading level in MW.

    :return: True if load can be served, False if it cannot.

    NOTE: There's some nasty hard-coding in here. This function could
    be a LOT better if this weren't a simple, static, class project.
    """
    # Compute available capacity for generators 1 and 2.
    gen_import = 0
    for g in gens[0:2]:
        if g.state:
            gen_import += 75

    # Compute available transmission capacity.
    trans_capacity = 0
    for t in t_lines:
        if t.state:
            trans_capacity += 100

    # The power that can be imported over the lines is the minimum of
    # the generation capacity and transmission capacity.
    import_capacity = min(gen_import, trans_capacity)

    # Compute power that can be served by generator 3.
    g3 = 0
    if gens[2].state:
        g3 += 75

    # Total capacity is import capacity plus capacity from generator 3.
    total_capacity = import_capacity + g3

    log.debug('Total capacity: {}, Load: {}'.format(total_capacity, load))

    return total_capacity >= load


class TwoStateComponent:
    """Class for basic two-state components."""

    def __init__(self, failure_rate, repair_rate, random_state, name=''):
        """ Simply assign parameters.

        :param failure_rate: Exponentially distributed failure rate,
               units are 1/time.
        :param repair_rate: Exponentially distributed repair rate,
               units are 1/time.
        :param random_state: numpy.random.RandomState instance.
        """
        # Failure rate uses a getter so it can be overridden by child
        # classes (e.g. TransmissionLine)
        self._failure_rate = failure_rate
        # Simply set remaining attributes.
        self.repair_rate = repair_rate
        self.random_state = random_state
        self.name = name

        # Whether component is up or down - start up.
        self.state = True

        # Set the time to the next event. Note tte has both a getter and
        # a setter.
        self.tte = self.time_to_event()

    def __repr__(self):
        return self.name

    # Define a getter for failure_rate so it can be overridden by
    # children classes.
    @property
    def failure_rate(self):
        return self._failure_rate

    # This property will also be overridden by TransmissionLine
    @property
    def tte(self):
        """Time to event."""
        return self._tte

    @tte.setter
    def tte(self, value):
        self._tte = value

    def time_to_event(self):
        """Compute the time to the next event, depending on self.state.
        """
        # Choose the rate based on state.
        if self.state:
            rate = self.failure_rate
        else:
            rate = self.repair_rate

        # Draw tte.
        tte = time_to_event(rn=self.random_state.rand(), rho=rate)

        # Return.
        return tte

    def change_state_update_tte(self):
        """Flip self.state, update self.tte."""
        self.state = not self.state
        self.tte = self.time_to_event()

    def update_tte(self, delta):
        """Update time to event (tte) given a change in time (delta).
        """
        # Decrease tte.
        tte = self.tte - delta

        # If tte went negative, set to 0. This should only happen for
        # transmission lines when the weather changes state.
        if tte < 0:
            log.debug('TTE for {} is < 0. Setting to 0.'.format(self.name))
            tte = 0

        # Set tte.
        self.tte = tte


class TransmissionLine(TwoStateComponent):
    """Class for transmission lines, which have different rates based
    on the weather.

    I'm wondering if inheritance was really the way to go here. It made
    this slightly complicated, though TwoStateComponents would have been
    rewritten otherwise...
    """

    def __init__(self, weather, fr_nw, fr_aw, rr, random_state, name=''):
        """

        :param weather: TwoStateComponent representing the weather.
               If weather.state = True, weather is normal. If
               weather.state is false, weather is adverse.
        :param fr_nw: Failure rate for normal weather.
        :param fr_aw: Failure rate for adverse weather.
        :param rr: Repair rate.
        :param random_state: numpy.random.RandomState object.
        """
        # Create reference to weather object.
        self.weather = weather

        # Track the weather's previous state.
        self.prev_weather_state = weather.state

        # Assign rates.
        self.fr_nw = fr_nw
        self.fr_aw = fr_aw

        # Call the super constructor. Note failure_rate has a getter
        # which determines failure rate based on the weather.
        super().__init__(failure_rate=self.failure_rate, repair_rate=rr,
                         random_state=random_state, name=name)

    # Override the TwoStateComponent's getter for failure_rate.
    @TwoStateComponent.failure_rate.getter
    def failure_rate(self):
        # Use different failure rates depending on the weather.
        if self.weather.state:
            return self.fr_nw
        else:
            return self.fr_aw

    # Override the TwoStateComponent's getter method for tte.
    @TwoStateComponent.tte.getter
    def tte(self):
        # If the weather's state has changed, we need to update our tte.
        if self.prev_weather_state != self.weather.state:

            # Draw new tte (note this uses the setter to set self._tte).
            self.tte = self.time_to_event()

            # Reset the weather's previous state.
            self.prev_weather_state = self.weather.state

            log.debug('Transmission line rate and tte changed due to weather.')

        return self._tte


class Load:
    """Class for representing the demand. Property names and methods
    will line up with TwoStateComponent so it can be used in the same
    way.
    """
    # Start times for each load condition.
    START_TIMES = np.array([0, 4, 8, 12, 16])

    # Load corresponding to the end times.
    LOAD = np.array([60, 105, 205, 105, 60])

    def __init__(self):
        # Hard-code our starting index to be 0.
        self.idx = 0

        # Set load state.
        self.state = self.LOAD[self.idx]

        # Update the tte.
        self.tte = self.time_to_event()

        # Hard-code the name.
        self.name = 'Load'

    def __repr__(self):
        return self.name

    def time_to_event(self):
        # Compute the time between the current time and the next
        # event.
        try:
            tte = self.START_TIMES[self.idx + 1] - self.START_TIMES[self.idx]
        except IndexError:
            # If self.idx + 1 is out of bounds, we're wrapping the day.
            tte = 24 - self.START_TIMES[self.idx]

        return tte

    def change_state_update_tte(self):
        # Update index.
        idx = self.idx + 1

        # If the index has hit the length of the array, reset it.
        if idx == self.START_TIMES.shape[0]:
            idx = 0

        # Set new index.
        self.idx = idx

        # Update state.
        self.state = self.LOAD[self.idx]

        # Update tte.
        self.tte = self.time_to_event()

    def update_tte(self, delta):
        self.tte = self.tte - delta


def main():
    """Main method for performing Monte Carlo analysis.

    All rates and times will be in per hour.
    """
    # Create a random instance.
    random_state = np.random.RandomState(SEED)

    ####################################################################
    # GENERATORS
    ####################################################################
    # Set failure and repair rates.
    # (0.1/day) * (1 day / 24 hours)
    gen_failure_rate = 0.1 / 24

    # (1/8hr)
    gen_repair_rate = 1 / 8

    # Create list of three identical generators via list comprehension.
    gens = [
        TwoStateComponent(failure_rate=gen_failure_rate,
                          repair_rate=gen_repair_rate,
                          random_state=random_state, name='G{}'.format(i + 1))
        for i in range(3)
    ]

    ####################################################################
    # WEATHER
    ####################################################################

    # Create two state component for weather.
    weather = TwoStateComponent(failure_rate=1/200, repair_rate=1/20,
                                random_state=random_state, name='Weather')

    ####################################################################
    # TRANSMISSION LINES
    ####################################################################

    # Get failure rates for normal and adverse weather.
    # 10 / year * 1 year / 8760 hours
    fr_nw = 10 / 8760
    # 100 / year * 1 year / 8760 hours
    fr_aw = 100 / 8760
    # Repair rate is 1/8hrs
    rr = 1 / 8

    # Use list comprehension to create list of two identical
    # transmission lines.
    t_lines = [
        TransmissionLine(weather=weather, fr_nw=fr_nw, fr_aw=fr_aw,
                         rr=rr, random_state=random_state,
                         name='T{}'.format(i+1))
        for i in range(2)
    ]

    ####################################################################
    # LOAD
    ####################################################################
    # Not much to do, just initialize the object.
    load = Load()

    ####################################################################
    # PERFORM MONTE CARLO
    ####################################################################
    # Flow:
    # - Determine time to next event.
    # - If load was not served between the last event and this upcoming
    #   one, update running time of un-served load.
    # - Make the next event occur by changing the state of the relevant
    #   component.
    # - Update time.
    # - Update tte's for all other components.
    # - Determine whether or not load is served.

    # Create list of all components.
    components = [*gens, weather, *t_lines, load]

    # Initialize the time to 0.
    time = 0
    total_time = 0
    it_count = 0

    # Initialize variables for handling un-served load.
    time_unserved = 0
    # ls --> load served. Hard-coding that we start serving all load.
    ls = True
    # In order to track frequency, we need to look for changes in ls.
    prev_ls = True
    # We'll also count loss of load (lol) occurrences.
    lol_count = 0

    # Initialize variables for tracking convergence
    # tus --> time un-served
    tus_arr = np.array([])
    # Compute the average tus for each time year.
    tus_avg_arr = np.array([])
    year = 0
    cov_arr = np.array([])

    # Note that we have a "break" statement in the loop, so we'll loop
    # for some large, finite count.
    while it_count < 100000:
        # Sort components based on their time to event (tte). The
        # component with the shortest time to event will be in position
        # 0.
        # NOTE: This may be faster if we used a priority queue.
        components.sort(key=lambda x: x.tte)

        # Extract the component with the minimum time to event.
        this_component = components[0]

        # Grab time delta to next event
        delta = this_component.tte

        # Sanity check:
        if delta < 0:
            raise ValueError('delta is < 0 ?')

        # Check if the previous loop iteration failed to serve load.
        if not ls:
            # Log the unserved load event.
            log.debug('Load could not be served from time '
                      + '{:.2f} to {:.2f}!'.format(time, time + delta))

            # Increment the unserved time by the time to the next
            # event.
            time_unserved += delta

            # Check to see if this is a new loss of load event.
            if prev_ls:
                # Increment the loss of load counter if this is a loss
                # of load event, but load was being served before.
                lol_count += 1

        # Set prev_ls to ls.
        prev_ls = ls

        # Grab pre-event state of component.
        pre_event_state = this_component.state

        # Toggle the state of the component and update it's tte.
        this_component.change_state_update_tte()

        # Check to see if we've moved into a new year (hard-code 8760)
        if time > 8760:
            # If we've moved into a new year, compute loss of load
            # information for convergence criterion.

            # Log it.
            log.info('Year {} complete.'.format(year))

            # Bump the year counter and total time, reset time.
            year += 1
            total_time += time
            time = 0

            # Put the unserved time in the array and reset it.
            tus_arr = np.append(tus_arr, time_unserved)
            time_unserved = 0

            # Compute the mean unserved time thus far, and append to the
            # appropriate array. A running total would be more
            # efficient, but oh well.
            tus_avg = tus_arr.mean()
            tus_avg_arr = np.append(tus_avg_arr, tus_avg)

            # Compute the estimated coefficient of variation.
            cov = np.std(tus_avg_arr) / np.mean(tus_arr)
            cov_arr = np.append(cov_arr, cov)

            # Log.
            log.info('Coefficient of variation: {:.3f}'.format(cov))

            # We're done if less than 0.05.
            if cov < 0.05 and year > 1:
                log.info('Exiting Monte Carlo simulation.')
                break

        else:
            # We haven't crossed into a new year - increment the time.
            time += delta

        # Update tte for all other components.
        for c in components[1:]:
            c.update_tte(delta)

        # Log event.
        log_str = (
            'At time {:.2f}, {} transitioned from {} to {}'
            ).format(time, this_component.name, pre_event_state,
                     this_component.state)
        log.debug(log_str)

        # Determine if we're able to meet load. ls --> "load_served"
        ls = load_served(gens, t_lines, load.state)

        # Update iteration counter.
        it_count += 1

    log.info('While loop broken after {} iterations.'.format(it_count))
    log.info('Total simulation "time": {} hours'.format(total_time))

    # Report final statistics.
    print('\n' + '*' * 80)
    print('Loss of Load Probability: {:.4f}'.format(
        round(tus_arr.sum()/total_time, 4)))
    print('Frequency of Load Loss: {:.4f}/year'.format(
        round(lol_count / (total_time/8760), 4)))
    pass


if __name__ == '__main__':
    main()
