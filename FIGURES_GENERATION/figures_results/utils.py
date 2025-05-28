import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def gsigmoid(V, A, B, C, D):
    """
    Generalized sigmoid function.

    Computes a sigmoid function with adjustable parameters.

    Parameters
    ----------
    V : float or np.ndarray
        The input value(s) for the function.
    A : float
        The vertical shift (baseline value).
    B : float
        The amplitude of the sigmoid.
    C : float
        The steepness of the curve (scaling factor).
    D : float
        The horizontal shift of the sigmoid.

    Returns
    -------
    float or np.ndarray
        The result of the sigmoid function applied to `V`.

    Notes
    -----
    The generalized sigmoid function is given by:
        gsigmoid(V, A, B, C, D) = A + B / (1 + exp((V + D) / C))
    """
    return A + B / (1 + np.exp((V + D) / C))


def d_gsigmoid(V, A, B, C, D):
    """
    Derivative of the generalized sigmoid function.

    Computes the derivative of the sigmoid function with respect to `V`.

    Parameters
    ----------
    V : float or np.ndarray
        The input value(s) for the function.
    A : float
        The vertical shift (baseline value) of the sigmoid.
    B : float
        The amplitude of the sigmoid.
    C : float
        The steepness of the curve (scaling factor).
    D : float
        The horizontal shift of the sigmoid.

    Returns
    -------
    float or np.ndarray
        The derivative of the sigmoid function at `V`.

    Notes
    -----
    The derivative is given by:
        d_gsigmoid(V, A, B, C, D) = -B * exp((V + D) / C) /
                                    (C * (1 + exp((V + D) / C))^2)
    """
    return -B * np.exp((V + D) / C) / (C * (1 + np.exp((V + D) / C)) ** 2)


def gamma_uniform_mean_std_matching(uniform_a, uniform_b):
    """
    Match a gamma distribution's parameters to a uniform distribution.

    Solves for the shape (k) and scale (theta) parameters of a gamma
    distribution that matches the mean and variance of a uniform distribution
    over the interval [a, b].

    Parameters
    ----------
    uniform_a : float
        The lower bound of the uniform distribution.
    uniform_b : float
        The upper bound of the uniform distribution.

    Returns
    -------
    k : float
        The shape parameter of the gamma distribution.
    theta : float
        The scale parameter of the gamma distribution.

    Notes
    -----
    The uniform distribution has:
        - Mean: (a + b) / 2
        - Variance: (b - a)^2 / 12

    The gamma distribution is parameterized as:
        p(x) = x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k)),
    where the mean is k * theta and the variance is k * theta^2.

    The parameters are solved as:
        k = 3 * (a + b)^2 / (b - a)^2
        theta = (b - a)^2 / (6 * (a + b))
    """
    a = uniform_a
    b = uniform_b
    
    p = a + b
    q_sq = (b - a) ** 2
    k = 3 * p ** 2 / q_sq
    theta = q_sq / (6 * p)
    return k, theta


# == simulation utils functions ==

def simulate_population_multiprocessing(simulation_function, population, u0, T_final, dt, params, max_workers=8, verbose=False):
    """
    Simulate a population using multiprocessing over a fixed time duration.

    Parameters
    ----------
    simulation_function : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    T_final : float
        The final time of the simulation.
    dt : float
        The time step for the simulation.
    params : dict
        Additional parameters required by the simulation function.
    max_workers : int, optional
        The maximum number of worker processes for multiprocessing. Defaults to 8.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function divides the population among multiple processes and evaluates
    the simulation function for each individual in parallel.
    """
    return simulate_population_t_eval_multiprocessing(simulation_function, population, u0, np.arange(0, T_final, dt), params, max_workers, verbose)


def simulate_population_t_eval_multiprocessing(simulation_function, population, u0, t_eval, params, max_workers=8, verbose=False):
    """
    Simulate a population using multiprocessing over specified evaluation times.

    Parameters
    ----------
    simulation_function : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    t_eval : array-like
        An array of time points at which to evaluate the simulation.
    params : dict
        Additional parameters required by the simulation function.
    max_workers : int, optional
        The maximum number of worker processes for multiprocessing. Defaults to 8.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function uses `ProcessPoolExecutor` to parallelize the simulations for
    the population. Each individual in the population is simulated independently.
    """
    traces = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(u0, individual, t_eval, params) for individual in population]
        results = list(
            tqdm(
                executor.map(simulation_function, tasks),
                total=len(population),
                desc='Simulating population (multiprocessing)',
                disable=not verbose  # Disable tqdm if verbose is False
            )
        )
    for result in results:
        traces.append(result)
    return traces


def simulate_population(simulation_function, population, u0, T_final, dt, params, verbose=False):
    """
    Simulate a population sequentially over a fixed time duration.

    Parameters
    ----------
    simulation_function : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    T_final : float
        The final time of the simulation.
    dt : float
        The time step for the simulation.
    params : dict
        Additional parameters required by the simulation function.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function simulates each individual in the population sequentially.
    """
    return simulate_population_t_eval(simulation_function, population, u0, np.arange(0, T_final, dt), params, verbose)


def simulate_population_t_eval(simulation, population, u0, t_eval, params, verbose=False):
    """
    Simulate a population sequentially over specified evaluation times.

    Parameters
    ----------
    simulation : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    t_eval : array-like
        An array of time points at which to evaluate the simulation.
    params : dict
        Additional parameters required by the simulation function.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function iterates over the population and evaluates the simulation
    function for each individual sequentially.
    """
    traces = []
    for i in tqdm(range(len(population)), desc='Simulating population', disable=not verbose):
        individual = population[i]
        trace = simulation([u0, individual, t_eval, params])
        traces.append(trace)
    return traces


# == DICs computation related utils functions ==

def w_factor(V, tau_x, tau_1, tau_2, default=1):
    """
    Compute the weighting factor based on dynamic tau values.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : callable
        A function that computes tau_x(V).
    tau_1 : callable
        A function that computes tau_1(V).
    tau_2 : callable
        A function that computes tau_2(V).
    default : float, optional
        The default value to assign when no conditions are met. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        An array of weighting factors corresponding to each value in `V`.

    Notes
    -----
    - The weighting factor is calculated based on logarithmic differences
      between tau_x(V), tau_1(V), and tau_2(V).
    - If tau_x(V) > tau_2(V), the weighting factor is set to 0.
    - If tau_x(V) is between tau_1(V) and tau_2(V), the weighting factor is
      calculated proportionally.
    """
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x(V) > tau_1(V)) & (tau_x(V) <= tau_2(V))
    mask_2 = tau_x(V) > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1])) - np.log(tau_x(V[mask_1]))) / (np.log(tau_2(V[mask_1])) - np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result


def w_factor_constant_tau(V, tau_x, tau_1, tau_2, default=1):
    """
    Compute the weighting factor based on constant tau_x and dynamic tau_1 and tau_2.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : float
        A constant value for tau_x.
    tau_1 : callable
        A function that computes tau_1(V).
    tau_2 : callable
        A function that computes tau_2(V).
    default : float, optional
        The default value to assign when no conditions are met. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        An array of weighting factors corresponding to each value in `V`.

    Notes
    -----
    - The weighting factor is calculated based on logarithmic differences
      between tau_x, tau_1(V), and tau_2(V).
    - If tau_x > tau_2(V), the weighting factor is set to 0.
    - If tau_x is between tau_1(V) and tau_2(V), the weighting factor is
      calculated proportionally.
    """
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x > tau_1(V)) & (tau_x <= tau_2(V))
    mask_2 = tau_x > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1])) - np.log(tau_x)) / (np.log(tau_2(V[mask_1])) - np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result


def get_w_factors(V, tau_x, tau_f, tau_s, tau_u):
    """
    Compute two weighting factors using dynamic tau values.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : callable
        A function that computes tau_x(V).
    tau_f : callable
        A function that computes tau_f(V).
    tau_s : callable
        A function that computes tau_s(V).
    tau_u : callable
        A function that computes tau_u(V).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - The first array is the weighting factor for (tau_f, tau_s).
        - The second array is the weighting factor for (tau_s, tau_u).
    """
    return w_factor(V, tau_x, tau_f, tau_s, default=1), w_factor(V, tau_x, tau_s, tau_u, default=1)


def get_w_factors_constant_tau(V, tau_x, tau_f, tau_s, tau_u):
    """
    Compute two weighting factors using a constant tau_x and dynamic tau values.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : float
        A constant value for tau_x.
    tau_f : callable
        A function that computes tau_f(V).
    tau_s : callable
        A function that computes tau_s(V).
    tau_u : callable
        A function that computes tau_u(V).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - The first array is the weighting factor for (tau_f, tau_s).
        - The second array is the weighting factor for (tau_s, tau_u).
    """
    return w_factor_constant_tau(V, tau_x, tau_f, tau_s), w_factor_constant_tau(V, tau_x, tau_s, tau_u)

# == Analytical and numerical utils functions ==

def bisection(f, a, b, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    """
    Finds the root of a continuous function using the bisection method.

    Parameters
    ----------
    f : callable
        The function for which to find the root. It must be continuous on the interval [a, b].
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    y_tol : float, optional
        The tolerance for the absolute value of the function at the root. Defaults to 1e-6.
    x_tol : float, optional
        The tolerance for the interval width. Defaults to 1e-6.
    max_iter : int, optional
        The maximum number of iterations. Defaults to 1000.
    verbose : bool, optional
        If True, prints a message if the method fails to converge. Defaults to False.

    Returns
    -------
    float
        The approximate root of the function within the specified tolerances.

    Raises
    ------
    ValueError
        If f(a) and f(b) do not have different signs, which violates the assumption of the method.

    Notes
    -----
    - The bisection method assumes that the function `f` is continuous on the interval [a, b].
    - The function values at the endpoints, f(a) and f(b), must have opposite signs (f(a) * f(b) < 0).
    - The method iteratively bisects the interval [a, b] and narrows down the interval until it finds
      a root or satisfies one of the stopping criteria:
        1. |f(c)| <= y_tol (function value tolerance).
        2. (b - a) / 2 < x_tol (interval width tolerance).
    - If the method does not converge within the maximum number of iterations, it returns the last midpoint.

    Examples
    --------
    >>> def f(x):
    ...     return x**3 - x - 2
    >>> root = bisection(f, 1, 2)
    >>> print(root)
    1.5213797092437744
    """

    f_a = f(a)
    f_b = f(b)

    if abs(f_a) <= y_tol:
        return a
    if abs(f_b) <= y_tol:
        return b

    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) <= y_tol or (b - a) / 2 < x_tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    if verbose:
        print("Bisection method did not converge after {} iterations".format(max_iter))

    return c

def find_first_decreasing_zero(x, y, get_index=False):
    """
    Find the first index where the values in `y` transition from non-negative to negative.

    Parameters
    ----------
    x : array-like
        The corresponding x-values for the y-values.
    y : array-like
        The array of y-values to evaluate.
    get_index : bool, optional
        If True, returns both the index and the corresponding x-value. Defaults to False.

    Returns
    -------
    float or tuple
        - If `get_index` is False, returns the x-value where the transition occurs.
        - If `get_index` is True, returns a tuple `(index, x_value)` for the transition point.
        - Returns `None` if no such transition is found.

    Notes
    -----
    A transition is identified when a value in `y` is non-negative (y[i] >= 0) and the
    subsequent value is negative (y[i+1] < 0).
    """
    for i in range(len(y)-1):
        if y[i] >= 0 and y[i+1] < 0:
            if get_index:
                return i, x[i]
            else:
                return x[i]

    if get_index:
        return None, None        
    return None


def find_first_decreasing_zero_bisection(x_init, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    """
    Find the first zero of a decreasing function using the bisection method.

    Parameters
    ----------
    x_init : array-like
        The initial x-values used to evaluate the function.
    f : callable
        The function for which to find the zero.
    y_tol : float, optional
        The tolerance for the absolute value of the function at the root. Defaults to 1e-6.
    x_tol : float, optional
        The tolerance for the interval width. Defaults to 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the bisection method. Defaults to 1000.
    verbose : bool, optional
        If True, prints a message if the bisection method fails to converge. Defaults to False.

    Returns
    -------
    float
        The x-value of the first zero where the function transitions from positive to negative.
        Returns NaN if no such transition is found.

    Notes
    -----
    The function `f` must be continuous and should transition from positive to negative for the
    bisection method to succeed.
    """
    x = x_init
    y = f(x)

    for i in range(len(y)-1):
        if y[i] > 0 and y[i+1] < 0:
            return bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
                
    return np.nan


def find_first_increasing_zero_bisection(x_init, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    """
    Find the first zero of an increasing function using the bisection method.

    Parameters
    ----------
    x_init : array-like
        The initial x-values used to evaluate the function.
    f : callable
        The function for which to find the zero.
    y_tol : float, optional
        The tolerance for the absolute value of the function at the root. Defaults to 1e-6.
    x_tol : float, optional
        The tolerance for the interval width. Defaults to 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the bisection method. Defaults to 1000.
    verbose : bool, optional
        If True, prints a message if the bisection method fails to converge. Defaults to False.

    Returns
    -------
    float
        The x-value of the first zero where the function transitions from negative to positive.
        Returns NaN if no such transition is found.

    Notes
    -----
    The function `f` must be continuous and should transition from negative to positive for the
    bisection method to succeed.
    """
    minus_f = lambda x: -f(x)
    return -find_first_decreasing_zero_bisection(x_init, minus_f, y_tol, x_tol, max_iter, verbose)


def find_first_decreasing_and_first_increasing_zero_bisection(x_init, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    """
    Find the first zeros where a function transitions from positive to negative (decreasing zero)
    and from negative to positive (increasing zero) using the bisection method.

    Parameters
    ----------
    x_init : array-like
        The initial x-values used to evaluate the function.
    f : callable
        The function for which to find the zeros.
    y_tol : float, optional
        The tolerance for the absolute value of the function at the root. Defaults to 1e-6.
    x_tol : float, optional
        The tolerance for the interval width. Defaults to 1e-6.
    max_iter : int, optional
        The maximum number of iterations for the bisection method. Defaults to 1000.
    verbose : bool, optional
        If True, prints a message if the bisection method fails to converge. Defaults to False.

    Returns
    -------
    tuple of floats
        A tuple `(first_decreasing_zero, first_increasing_zero)`:
        - `first_decreasing_zero`: The x-value of the first zero where the function transitions
          from positive to negative.
        - `first_increasing_zero`: The x-value of the first zero where the function transitions
          from negative to positive.
        - Both values are NaN if no corresponding transitions are found.

    Notes
    -----
    This function uses the bisection method for finding the zeros, assuming the function `f` is
    continuous and transitions between positive and negative values in the provided range.
    """
    x = x_init
    y = f(x)

    first_decreasing_zero = np.nan
    first_increasing_zero = np.nan

    for i in range(len(y)-1):
        if y[i] > 0 and y[i+1] < 0 and np.isnan(first_decreasing_zero):
            first_decreasing_zero = bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
        if y[i] < 0 and y[i+1] > 0 and np.isnan(first_increasing_zero):
            first_increasing_zero = bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
                
    return first_decreasing_zero, first_increasing_zero


def get_spiking_times(t, V, spike_high_threshold=10, spike_low_threshold=0):
    """
    Extract the spiking times from a voltage trace based on threshold crossings.

    Parameters
    ----------
    t : array-like
        Time points corresponding to the voltage trace `V`.
    V : array-like
        Voltage trace to analyze for spiking events.
    spike_high_threshold : float, optional
        The voltage threshold above which a spike is considered to start. Defaults to 10.
    spike_low_threshold : float, optional
        The voltage threshold below which a spike is considered to end. Defaults to 0.

    Returns
    -------
    tuple of (array-like, array-like)
        - `valid_starts` : The indices of `t` where spikes start (crossing above `spike_high_threshold`).
        - `spike_times` : The corresponding time points in `t` for the spike starts.

    Notes
    -----
    - This function assumes that the voltage trace is continuous and does NOT handle batch processing.
    - A spike is defined as a region where the voltage exceeds `spike_high_threshold` and eventually falls below `spike_low_threshold`.
    - Only spike starts that have a corresponding spike end are considered valid.
    - If no spikes are detected, the function returns two empty arrays.
    """
    above_threshold = V > spike_high_threshold
    below_threshold = V < spike_low_threshold

    spike_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    spike_ends = np.where(np.diff(below_threshold.astype(int)) == 1)[0] + 1
    
    # Only consider starts that have a corresponding end after them
    if len(spike_starts) == 0 or len(spike_ends) == 0:
        return np.array([]), np.array([])
    
    valid_starts = spike_starts[spike_starts < spike_ends[-1]]

    spike_times = t[valid_starts]
    
    return valid_starts, spike_times


# == Sampling utils functions ==

def latin_hyper_cube_sampling(n_samples, n_dim, ranges=None):
    """
    Perform Latin Hypercube Sampling (LHS) to generate a set of samples for a given number of dimensions.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dim : int
        The number of dimensions for the samples.
    ranges : array-like of shape (n_dim, 2), optional
        The range for each dimension, specified as [[min1, max1], [min2, max2], ...].
        If `None`, the range for all dimensions is assumed to be [0, 1]. Defaults to None.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_samples, n_dim) containing the generated samples.

    Notes
    -----
    - Latin Hypercube Sampling divides the sampling space into equal intervals and ensures that each
      interval in every dimension is sampled exactly once.
    - The function optionally scales the samples to the specified ranges for each dimension.
    - Samples are shuffled within each dimension to ensure randomness.

    Examples
    --------
    To generate 5 samples in 2 dimensions within default ranges [0, 1]:

    >>> samples = latin_hyper_cube_sampling(5, 2)

    To generate 10 samples in 3 dimensions with custom ranges:

    >>> ranges = [[0, 10], [20, 30], [100, 200]]
    >>> samples = latin_hyper_cube_sampling(10, 3, ranges)
    """

    if ranges is None:
        ranges = np.array([[0, 1] for i in range(n_dim)])

    # generate the intervals
    intervals = np.linspace(0, 1, n_samples + 1)
    
    # generate the samples
    samples = np.zeros((n_samples, n_dim))
    for i in range(n_dim):
        samples[:, i] = np.random.uniform(intervals[:-1], intervals[1:], n_samples)
        
    # shuffle the samples
    for i in range(n_dim):
        np.random.shuffle(samples[:, i])

    # scale the samples
    for i in range(n_dim):
        samples[:, i] = samples[:, i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]

    return samples