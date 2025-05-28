import numpy as np
from utils import gsigmoid
from utils import get_w_factors, get_w_factors_constant_tau
from utils import d_gsigmoid
from utils import find_first_decreasing_zero_bisection
from scipy.integrate import solve_ivp
from utils import gamma_uniform_mean_std_matching

# == simulation functions == #

def simulate_individual(args):
    """
    Simulates the dynamics of a single STG neuron based on its conductance parameters.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - u0 : array-like
            Initial conditions for the state variables.
        - individual : array-like
            Conductance parameters for the neuron (e.g., maximal conductances for ion channels).
        - T_final : float
            Final time for the simulation.
        - dt : float
            Time step for evaluating the solution.
        - params : dict
            Dictionary of fixed neuron parameters (e.g., reversal potentials and calcium dynamics).

    Returns
    -------
    np.ndarray
        A 2D array where the first row contains the time points and the second row contains
        the voltage trace of the neuron during the simulation.
    """
    u0, individual, T_final, dt, params = args
    t_eval = np.arange(0, T_final, dt)
    return simulate_individual_t_eval((u0, individual, t_eval, params))


def simulate_individual_t_eval(args):
    """
    Simulates the dynamics of a single STG neuron, evaluating the solution at specific time points.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - u0 : array-like
            Initial conditions for the state variables.
        - individual : array-like
            Conductance parameters for the neuron (e.g., maximal conductances for ion channels).
        - t_eval : array-like
            Time points at which to evaluate the solution.
        - params : dict
            Dictionary of fixed neuron parameters (e.g., reversal potentials and calcium dynamics).

    Returns
    -------
    np.ndarray
        A 2D array where the first row contains the specified time points and the second row contains
        the voltage trace of the neuron during the simulation.
    """
    u0, individual, t_eval, params = args
    sol = solve_ivp(
        ODEs,
        [0, t_eval[-1]],
        u0,
        t_eval=t_eval,
        args=(
            individual[0], individual[1], individual[2], individual[3],
            individual[4], individual[5], individual[6], individual[7], 
            params['E_Na'], params['E_K'], params['E_H'], params['E_leak'], 
            params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca']
        ),
        method='BDF',
        dense_output=False,
        jac=jacobian
    )
    return np.array((sol.t, sol.y[0]))


def get_u0(V0, Ca0):
    """
    Generates the initial conditions for the state variables of the STG neuron.

    Parameters
    ----------
    V0 : float
        Initial membrane voltage.
    Ca0 : float
        Initial intracellular calcium concentration.

    Returns
    -------
    np.ndarray
        An array of initial values for the state variables, including the gating variables and calcium concentration.
    """
    u0 = np.zeros(13)

    u0[0] = V0
    u0[1] = m_inf_Na(V0)
    u0[2] = h_inf_Na(V0)
    u0[3] = m_inf_Kd(V0)
    u0[4] = m_inf_CaT(V0)
    u0[5] = h_inf_CaT(V0)
    u0[6] = m_inf_CaS(V0)
    u0[7] = h_inf_CaS(V0)
    u0[8] = m_inf_KCa(V0, Ca0)
    u0[9] = m_inf_A(V0)
    u0[10] = h_inf_A(V0)
    u0[11] = m_inf_H(V0)
    u0[12] = Ca0

    return u0


def get_default_parameters():
    """
    Provides the default neuron parameters for the STG model.

    Returns
    -------
    dict
        A dictionary containing the reversal potentials and calcium dynamics parameters.
    """
    params = {}

    params['E_leak'] = -50  # Leak reversal potential
    params['E_Na'] = 50    # Sodium reversal potential
    params['E_K'] = -80    # Potassium reversal potential
    params['E_H'] = -20    # H-current reversal potential
    params['E_Ca'] = 80    # Calcium reversal potential

    params['tau_Ca'] = 20          # Calcium decay time constant
    params['alpha_Ca'] = 0.94      
    params['beta_Ca'] = 0.05       

    return params


def get_default_u0():
    """
    Provides default initial conditions for the STG neuron state variables.
    The default values are for a resting neuron with a membrane potential of -70 mV and a calcium concentration of 0.5 µM.
    Returns
    -------
    np.ndarray
        Default initial state variables, including resting membrane potential and calcium concentration.
    """
    V0 = -70  # Resting membrane potential
    Ca0 = 0.5 # Initial calcium concentration
    u0 = get_u0(V0, Ca0)
    return u0


def get_best_set(g_s, g_u):
    """
    Determines the best set of conductances to neuromulate based on the slow and ultra-slow DIC conductance parameters.
    Derived from the reachability analysis of the STG model.

    Parameters
    ----------
    g_s : float
        Slow DIC conductance parameter.
    g_u : float
        Ultra-slow DIC conductance parameter.

    Returns
    -------
    list
        A list of strings identifying the best conductances ('A', 'H', or 'CaS').

    Notes
    -----
    - If `g_u` is negative, a warning is printed and the function returns ['CaS', 'A'].
    - If `g_s` is positive, the function returns ['A', 'H'].
    - If `g_s` is negative, the function returns ['CaS', 'H'].
    """
    if g_u < 0:
        #print('Cautious, g_u is negative!')
        return ['CaS', 'A']
    
    if g_s >= 0:
        return ['A', 'H']
    if g_s < 0:
        return ['CaS', 'H']

    
# == Gating variables functions == #

def m_inf_Na(V):
    """
    Computes the steady-state activation variable (m) for sodium (Na) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the m variable for Na channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=-5.29, D=25.5)


def h_inf_Na(V):
    """
    Computes the steady-state inactivation variable (h) for sodium (Na) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the h variable for Na channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=5.18, D=48.9)


def tau_m_Na(V):
    """
    Computes the time constant for the activation variable (m) for sodium (Na) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for Na channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=1.32, B=-1.26, C=-25, D=120)


def tau_h_Na(V):
    """
    Computes the time constant for the inactivation variable (h) for sodium (Na) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the h variable for Na channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=0.67, C=-10, D=62.9) * gsigmoid(V, A=1.5, B=1, C=3.6, D=34.9)


def m_inf_Kd(V):
    """
    Computes the steady-state activation variable (m) for delayed rectifier potassium (Kd) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the m variable for Kd channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=-11.8, D=12.3)


def tau_m_Kd(V):
    """
    Computes the time constant for the activation variable (m) for delayed rectifier potassium (Kd) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for Kd channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=7.2, B=-6.4, C=-19.2, D=28.3)


def m_inf_CaT(V):
    """
    Computes the steady-state activation variable (m) for T-type calcium (CaT) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the m variable for CaT channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=-7.2, D=27.1)


def h_inf_CaT(V):
    """
    Computes the steady-state inactivation variable (h) for T-type calcium (CaT) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the h variable for CaT channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=5.5, D=32.1)


def tau_m_CaT(V):
    """
    Computes the time constant for the activation variable (m) for T-type calcium (CaT) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for CaT channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=21.7, B=-21.3, C=-20.5, D=68.1)


def tau_h_CaT(V):
    """
    Computes the time constant for the inactivation variable (h) for T-type calcium (CaT) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the h variable for CaT channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=105, B=-89.8, C=-16.9, D=55)

def m_inf_CaS(V):
    """
    Computes the steady-state activation variable (m) for S-type calcium (CaS) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the m variable for CaS channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=-8.1, D=33)


def h_inf_CaS(V):
    """
    Computes the steady-state inactivation variable (h) for S-type calcium (CaS) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the h variable for CaS channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=6.2, D=60)


def tau_m_CaS(V):
    """
    Computes the time constant for the activation variable (m) for S-type calcium (CaS) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for CaS channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return 1.4 + 7/(np.exp((V+27)/10) + np.exp((V+70)/-13))


def tau_h_CaS(V):
    """
    Computes the time constant for the inactivation variable (h) for S-type calcium (CaS) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the h variable for CaS channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return 60 + 150/(np.exp((V+55)/9) + np.exp((V+65)/-16))


def m_inf_KCa(V, Ca):
    """
    Computes the steady-state activation variable (m) for calcium-dependent potassium (KCa) channels as a function of membrane potential and calcium concentration.

    Parameters
    ----------
    V : float
        Membrane potential.
    Ca : float
        Calcium concentration.

    Returns
    -------
    float
        The steady-state value of the m variable for KCa channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return Ca/(Ca + 3) * gsigmoid(V, A=0, B=1, C=-12.6, D=28.3)


def tau_m_KCa(V):
    """
    Computes the time constant for the activation variable (m) for calcium-dependent potassium (KCa) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for KCa channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=90.3, B=-75.1, C=-22.7, D=46)


def m_inf_A(V):
    """
    Computes the steady-state activation variable (m) for A-type potassium (A) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the m variable for A-type K channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=-8.7, D=27.2)


def h_inf_A(V):
    """
    Computes the steady-state inactivation variable (h) for A-type potassium (A) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the h variable for A-type K channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=4.9, D=56.9)


def tau_m_A(V):
    """
    Computes the time constant for the activation variable (m) for A-type potassium (A) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for A-type K channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=11.6, B=-10.4, C=-15.2, D=32.9)


def tau_h_A(V):
    """
    Computes the time constant for the inactivation variable (h) for A-type potassium (A) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the h variable for A-type K channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=38.6, B=-29.2, C=-26.5, D=38.9)


def m_inf_H(V):
    """
    Computes the steady-state activation variable (m) for H-type potassium (H) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The steady-state value of the m variable for H-type K channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=0, B=1, C=6, D=70)


def tau_m_H(V):
    """
    Computes the time constant for the activation variable (m) for H-type potassium (H) channels as a function of membrane potential.

    Parameters
    ----------
    V : float
        Membrane potential.

    Returns
    -------
    float
        The time constant of the m variable for H-type K channels.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    return gsigmoid(V, A=272, B=1499, C=-8.73, D=42.2)


def tau_Ca_constant_function(V, tau=20):
    """
    Returns a constant time constant for calcium decay.

    Parameters
    ----------
    V : float
        Membrane potential (not used in this function, but kept for consistency with other functions).
    tau : float
        The constant time constant value.

    Returns
    -------
    float
        The constant time constant.

    Reference
    ---------
    Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    """
    if np.isscalar(V):
        return tau
    return np.full_like(V, tau)

# == Derivatives == #

def d_m_inf_Na(V):
    """
    Compute the derivative of the Na activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = -5.29, D = 25.5)

def d_h_inf_Na(V):
    """
    Compute the derivative of the Na inactivation gating variable, h_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of h_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = 5.18, D = 48.9)

def d_m_inf_Kd(V):
    """
    Compute the derivative of the K delayed rectifier activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = -11.8, D = 12.3)

def d_m_inf_CaT(V):
    """
    Compute the derivative of the Ca T-type activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = -7.2, D = 27.1)

def d_h_inf_CaT(V):
    """
    Compute the derivative of the Ca T-type inactivation gating variable, h_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of h_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = 5.5, D = 32.1)

def d_m_inf_CaS(V):
    """
    Compute the derivative of the Ca S-type activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = -8.1, D = 33)

def d_h_inf_CaS(V):
    """
    Compute the derivative of the Ca S-type inactivation gating variable, h_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of h_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = 6.2, D = 60)

def d_m_inf_KCa_dV(V, Ca):  
    """
    Compute the derivative of the K Ca-dependent activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.
    Ca : float
        Intracellular calcium concentration in µM.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return Ca/(Ca + 3) * d_gsigmoid(V, A = 0, B = 1, C = -12.6, D = 28.3)

def d_m_inf_KCa_dCa(V, Ca):  
    """
    Compute the derivative of the K Ca-dependent activation gating variable, m_inf, with respect to calcium concentration (Ca).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.
    Ca : float
        Intracellular calcium concentration in µM.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to Ca.
    """
    return 3 * gsigmoid(V, A = 0, B = 1, C = -12.6, D = 28.3) / (Ca + 3)**2

def d_Ca_inf_dV(V, alpha, E_Ca, g_CaT, g_CaS, m_inf_CaT_values, m_inf_CaS_values, h_inf_CaT_values, h_inf_CaS_values, d_m_inf_CaT_values, d_m_inf_CaS_values, d_h_inf_CaT_values, d_h_inf_CaS_values): 
    """
    Compute the derivative of the intracellular calcium concentration with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.
    alpha : float
        Constant for scaling the calcium concentration change.
    E_Ca : float
        Calcium reversal potential in mV.
    g_CaT : float
        Maximal conductance for Ca T-type channels.
    g_CaS : float
        Maximal conductance for Ca S-type channels.
    m_inf_CaT_values : array
        m_inf values for the Ca T-type channel.
    m_inf_CaS_values : array
        m_inf values for the Ca S-type channel.
    h_inf_CaT_values : array
        h_inf values for the Ca T-type channel.
    h_inf_CaS_values : array
        h_inf values for the Ca S-type channel.
    d_m_inf_CaT_values : array
        Derivative of m_inf values for Ca T-type.
    d_m_inf_CaS_values : array
        Derivative of m_inf values for Ca S-type.
    d_h_inf_CaT_values : array
        Derivative of h_inf values for Ca T-type.
    d_h_inf_CaS_values : array
        Derivative of h_inf values for Ca S-type.

    Returns
    -------
    float or array
        Derivative of calcium concentration with respect to V.

    Notes
    -----
    The equation is derived from the chain rule of differentiation. The original equation can be found in the STG model ODEs.
    """
    d = np.zeros_like(V)
    d = g_CaT * 3 * m_inf_CaT_values**2 * h_inf_CaT_values * d_m_inf_CaT_values * (V - E_Ca) +\
        g_CaT * m_inf_CaT_values**3 * d_h_inf_CaT_values * (V - E_Ca) +\
        g_CaT * m_inf_CaT_values**3 * h_inf_CaT_values +\
        g_CaS * 3 * m_inf_CaS_values**2 * h_inf_CaS_values * d_m_inf_CaS_values * (V - E_Ca) +\
        g_CaS * m_inf_CaS_values**3 * d_h_inf_CaS_values * (V - E_Ca) +\
        g_CaS * m_inf_CaS_values**3 * h_inf_CaS_values
    return - alpha * d

def d_m_inf_A(V):
    """
    Compute the derivative of the A-type K activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = -8.7, D = 27.2)

def d_h_inf_A(V):
    """
    Compute the derivative of the A-type K inactivation gating variable, h_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of h_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = 4.9, D = 56.9)

def d_m_inf_H(V):
    """
    Compute the derivative of the H-type K activation gating variable, m_inf, with respect to membrane potential (V).

    Parameters
    ----------
    V : float or array
        Membrane potential in mV.

    Returns
    -------
    float or array
        Derivative of m_inf with respect to V.
    """
    return d_gsigmoid(V, A = 0, B = 1, C = 6, D = 70)

# == UTILS == #

def compute_equilibrium_Ca(alpha, I_Ca, beta):
    """
    Compute the equilibrium calcium concentration based on the calcium current (I_Ca), 
    the scaling constant (alpha), and the calcium influx rate (beta).

    The equilibrium condition is given by:
    dCa/dt = 0 = -alpha * I_Ca - Ca + beta

    Parameters
    ----------
    alpha : float
        A constant scaling factor representing the effect of the calcium current (I_Ca) on calcium concentration.
    I_Ca : float
        The calcium current (typically in µA or similar units).
    beta : float
        The calcium influx rate (or leakage rate) in the system.

    Returns
    -------
    float
        The equilibrium calcium concentration at the steady-state (Ca).
    """
    return -alpha * I_Ca + beta


def find_V_th_DICs(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
             E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
             tau_f_stg = tau_m_Na, tau_s_stg = tau_m_Kd, tau_u_stg = tau_m_H, get_I_static = False, normalize = True, y_tol = 1e-6, x_tol=1e-6, max_iter = 1000, verbose=True):
    """
    Find the threshold voltage (V_th) for dynamic input conductances (DICs).

    This function uses a bisection method to find the first voltage where the total conductance
    (g_t) decreases to zero. It also returns the values of the DICs at this threshold voltage.

    Parameters
    ----------
    V : array-like
        Array of membrane potentials (in mV) to search for the threshold voltage.
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak : float
        Maximum conductances (in mS/cm²) for the respective ion channels.
    E_Na, E_K, E_H, E_leak, E_Ca : float
        Reversal potentials (in mV) for the respective ion channels.
    alpha_Ca : float
        Scaling factor for calcium influx.
    beta_Ca : float
        Rate constant for calcium extrusion.
    tau_Ca : float
        Time constant for calcium concentration dynamics.
    tau_f_stg, tau_s_stg, tau_u_stg : callable, optional
        Functions to compute the time constants for fast, slow, and ultra-slow dynamics.
    get_I_static : bool, optional
        If True, also compute the static current.
    normalize : bool, optional
        If True, normalize the sensitivity matrix by the leak conductance.
    y_tol : float, optional
        Tolerance for the y-axis (conductance) in the bisection method.
    x_tol : float, optional
        Tolerance for the x-axis (voltage) in the bisection method.
    max_iter : int, optional
        Maximum number of iterations for the bisection method.
    verbose : bool, optional
        If True, print additional information during the bisection process.

    Returns
    -------
    V_th : float
        The threshold voltage where the total conductance decreases to zero.
    values : tuple
        A tuple containing the values of the DICs (g_f, g_s, g_u, g_t) at the threshold voltage.

    References
    ----------
    Fyon, A., Franci, A., Sacré, P., & Drion, G. (2024). Dimensionality reduction of neuronal degeneracy reveals two interfering physiological mechanisms. PNAS Nexus, 3(10), pgae415. https://doi.org/10.1093/pnasnexus/pgae415
    """
    g_t = lambda V_scalar : DICs(np.asarray([V_scalar,]), g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg, False, normalize)[3]

    V_th = find_first_decreasing_zero_bisection(V, g_t, y_tol = y_tol, x_tol=x_tol, max_iter = max_iter, verbose=verbose)
    V_th = np.asarray([V_th,], dtype=np.float64)

    if V_th is None or np.isnan(V_th):
        return V_th, (np.atleast_1d(np.nan), np.atleast_1d(np.nan), np.atleast_1d(np.nan), np.atleast_1d(np.nan))
        
    values = DICs(V_th, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg, get_I_static, normalize)
    
    return V_th, values


# == ODEs == #

def jacobian(t, u, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca):       
    """
    Compute the Jacobian matrix for the STG model.

    Parameters
    ----------
    t : float
        Time variable (not used in this function, but kept for consistency with other functions).
    u : array-like, shape (13,)
        State vector containing the following variables:
        - u[0]: V (membrane potential, mV)
        - u[1]: m_Na (activation of sodium channel)
        - u[2]: h_Na (inactivation of sodium channel)
        - u[3]: m_Kd (activation of delayed rectifier potassium channel)
        - u[4]: m_CaT (activation of T-type calcium channel)
        - u[5]: h_CaT (inactivation of T-type calcium channel)
        - u[6]: m_CaS (activation of S-type calcium channel)
        - u[7]: h_CaS (inactivation of S-type calcium channel)
        - u[8]: m_KCa (activation of calcium-activated potassium channel)
        - u[9]: m_A (activation of A-type potassium channel)
        - u[10]: h_A (inactivation of A-type potassium channel)
        - u[11]: m_H (activation of H-current channel)
        - u[12]: Ca (intracellular calcium concentration, μM)
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak : float
        Maximum conductances (mS/cm²) of the respective ion channels.
    E_Na, E_K, E_H, E_leak, E_Ca : float
        Reversal potentials (mV) of the respective ion channels.
    alpha_Ca : float
        Scaling factor for the calcium current.
    beta_Ca : float
        Calcium influx rate (or leakage rate).
    tau_Ca : float
        Time constant for calcium decay.

    Returns
    -------
    J : ndarray, shape (13, 13)
        The Jacobian matrix of the system, where J[i, j] is the partial derivative
        of the i-th state variable's time derivative with respect to the j-th state variable.
        The rows and columns of J correspond to the variables in `u`.
    """

    # Initialize Jacobian matrix (13x13 for your system)
    J = np.zeros((13, 13))
    
    # Unpack variables (for clarity)
    V = u[0]
    m_Na, h_Na = u[1], u[2]
    m_Kd = u[3]
    m_CaT, h_CaT = u[4], u[5]
    m_CaS, h_CaS = u[6], u[7]
    m_KCa = u[8]
    m_A, h_A = u[9], u[10]
    m_H = u[11]
    Ca = u[12]

    # 1. Compute partial derivatives for V equation (voltage dynamics)
    J[0, 0] = -(
        g_Na * m_Na**3 * h_Na + g_Kd * m_Kd**4 + g_CaT * m_CaT**3 * h_CaT +
        g_CaS * m_CaS**3 * h_CaS + g_KCa * m_KCa**4 + g_A * m_A**3 * h_A +
        g_H * m_H + g_leak
    )  # ∂V_dot/∂V
    
    J[0, 1] = -3 * g_Na * m_Na**2 * h_Na * (V - E_Na)  # ∂V_dot/∂m_Na
    J[0, 2] = -g_Na * m_Na**3 * (V - E_Na)  # ∂V_dot/∂h_Na
    J[0, 3] = -4 * g_Kd * m_Kd**3 * (V - E_K)  # ∂V_dot/∂m_Kd
    J[0, 4] = -3 * g_CaT * m_CaT**2 * h_CaT * (V - E_Ca)  # ∂V_dot/∂m_CaT
    J[0, 5] = -g_CaT * m_CaT**3 * (V - E_Ca)  # ∂V_dot/∂h_CaT
    J[0, 6] = -3 * g_CaS * m_CaS**2 * h_CaS * (V - E_Ca)  # ∂V_dot/∂m_CaS
    J[0, 7] = -g_CaS * m_CaS**3 * (V - E_Ca)  # ∂V_dot/∂h_CaS
    J[0, 8] = -4 * g_KCa * m_KCa**3 * (V - E_K)  # ∂V_dot/∂m_KCa
    J[0, 9] = -3 * g_A * m_A**2 * h_A * (V - E_K)  # ∂V_dot/∂m_A
    J[0, 10] = -g_A * m_A**3 * (V - E_K)  # ∂V_dot/∂h_A
    J[0, 11] = -g_H * (V - E_H)  # ∂V_dot/∂m_H

    # 2. Compute partial derivatives for calcium concentration dynamics
    J[12, 0] = -(alpha_Ca * (g_CaT * m_CaT**3 * h_CaT + g_CaS * m_CaS**3 * h_CaS)) / tau_Ca  # ∂Ca_dot/∂V
    J[12, 4] = -(alpha_Ca * 3 * g_CaT * m_CaT**2 * h_CaT * (V - E_Ca)) / tau_Ca  # ∂Ca_dot/∂m_CaT
    J[12, 5] = -(alpha_Ca * g_CaT * m_CaT**3 * (V - E_Ca)) / tau_Ca  # ∂Ca_dot/∂h_CaT
    J[12, 6] = -(alpha_Ca * 3 * g_CaS * m_CaS**2 * h_CaS * (V - E_Ca)) / tau_Ca  # ∂Ca_dot/∂m_CaS
    J[12, 7] = -(alpha_Ca * g_CaS * m_CaS**3 * (V - E_Ca)) / tau_Ca  # ∂Ca_dot/∂h_CaS
    J[12, 12] = -1 / tau_Ca  # ∂Ca_dot/∂Ca

    # 3. Partial derivatives for gating variables
    # m_Na and h_Na dynamics
    J[1, 0] = d_m_inf_Na(V) / tau_m_Na(V)  # ∂m_Na_dot/∂V
    J[1, 1] = -1 / tau_m_Na(V)  # ∂m_Na_dot/∂m_Na
    J[2, 0] = d_h_inf_Na(V) / tau_h_Na(V)  # ∂h_Na_dot/∂V
    J[2, 2] = -1 / tau_h_Na(V)  # ∂h_Na_dot/∂h_Na

    # m_Kd dynamics
    J[3, 0] = d_m_inf_Kd(V) / tau_m_Kd(V)  # ∂m_Kd_dot/∂V
    J[3, 3] = -1 / tau_m_Kd(V)  # ∂m_Kd_dot/∂m_Kd

    # m_CaT and h_CaT dynamics
    J[4, 0] = d_m_inf_CaT(V) / tau_m_CaT(V)  # ∂m_CaT_dot/∂V
    J[4, 4] = -1 / tau_m_CaT(V)  # ∂m_CaT_dot/∂m_CaT
    J[5, 0] = d_h_inf_CaT(V) / tau_h_CaT(V)  # ∂h_CaT_dot/∂V
    J[5, 5] = -1 / tau_h_CaT(V)  # ∂h_CaT_dot/∂h_CaT

    # m_CaS and h_CaS dynamics
    J[6, 0] = d_m_inf_CaS(V) / tau_m_CaS(V)  # ∂m_CaS_dot/∂V
    J[6, 6] = -1 / tau_m_CaS(V)  # ∂m_CaS_dot/∂m_CaS
    J[7, 0] = d_h_inf_CaS(V) / tau_h_CaS(V)  # ∂h_CaS_dot/∂V
    J[7, 7] = -1 / tau_h_CaS(V)  # ∂h_CaS_dot/∂h_CaS

    # m_KCa dynamics
    J[8, 0] = d_m_inf_KCa_dV(V, Ca) / tau_m_KCa(V)  # ∂m_KCa_dot/∂V
    J[8, 8] = -1 / tau_m_KCa(V)  # ∂m_KCa_dot/∂m_KCa

    # m_A and h_A dynamics
    J[9, 0] = d_m_inf_A(V) / tau_m_A(V)  # ∂m_A_dot/∂V
    J[9, 9] = -1 / tau_m_A(V)  # ∂m_A_dot/∂m_A
    J[10, 0] = d_h_inf_A(V) / tau_h_A(V)  # ∂h_A_dot/∂V
    J[10, 10] = -1 / tau_h_A(V)  # ∂h_A_dot/∂h_A

    # m_H dynamics
    J[11, 0] = d_m_inf_H(V) / tau_m_H(V)  # ∂m_H_dot/∂V
    J[11, 11] = -1 / tau_m_H(V)  # ∂m_H_dot/∂m_H

    return J

def ODEs(t, u, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca):        
    """
    Compute the time derivatives of the state variables in the STG model.

    This function implements the system of ordinary differential equations (ODEs) governing
    the dynamics of the membrane potential, gating variables, and intracellular calcium concentration
    in a neuron modeled after the stomatogastric ganglion (STG).

    Parameters
    ----------
    t : float
        Time variable (included for compatibility with ODE solvers, but not explicitly used in the equations).
    u : array-like, shape (13,)
        State vector containing the following variables in order:
        - u[0] : V (membrane potential, mV)
        - u[1] : m_Na (activation variable for sodium channel)
        - u[2] : h_Na (inactivation variable for sodium channel)
        - u[3] : m_Kd (activation variable for delayed rectifier potassium channel)
        - u[4] : m_CaT (activation variable for T-type calcium channel)
        - u[5] : h_CaT (inactivation variable for T-type calcium channel)
        - u[6] : m_CaS (activation variable for S-type calcium channel)
        - u[7] : h_CaS (inactivation variable for S-type calcium channel)
        - u[8] : m_KCa (activation variable for calcium-activated potassium channel)
        - u[9] : m_A (activation variable for A-type potassium channel)
        - u[10]: h_A (inactivation variable for A-type potassium channel)
        - u[11]: m_H (activation variable for H-current channel)
        - u[12]: Ca (intracellular calcium concentration, μM)
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak : float
        Maximum conductances (μS/nF) of the respective ion channels.
    E_Na, E_K, E_H, E_leak, E_Ca : float
        Reversal potentials (mV) for sodium, potassium, leak, and calcium currents.
    alpha_Ca : float
        Proportionality constant for calcium influx (μM·cm²·ms⁻¹).
    beta_Ca : float
        Rate constant for calcium extrusion (ms⁻¹).
    tau_Ca : float
        Time constant for calcium concentration dynamics (ms).

    Returns
    -------
    du : ndarray, shape (13,)
        Time derivatives of the state variables. The output contains:
        - du[0] : dV/dt (rate of change of membrane potential)
        - du[1] : dm_Na/dt (rate of change of sodium channel activation)
        - du[2] : dh_Na/dt (rate of change of sodium channel inactivation)
        - du[3] : dm_Kd/dt (rate of change of delayed rectifier potassium activation)
        - du[4] : dm_CaT/dt (rate of change of T-type calcium activation)
        - du[5] : dh_CaT/dt (rate of change of T-type calcium inactivation)
        - du[6] : dm_CaS/dt (rate of change of S-type calcium activation)
        - du[7] : dh_CaS/dt (rate of change of S-type calcium inactivation)
        - du[8] : dm_KCa/dt (rate of change of calcium-activated potassium activation)
        - du[9] : dm_A/dt (rate of change of A-type potassium activation)
        - du[10]: dh_A/dt (rate of change of A-type potassium inactivation)
        - du[11]: dm_H/dt (rate of change of H-current activation)
        - du[12]: dCa/dt (rate of change of intracellular calcium concentration)

    References
    ----------
    The STG model is based on the following paper: 
        Liu, Z., Golowasch, J., Marder, E., & Abbott, L. F. (1998). A model neuron with activity-dependent conductances regulated by multiple calcium sensors. The Journal of neuroscience: the official journal of the Society for Neuroscience, 18(7), 2309–2320. https://doi.org/10.1523/JNEUROSCI.18-07-02309.1998
    - But the equations are adapted and the g_bar dynamics is not considered in this implementation.
    - The capacitance should be considered in the conductances directly.
    - No external input current is considered in this implementation.
    """
    
    # Preallocate du (the vector of derivatives)
    du = np.zeros_like(u)
    # Extract variables (u is treated as a vector)
    V = u[0]
    m_Na, h_Na = u[1], u[2]
    m_Kd = u[3]
    m_CaT, h_CaT = u[4], u[5]
    m_CaS, h_CaS = u[6], u[7]
    m_KCa = u[8]
    m_A, h_A = u[9], u[10]
    m_H = u[11]
    Ca = u[12]

    # Compute currents using vectorized operations
    I_Na = g_Na * m_Na**3 * h_Na * (V - E_Na)
    I_Kd = g_Kd * m_Kd**4 * (V - E_K)
    I_CaT = g_CaT * m_CaT**3 * h_CaT * (V - E_Ca)
    I_CaS = g_CaS * m_CaS**3 * h_CaS * (V - E_Ca)
    I_KCa = g_KCa * m_KCa**4 * (V - E_K)
    I_A = g_A * m_A**3 * h_A * (V - E_K)
    I_H = g_H * m_H * (V - E_H)
    I_leak = g_leak * (V - E_leak)

    # Compute the voltage derivative
    du[0] = -(I_Na + I_Kd + I_CaT + I_CaS + I_KCa + I_A + I_H + I_leak)

    # Calcium concentration derivative
    I_Ca = I_CaT + I_CaS
    du[12] = (-alpha_Ca * I_Ca - Ca + beta_Ca) / tau_Ca

    # Gating variables (vectorized)
    du[1] = (m_inf_Na(V) - m_Na) / tau_m_Na(V)
    du[2] = (h_inf_Na(V) - h_Na) / tau_h_Na(V)
    du[3] = (m_inf_Kd(V) - m_Kd) / tau_m_Kd(V)
    du[4] = (m_inf_CaT(V) - m_CaT) / tau_m_CaT(V)
    du[5] = (h_inf_CaT(V) - h_CaT) / tau_h_CaT(V)
    du[6] = (m_inf_CaS(V) - m_CaS) / tau_m_CaS(V)
    du[7] = (h_inf_CaS(V) - h_CaS) / tau_h_CaS(V)
    du[8] = (m_inf_KCa(V, Ca) - m_KCa) / tau_m_KCa(V)
    du[9] = (m_inf_A(V) - m_A) / tau_m_A(V)
    du[10] = (h_inf_A(V) - h_A) / tau_h_A(V)
    du[11] = (m_inf_H(V) - m_H) / tau_m_H(V)

    return du

# == DICs related functions == #

def DICs(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
         E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
         tau_f_stg=tau_m_Na, tau_s_stg=tau_m_Kd, tau_u_stg=tau_m_H, get_I_static=False, normalize=True):
    """
    Computes the dynamic input conductances (DICs) for a given set of membrane potentials and conductances.

    Parameters
    ----------
    V : array-like
        Membrane potentials at which to compute the DICs. Can be a scalar or a 1D array.
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak : array-like
        Maximum conductances for the respective ion channels. Each can be a scalar or a 1D array.
    E_Na, E_K, E_H, E_leak, E_Ca : float
        Reversal potentials for the respective ion channels.
    alpha_Ca : float
        Proportionality constant for calcium influx.
    beta_Ca : float
        Rate constant for calcium extrusion.
    tau_Ca : float
        Time constant for calcium concentration dynamics.
    tau_f_stg, tau_s_stg, tau_u_stg : callable, optional
        Functions to compute the time constants for fast, slow, and ultra-slow dynamics.
    get_I_static : bool, optional
        If True, also compute the static current.
    normalize : bool, optional
        If True, normalize the sensitivity matrix by the leak conductance.

    Returns
    -------
    g_f, g_s, g_u, g_t : array-like
        The fast, slow, ultra-slow, and total conductances. Each can be a scalar, a 1D array, or a 2D array.
    I_static : array-like, optional
        The static current, returned only if `get_I_static` is True.

    Notes
    -----
    - The dimensions of the inputs can vary:
      - If all inputs are scalars, the outputs will be scalars.
      - If `V` is a 1D array and the conductances are scalars, the outputs will be 1D arrays of length N (N,).
      - If `V` is a scalar and the conductances are 1D arrays of length M, the outputs will be 1D arrays of length M (M,).
      - If both `V` and the conductances are 1D arrays of length N and M respectively, the outputs will be 2D arrays of shape (M, N).

    References
    ----------
    Drion, G., Franci, A., Dethier, J., & Sepulchre, R. (2015). Dynamic Input Conductances Shape Neuronal Spiking. eNeuro, 2(1), ENEURO.0031-14.2015. https://doi.org/10.1523/ENEURO.0031-14.2015
    """

    # get the S matrix
    if not get_I_static:
        S = sensitivity_matrix(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
                               E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
                               tau_f_stg, tau_s_stg, tau_u_stg, normalize, get_I_static)
    else:
        S, S_static = sensitivity_matrix(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
                                         E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
                                         tau_f_stg, tau_s_stg, tau_u_stg, normalize, get_I_static)

    if S.ndim == 3:
        S = S[np.newaxis, :, :, :]

    m = S.shape[0]

    g_vec = np.array([g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak]).T
    g_vec = np.atleast_2d(g_vec)
    g_vec = g_vec[:, :, np.newaxis, np.newaxis].transpose(0, 2, 1, 3)

    S_mul = np.sum(S * g_vec, axis=2)

    g_f = S_mul[:, 0, :]
    g_s = S_mul[:, 1, :]
    g_u = S_mul[:, 2, :]
    g_t = g_f + g_s + g_u

    if get_I_static:
        V_Na = V - E_Na
        V_K = V - E_K
        V_Ca = V - E_Ca
        V_H = V - E_H
        V_leak = V - E_leak

        V_vec = np.array([V_Na, V_K, V_Ca, V_Ca, V_K, V_K, V_H, V_leak])
        g_vec = g_vec[:, 0, :, 0][:, :, np.newaxis]
        
        I_static = np.sum(S_static * g_vec * V_vec, axis=1)

        if m == 1:
            I_static = I_static[0]
            g_f = g_f[0]
            g_s = g_s[0]
            g_u = g_u[0]
            g_t = g_t[0]

        return g_f, g_s, g_u, g_t, I_static
    
    if m == 1:
        g_f = g_f[0]
        g_s = g_s[0]
        g_u = g_u[0]
        g_t = g_t[0]

    return g_f, g_s, g_u, g_t

def sensitivity_matrix(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
             E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
             tau_f_stg = tau_m_Na, tau_s_stg = tau_m_Kd, tau_u_stg = tau_m_H, normalize = True, get_I_static = False):
    """
    Computes the sensitivity matrix for the dynamic input conductances (DICs) of a neuron model.

    Parameters
    ----------
    V : array-like
        Membrane potentials at which to compute the sensitivity matrix. Can be a scalar or a 1D array.
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak : array-like
        Maximum conductances for the respective ion channels. Each can be a scalar or a 1D array.
    E_Na, E_K, E_H, E_leak, E_Ca : float
        Reversal potentials for the respective ion channels.
    alpha_Ca : float
        Proportionality constant for calcium influx.
    beta_Ca : float
        Rate constant for calcium extrusion.
    tau_Ca : float
        Time constant for calcium concentration dynamics.
    tau_f_stg, tau_s_stg, tau_u_stg : callable, optional
        Functions to compute the time constants for fast, slow, and ultra-slow dynamics.
    normalize : bool, optional
        If True, normalize the sensitivity matrix by the leak conductance.
    get_I_static : bool, optional
        If True, also compute the static current.

    Returns
    -------
    S : ndarray
        The sensitivity matrix of shape (m, 3, 8, n) where n is the length of V and m is the number of conductances.
    S_static : ndarray, optional
        The static current sensitivity matrix, returned only if `get_I_static` is True.

    Notes
    -----
    - The dimensions of the inputs can vary:
        - If all inputs are scalars, the outputs will be 3D arrays of shape (3, 8, 1).
        - If `V` is a 1D array and the conductances are scalars, the outputs will be 3D arrays of shape (3, 8, N).
        - If `V` is a scalar and the conductances are 1D arrays of length M, the outputs will be 3D arrays of shape (M, 3, 8, 1).
        - If both `V` and the conductances are 1D arrays of length N and M respectively, the outputs will be 4D arrays of shape (M, 3, 8, N).

    References
    ----------
    Drion, G., Franci, A., Dethier, J., & Sepulchre, R. (2015). Dynamic Input Conductances Shape Neuronal Spiking. eNeuro, 2(1), ENEURO.0031-14.2015. https://doi.org/10.1523/ENEURO.0031-14.2015
    """
    
    V = np.atleast_1d(V)
    g_Na = np.atleast_1d(g_Na)
    g_Kd = np.atleast_1d(g_Kd)
    g_CaT = np.atleast_1d(g_CaT)
    g_CaS = np.atleast_1d(g_CaS)
    g_KCa = np.atleast_1d(g_KCa)
    g_A = np.atleast_1d(g_A)
    g_H = np.atleast_1d(g_H)
    g_leak = np.atleast_1d(g_leak)

    # m represents the number of conductances, it will be 1 for scalars, or match the size of arrays
    m = g_Na.size # we assume all conductances have the same size ... should be the case with a proper call to the function !
    n = V.size  # n is the size of V

    # S will be a 3xN matrix with N = 7+1. Each row will correspond to a different variable (f, s, u) and each column to a different channel.
    S_Na = np.zeros((m, 3, n))
    S_Kd = np.zeros((m, 3, n))
    S_CaT = np.zeros((m, 3, n))
    S_CaS = np.zeros((m, 3, n))
    S_KCa = np.zeros((m, 3, n))
    S_A = np.zeros((m, 3, n))
    S_H = np.zeros((m, 3, n))
    S_leak = np.zeros((m, 3, n))

    V = np.atleast_2d(V).T

    m_inf_Na_values = m_inf_Na(V)
    h_inf_Na_values = h_inf_Na(V)
    m_inf_Kd_values = m_inf_Kd(V)
    m_inf_CaT_values = m_inf_CaT(V)
    h_inf_CaT_values = h_inf_CaT(V)
    m_inf_CaS_values = m_inf_CaS(V)
    h_inf_CaS_values = h_inf_CaS(V)

    I_CaT = g_CaT * m_inf_CaT_values**3 * h_inf_CaT_values * (V - E_Ca)
    I_CaS = g_CaS * m_inf_CaS_values**3 * h_inf_CaS_values * (V - E_Ca)

    I_Ca = I_CaT + I_CaS
    Ca = compute_equilibrium_Ca(alpha_Ca, I_Ca, beta_Ca)
    
    m_inf_KCa_values = m_inf_KCa(V, Ca)
    m_inf_A_values = m_inf_A(V)
    h_inf_A_values = h_inf_A(V)
    m_inf_H_values = m_inf_H(V)

    d_m_inf_Na_values = d_m_inf_Na(V)
    d_h_inf_Na_values = d_h_inf_Na(V)
    d_m_inf_Kd_values = d_m_inf_Kd(V)
    d_m_inf_CaT_values = d_m_inf_CaT(V)
    d_h_inf_CaT_values = d_h_inf_CaT(V)
    d_m_inf_CaS_values = d_m_inf_CaS(V)
    d_h_inf_CaS_values = d_h_inf_CaS(V)
    d_m_inf_A_values = d_m_inf_A(V)
    d_h_inf_A_values = d_h_inf_A(V)
    d_m_inf_H_values = d_m_inf_H(V)
    d_m_inf_KCa_dCa_values = d_m_inf_KCa_dCa(V, Ca)
    d_m_inf_KCa_dV_values = d_m_inf_KCa_dV(V, Ca)
    d_Ca_inf_dV_values = d_Ca_inf_dV(V, alpha_Ca, E_Ca, g_CaT, g_CaS, m_inf_CaT_values, m_inf_CaS_values, h_inf_CaT_values, h_inf_CaS_values, d_m_inf_CaT_values, d_m_inf_CaS_values, d_h_inf_CaT_values, d_h_inf_CaS_values)

    S_Na[:, 0, :] += (m_inf_Na_values**3 * h_inf_Na_values).T
    S_Kd[:,0,:] += (m_inf_Kd_values**4).T
    S_CaT[:,0,:] += (m_inf_CaT_values**3 * h_inf_CaT_values).T
    S_CaS[:,0,:] += (m_inf_CaS_values**3 * h_inf_CaS_values).T
    S_KCa[:,0,:] += (m_inf_KCa_values**4).T
    S_A[:,0,:] += (m_inf_A_values**3 * h_inf_A_values).T
    S_H[:,0,:] += (m_inf_H_values).T
    S_leak[:,0,:] += 1.0

    if get_I_static:
        S_Na_static = S_Na[:, 0, :].copy()
        S_Kd_static = S_Kd[:, 0, :].copy()
        S_CaT_static = S_CaT[:, 0, :].copy()
        S_CaS_static = S_CaS[:, 0, :].copy()
        S_KCa_static = S_KCa[:, 0, :].copy()
        S_A_static = S_A[:, 0, :].copy()
        S_H_static = S_H[:, 0, :].copy()
        S_leak_static = S_leak[:, 0, :].copy()

    dV_dot_dm_Na = - 3 * m_inf_Na_values**2 * h_inf_Na_values * d_m_inf_Na_values * (V - E_Na)
    dV_dot_dh_Na = - m_inf_Na_values**3 * d_h_inf_Na_values * (V - E_Na)
    dV_dot_dm_Kd = - 4 * m_inf_Kd_values**3 * d_m_inf_Kd_values * (V - E_K)
    dV_dot_dm_CaT = - 3 * m_inf_CaT_values**2 * h_inf_CaT_values * d_m_inf_CaT_values * (V - E_Ca)
    dV_dot_dh_CaT = - m_inf_CaT_values**3 * d_h_inf_CaT_values * (V - E_Ca) 
    dV_dot_dm_CaS = - 3 * m_inf_CaS_values**2 * h_inf_CaS_values * d_m_inf_CaS_values * (V - E_Ca)
    dV_dot_dh_CaS = - m_inf_CaS_values**3 * d_h_inf_CaS_values * (V - E_Ca)
    dV_dot_dm_KCa = - 4 * m_inf_KCa_values**3 * d_m_inf_KCa_dV_values * (V - E_K) 
    dV_dot_dCa_KCa = - 4 * m_inf_KCa_values**3 * d_m_inf_KCa_dCa_values * (V - E_K) * d_Ca_inf_dV_values 
    dV_dot_dm_A = - 3 * m_inf_A_values**2 * h_inf_A_values * d_m_inf_A_values * (V - E_K)
    dV_dot_dh_A = - m_inf_A_values**3 * d_h_inf_A_values * (V - E_K)
    dV_dot_dm_H = - d_m_inf_H_values * (V - E_H)

    w_fs_m_Na, w_su_m_Na = get_w_factors(V, tau_m_Na, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_Na, w_su_h_Na = get_w_factors(V, tau_h_Na, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_Kd, w_su_m_Kd = get_w_factors(V, tau_m_Kd, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_CaT, w_su_m_CaT = get_w_factors(V, tau_m_CaT, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_CaT, w_su_h_CaT = get_w_factors(V, tau_h_CaT, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_CaS, w_su_m_CaS = get_w_factors(V, tau_m_CaS, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_CaS, w_su_h_CaS = get_w_factors(V, tau_h_CaS, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_KCa, w_su_m_KCa = get_w_factors(V, tau_m_KCa, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_KCa2, w_su_m_KCa2 = get_w_factors_constant_tau(V, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg) # TO BE CHANGED
    w_fs_m_A, w_su_m_A = get_w_factors(V, tau_m_A, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_A, w_su_h_A = get_w_factors(V, tau_h_A, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_H, w_su_m_H = get_w_factors(V, tau_m_H, tau_f_stg, tau_s_stg, tau_u_stg)

    S_Na[:,0,:] += (- w_fs_m_Na * dV_dot_dm_Na - w_fs_h_Na * dV_dot_dh_Na).T
    S_Kd[:,0,:] += (- w_fs_m_Kd * dV_dot_dm_Kd).T
    S_CaT[:,0,:] += (- w_fs_m_CaT * dV_dot_dm_CaT - w_fs_h_CaT * dV_dot_dh_CaT).T
    S_CaS[:,0,:] += (- w_fs_m_CaS * dV_dot_dm_CaS - w_fs_h_CaS * dV_dot_dh_CaS).T
    S_KCa[:,0,:] += (- w_fs_m_KCa * dV_dot_dm_KCa - w_fs_m_KCa2 * dV_dot_dCa_KCa).T
    S_A[:,0,:] += (- w_fs_m_A * dV_dot_dm_A - w_fs_h_A * dV_dot_dh_A).T
    S_H[:,0,:] += (- w_fs_m_H * dV_dot_dm_H).T

    S_Na[:,1,:] += (- (w_su_m_Na - w_fs_m_Na) * dV_dot_dm_Na - (w_su_h_Na - w_fs_h_Na) * dV_dot_dh_Na).T
    S_Kd[:,1,:] += (- (w_su_m_Kd - w_fs_m_Kd) * dV_dot_dm_Kd).T
    S_CaT[:,1,:] += (- (w_su_m_CaT - w_fs_m_CaT) * dV_dot_dm_CaT - (w_su_h_CaT - w_fs_h_CaT) * dV_dot_dh_CaT).T
    S_CaS[:,1,:] += (- (w_su_m_CaS - w_fs_m_CaS) * dV_dot_dm_CaS - (w_su_h_CaS - w_fs_h_CaS) * dV_dot_dh_CaS).T
    S_KCa[:,1,:] += (- (w_su_m_KCa - w_fs_m_KCa) * dV_dot_dm_KCa - (w_su_m_KCa2 - w_fs_m_KCa2) * dV_dot_dCa_KCa).T
    S_A[:,1,:] += (- (w_su_m_A - w_fs_m_A) * dV_dot_dm_A - (w_su_h_A - w_fs_h_A) * dV_dot_dh_A).T
    S_H[:,1,:] += (- (w_su_m_H - w_fs_m_H) * dV_dot_dm_H).T

    S_Na[:,2,:] += (- (1 - w_su_m_Na) * dV_dot_dm_Na - (1 - w_su_h_Na) * dV_dot_dh_Na).T
    S_Kd[:,2,:] += (- (1 - w_su_m_Kd) * dV_dot_dm_Kd).T
    S_CaT[:,2,:] += (- (1 - w_su_m_CaT) * dV_dot_dm_CaT - (1 - w_su_h_CaT) * dV_dot_dh_CaT).T
    S_CaS[:,2,:] += (- (1 - w_su_m_CaS) * dV_dot_dm_CaS - (1 - w_su_h_CaS) * dV_dot_dh_CaS).T
    S_KCa[:,2,:] += (- (1 - w_su_m_KCa) * dV_dot_dm_KCa - (1 - w_su_m_KCa2) * dV_dot_dCa_KCa).T
    S_A[:,2,:] += (- (1 - w_su_m_A) * dV_dot_dm_A - (1 - w_su_h_A) * dV_dot_dh_A).T
    S_H[:,2,:] += (- (1 - w_su_m_H) * dV_dot_dm_H).T

    if normalize:
        S_Na /= g_leak[:, np.newaxis, np.newaxis]
        S_Kd /= g_leak[:, np.newaxis, np.newaxis]
        S_CaT /= g_leak[:, np.newaxis, np.newaxis]
        S_CaS /= g_leak[:, np.newaxis, np.newaxis]
        S_KCa /= g_leak[:, np.newaxis, np.newaxis]
        S_A /= g_leak[:, np.newaxis, np.newaxis]
        S_H /= g_leak[:, np.newaxis, np.newaxis]
        S_leak /= g_leak[:, np.newaxis, np.newaxis]

    S_Na = S_Na[:, :, np.newaxis, :]
    S_Kd = S_Kd[:, :, np.newaxis, :]
    S_CaT = S_CaT[:, :, np.newaxis, :]
    S_CaS = S_CaS[:, :, np.newaxis, :]
    S_KCa = S_KCa[:, :, np.newaxis, :]
    S_A = S_A[:, :, np.newaxis, :]
    S_H = S_H[:, :, np.newaxis, :]
    S_leak = S_leak[:, :, np.newaxis, :]

    if not get_I_static:
        if m == 1:
            return np.concatenate((S_Na[0], S_Kd[0], S_CaT[0], S_CaS[0], S_KCa[0], S_A[0], S_H[0], S_leak[0]), axis=1)
        else:
            return np.concatenate((S_Na, S_Kd, S_CaT, S_CaS, S_KCa, S_A, S_H, S_leak), axis=2)
    else:
        if m == 1:
            return np.concatenate((S_Na[0], S_Kd[0], S_CaT[0], S_CaS[0], S_KCa[0], S_A[0], S_H[0], S_leak[0]), axis=1), np.stack((S_Na_static, S_Kd_static, S_CaT_static, S_CaS_static, S_KCa_static, S_A_static, S_H_static, S_leak_static), axis=1)
        else:
            return np.concatenate((S_Na, S_Kd, S_CaT, S_CaS, S_KCa, S_A, S_H, S_leak), axis=2), np.stack((S_Na_static, S_Kd_static, S_CaT_static, S_CaS_static, S_KCa_static, S_A_static, S_H_static, S_leak_static), axis=1)

# == Compensation algorithms and generation functions == #

def generate_population(n_cells, V_th,  g_f_target, g_s_target, g_u_target, g_bar_range_leak, g_bar_range_Na, g_bar_range_Kd, g_bar_range_CaT, g_bar_range_CaS, g_bar_range_KCa, g_bar_range_A, g_bar_range_H, params, default_g_CaS_for_Ca = 10., default_g_CaT_for_Ca = 6.0, distribution='uniform', normalize_by_leak=True):
    """
    Generates a population of neurons with specified dynamic input conductances (DICs).

    Parameters
    ----------
    n_cells : int
        Number of neurons to generate.
    V_th : float
        Threshold voltage for dynamic input conductances (DICs).
    g_f_target : float
        Target fast DIC.
    g_s_target : float
        Target slow DIC.
    g_u_target : float
        Target ultra-slow DIC.
    g_bar_range_leak : list
        Range of leak conductances.
    g_bar_range_Na : list
        Range of sodium conductances.
    g_bar_range_Kd : list
        Range of delayed rectifier potassium conductances.
    g_bar_range_CaT : list
        Range of T-type calcium conductances.
    g_bar_range_CaS : list
        Range of S-type calcium conductances.
    g_bar_range_KCa : list
        Range of calcium-activated potassium conductances.
    g_bar_range_A : list
        Range of A-type potassium conductances.
    g_bar_range_H : list
        Range of H-current conductances.
    params : dict
        Dictionary of fixed neuron parameters (e.g., reversal potentials and calcium dynamics).
    default_g_CaS_for_Ca : float, optional
        Default S-type calcium conductance for calcium dynamics.
    default_g_CaT_for_Ca : float, optional
        Default T-type calcium conductance for calcium dynamics.
    distribution : str, optional
        Distribution type for generating conductances ('uniform' or 'gamma').
    normalize_by_leak : bool, optional
        If True, normalize the conductances by the leak conductance.

    Returns
    -------
    np.ndarray
        Array of generated neuron conductances.

    Notes
    -----
    -   The population generation from this method can fail. In practice, the method using neuromodulation introduced by A. Fyon et al. (2015) and improved by this work can be used to generate a population of neurons with specified DICs.
    -   The method using neuromodulation is implemented in the `generate_neuromodulated_population` function and using the `get_best_set` function ensure reachability from a spiking population.
    -   The method is based on the work of Drion et al. (2015) and the compensation algorithm for DICs.

    -   If g_CaS and g_CaT are among the conductances to be compensated, the system is non-linear and providing a default value for the conductances is necessary. The compensation is, in this case, approximate and the reached DICs may not be exactly the target DICs.
    References
    ----------
    For the compensation algorithm:
    -   Drion, G., Franci, A., Dethier, J., & Sepulchre, R. (2015). Dynamic Input Conductances Shape Neuronal Spiking. eNeuro, 2(1), ENEURO.0031-14.2015. https://doi.org/10.1523/ENEURO.0031-14.2015
    For the population generation:
    -   Fyon, A., Franci, A., Sacré, P., & Drion, G. (2024). Dimensionality reduction of neuronal degeneracy reveals two interfering physiological mechanisms. PNAS Nexus, 3(10), pgae415. https://doi.org/10.1093/pnasnexus/pgae415
    """
    g_Na = np.random.uniform(g_bar_range_Na[0], g_bar_range_Na[1], n_cells) if g_bar_range_Na is not None else np.full(n_cells, np.nan)
    g_Kd = np.random.uniform(g_bar_range_Kd[0], g_bar_range_Kd[1], n_cells) if g_bar_range_Kd is not None else np.full(n_cells, np.nan)
    g_CaT = np.random.uniform(g_bar_range_CaT[0], g_bar_range_CaT[1], n_cells) if g_bar_range_CaT is not None else np.full(n_cells, np.nan)
    g_CaS = np.random.uniform(g_bar_range_CaS[0], g_bar_range_CaS[1], n_cells) if g_bar_range_CaS is not None else np.full(n_cells, np.nan)
    g_KCa = np.random.uniform(g_bar_range_KCa[0], g_bar_range_KCa[1], n_cells) if g_bar_range_KCa is not None else np.full(n_cells, np.nan)
    g_A = np.random.uniform(g_bar_range_A[0], g_bar_range_A[1], n_cells) if g_bar_range_A is not None else np.full(n_cells, np.nan)
    g_H = np.random.uniform(g_bar_range_H[0], g_bar_range_H[1], n_cells) if g_bar_range_H is not None else np.full(n_cells, np.nan)
    
    if distribution == 'uniform':
        # here we assume that the ranges are the min and max values for the uniform distribution
        mean_leak = (g_bar_range_leak[0] + g_bar_range_leak[1]) / 2
        g_leak = np.random.uniform(g_bar_range_leak[0], g_bar_range_leak[1], n_cells)
        
    elif distribution == 'gamma':
        g_bar_range_leak = gamma_uniform_mean_std_matching(*g_bar_range_leak)
        mean_leak = g_bar_range_leak[0] * g_bar_range_leak[1]
        g_leak = np.random.gamma(g_bar_range_leak[0], g_bar_range_leak[1], n_cells)
    else:
        raise ValueError('Invalid distribution type ! Please use either "uniform" or "gamma".')

    if normalize_by_leak:
        f = g_leak/mean_leak
        g_Na *= f
        g_Kd *= f
        g_CaT *= f
        g_CaS *= f
        g_KCa *= f
        g_A *= f
        g_H *= f

    x = general_compensation_algorithm(V_th, [g_f_target, g_s_target, g_u_target], g_leak, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, params['E_Na'], params['E_K'], params['E_H'], params['E_leak'], params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca'], default_g_CaS_for_Ca=default_g_CaS_for_Ca, default_g_CaT_for_Ca=default_g_CaT_for_Ca)
    
    return x

def modulate_population(population, V_th, g_f_target, g_s_target, g_u_target, params, set_to_compensate, default_g_CaS_for_Ca = 10., default_g_CaT_for_Ca = 6.0, iterations=0):
    """
    Modulates a population of neurons to achieve specified dynamic input conductances (DICs).

    Parameters
    ----------
    population : array-like
        Array of neuron conductances to be modulated.
    V_th : float
        Threshold voltage for dynamic input conductances (DICs).
    g_f_target : float or None
        Target fast DIC. If None, it will be compensated.
    g_s_target : float or None
        Target slow DIC. If None, it will be compensated.
    g_u_target : float or None
        Target ultra-slow DIC. If None, it will be compensated.
    params : dict
        Dictionary of fixed neuron parameters (e.g., reversal potentials and calcium dynamics).
    set_to_compensate : list
        List of conductances to be compensated (e.g., ['Na', 'Kd', 'CaT']).
    default_g_CaS_for_Ca : float or array-like, optional
        Default S-type calcium conductance for calcium dynamics.
    default_g_CaT_for_Ca : float or array-like, optional
        Default T-type calcium conductance for calcium dynamics.
    iterations : int, optional
        Number of iterations for the compensation algorithm. Useless if g_CaS and g_CaT are not among the conductances to compensate.

    Returns
    -------
    np.ndarray
        Array of modulated neuron conductances.

    Notes
    -----
    - The function uses a compensation algorithm to adjust the conductances of the neurons to achieve the specified DICs.
    - The number of conductances to compensate should be equal to the number of target DICs that are None or NaN.

    - If g_CaS and g_CaT are among the conductances to be compensated, the system is non-linear and providing a default value for the conductances is necessary. The compensation is, in this case, approximate and the reached DICs may not be exactly the target DICs. Iterations can be used to refine the compensation.

    References
    ----------
    Drion, G., Franci, A., Dethier, J., & Sepulchre, R. (2015). Dynamic Input Conductances Shape Neuronal Spiking. eNeuro, 2(1), ENEURO.0031-14.2015. https://doi.org/10.1523/ENEURO.0031-14.2015
    """
    population = np.asarray(population)
    
    number_none_dics = 0
    if g_f_target is None or np.isnan(g_f_target):
        number_none_dics += 1
    if g_s_target is None or np.isnan(g_s_target):
        number_none_dics += 1
    if g_u_target is None or np.isnan(g_u_target):
        number_none_dics += 1

    if 3 - number_none_dics != len(set_to_compensate):
        raise ValueError('Number of conductances to compensate should be equal to the number of target DICS')

    while iterations >= 0:
        if 'Na' in set_to_compensate:
            population[:, 0] = np.nan
        if 'Kd' in set_to_compensate:
            population[:, 1] = np.nan
        if 'CaT' in set_to_compensate:
            population[:, 2] = np.nan
        if 'CaS' in set_to_compensate:
            population[:, 3] = np.nan
        if 'KCa' in set_to_compensate:
            population[:, 4] = np.nan
        if 'A' in set_to_compensate:
            population[:, 5] = np.nan
        if 'H' in set_to_compensate:
            population[:, 6] = np.nan

        if 'CaS' not in set_to_compensate and 'CaT' not in set_to_compensate:
            iterations = 0

        population = general_compensation_algorithm(V_th, [g_f_target, g_s_target, g_u_target], population[:, 7], population[:, 0], population[:, 1], population[:, 2], population[:, 3], population[:, 4], population[:, 5], population[:, 6], params['E_Na'], params['E_K'], params['E_H'], params['E_leak'], params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca'], default_g_CaS_for_Ca=default_g_CaS_for_Ca, default_g_CaT_for_Ca=default_g_CaT_for_Ca)
        iterations -= 1
        default_g_CaT_for_Ca = population[:, 2].copy()
        default_g_CaS_for_Ca = population[:, 3].copy()

    return population

def general_compensation_algorithm(V_th, target_DICs, new_g_leak, new_g_Na, new_g_Kd, new_g_CaT, new_g_CaS, new_g_KCa, new_g_A, new_g_H, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg = tau_m_Na, tau_s_stg = tau_m_Kd, tau_u_stg = tau_m_H, default_g_CaS_for_Ca = 10., default_g_CaT_for_Ca = 6.0):
    """
    General compensation algorithm to adjust conductances to achieve target dynamic input conductances (DICs).

    Parameters
    ----------
    V_th : float or array-like
        Threshold voltage for dynamic input conductances (DICs).
    target_DICs : list of float
        List containing the target fast, slow, and ultra-slow DICs. Use None or NaN for the DICs to be compensated.
    new_g_leak, new_g_Na, new_g_Kd, new_g_CaT, new_g_CaS, new_g_KCa, new_g_A, new_g_H : array-like or float
        Initial conductances for the respective ion channels. Use None or NaN for the conductances to be compensated.
    E_Na, E_K, E_H, E_leak, E_Ca : float
        Reversal potentials for the respective ion channels.
    alpha_Ca : float
        Proportionality constant for calcium influx.
    beta_Ca : float
        Rate constant for calcium extrusion.
    tau_Ca : float
        Time constant for calcium concentration dynamics.
    tau_f_stg, tau_s_stg, tau_u_stg : callable, optional
        Functions to compute the time constants for fast, slow, and ultra-slow dynamics.
    default_g_CaS_for_Ca : float or array-like, optional
        Default S-type calcium conductance for calcium dynamics.
    default_g_CaT_for_Ca : float or array-like, optional
        Default T-type calcium conductance for calcium dynamics.

    Returns
    -------
    np.ndarray
        Array of compensated conductances.

    Notes
    -----
    - The number of conductances to compensate should be equal to the number of target DICs that are None or NaN.
    - If g_CaS and g_CaT are among the conductances to be compensated, the system is non-linear and providing a default value for the conductances is necessary. The compensation is, in this case, approximate and the reached DICs may not be exactly the target DICs. Iterative compensation can be used to refine the compensation.

    References
    ----------
    Drion, G., Franci, A., Dethier, J., & Sepulchre, R. (2015). Dynamic Input Conductances Shape Neuronal Spiking. eNeuro, 2(1), ENEURO.0031-14.2015. https://doi.org/10.1523/ENEURO.0031-14.2015
    """
    none_index = []

    new_g_dict = {
        'Na': new_g_Na,
        'Kd': new_g_Kd,
        'CaT': new_g_CaT,
        'CaS': new_g_CaS,
        'KCa': new_g_KCa,
        'A': new_g_A,
        'H': new_g_H,
        'leak': new_g_leak
    }

    # Handle None and NaN values
    for idx, (key, g_value) in enumerate(new_g_dict.items()):
        g_value = np.atleast_1d(g_value)
        if g_value is None or np.isnan(g_value).all():
            none_index.append(idx)
            new_g_dict[key] = np.zeros_like(g_value) if key not in ['CaT', 'CaS'] else np.full_like(g_value, default_g_CaT_for_Ca if key == 'CaT' else default_g_CaS_for_Ca)

    if len(none_index) == 0 or len(none_index) > 3:
        raise ValueError('Number of conductances to compensate should be between 1 and 3')

    target_DICs = np.array(target_DICs, dtype=np.float64)
    not_none_index_target = np.where(~np.isnan(target_DICs))[0]
    target_DICs = target_DICs[not_none_index_target]

    # verify if the number of none index is equal to the number of none in target_DICs
    if len(none_index) != len(not_none_index_target):
        raise ValueError('Number of None in target_DICs should be equal to the number of None in the conductances')

    new_gs = np.asarray([new_g_dict[key] for key in new_g_dict.keys()])
    copy_new_gs = new_gs.copy().T

    if not isinstance(V_th, np.ndarray):
        V_th = np.array([V_th,])

    S_full = sensitivity_matrix(V_th, *new_gs, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg)

    S_full = S_full.squeeze()
    if S_full.ndim == 2:
        S_full = S_full[np.newaxis, :, :]

    not_none_index = [i for i in range(len(new_g_dict)) if i not in none_index]

    S_random = S_full[:, not_none_index_target, :][:, :, not_none_index]
    S_compensated = S_full[:, not_none_index_target, :][:, :, none_index]

    new_gs = new_gs[not_none_index]

    new_gs = new_gs.T[:, np.newaxis, :]
    A = S_compensated
    result_dot_product = np.sum(S_random * new_gs, axis=2)
    b = target_DICs[np.newaxis, :] - result_dot_product

    # add a dimension to b - I have to introduce this because of the new version of numpy ...
    b = b[:, :, np.newaxis]

    try:
        #print the condition number of the matrix
        #print(np.linalg.cond(A))
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.full((1, len(none_index)), -np.inf)

    x = x.squeeze()
    
    # refill new_gs with the new values
    copy_new_gs[:, none_index] = x
    return copy_new_gs

def generate_spiking_population(n_cells, V_th=-51.0):
    """
    Generates a population of spiking neurons (corresponding to g_f = -6.2, g_s = 4.0, g_u = 5.0).

    Parameters
    ----------
    n_cells : int
        Number of neurons to generate.
    V_th : float, optional
        Threshold voltage for dynamic input conductances (DICs). Default is -51.0 mV.

    Returns
    -------
    np.ndarray
        Array of generated spiking neuron conductances, shape (n_cells, 8).

    Notes
    -----
    - The population is generated with the following conductance ranges:
        - Leak: [0.007, 0.014]
        - (Na: [2500, 4500]) [compensated]
        - Kd: [70, 140]
        - CaT: [3, 7]
        - CaS: [6, 22]
        - KCa: [140, 180]
        - (A: [200, 400]) [compensated]
        - (H: [0.25, 0.5]) [compensated]
    - Fyon et al. (2024) used a uniform distribution to generate the conductances. We use a gamma distribution to generate the conductances in this work. Results are similar.

    References
    ----------
    Fyon, A., Franci, A., Sacré, P., & Drion, G. (2024). Dimensionality reduction of neuronal degeneracy reveals two interfering physiological mechanisms. PNAS Nexus, 3(10), pgae415. https://doi.org/10.1093/pnasnexus/pgae415
    """

    # FROM Fyon et al. 2024
    g_bar_range_Kd2 = [70, 140]
    g_bar_range_CaT2 = [3, 7]
    g_bar_range_CaS2 = [6, 22]
    g_bar_range_KCa2 = [140, 180]
    g_bar_range_leak2 = [0.007, 0.014]

    # g_Na is compensated
    g_bar_range_Kd = [g_bar_range_Kd2[0], g_bar_range_Kd2[1]]
    g_bar_range_CaT = [g_bar_range_CaT2[0], g_bar_range_CaT2[1]]
    g_bar_range_CaS = [g_bar_range_CaS2[0], g_bar_range_CaS2[1]]
    g_bar_range_KCa = [g_bar_range_KCa2[0], g_bar_range_KCa2[1]]
    # g_A is compensated
    # g_H is compensated
    g_bar_range_leak = [g_bar_range_leak2[0], g_bar_range_leak2[1]]

    g_s_spiking = 4.
    g_u_spiking = 5.
    g_f_spiking = -g_s_spiking - 2.2

    PARAMS = get_default_parameters()
    spiking_population = generate_population(n_cells, V_th, g_f_spiking, g_s_spiking, g_u_spiking, g_bar_range_leak, None, g_bar_range_Kd, g_bar_range_CaT, g_bar_range_CaS, g_bar_range_KCa, None, None, params=PARAMS, distribution="gamma")

    return spiking_population

def generate_neuromodulated_population(n_cells, V_th_target, g_s_target, g_u_target, set_to_compensate=None, clean=True, use_fitted_gCaS = lambda g_s, g_u : 34.12021074772369 -2.3296612301271464*g_s, use_fitted_gCaT = lambda g_s, g_u : 24.6 - 5.14 * g_s, iterations=5, d_gCaS=10., d_gCaT=6.0):
    """
    Generates a population of neurons with specified dynamic input conductances (DICs) using neuromodulation from a spiking population.

    Parameters
    ----------
    n_cells : int
        Number of neurons to generate.
    V_th_target : float
        Target threshold voltage for dynamic input conductances (DICs).
    g_s_target : float
        Target slow DIC.
    g_u_target : float
        Target ultra-slow DIC.
    set_to_compensate : list, optional
        List of conductances to be compensated (e.g., ['A', 'H']). If None, the best set will be determined automatically from the target DICs and the reachability from a spiking population.
    clean : bool, optional
        If True, remove any neuron with negative conductances.
    use_fitted_gCaS : callable, optional
        Function to determine the default S-type calcium conductance for calcium dynamics based on the target DICs. 
        The default function is a linear fit performed in this work and is associated with the best set of conductances to compensate.
    use_fitted_gCaT : callable, optional
        Function to determine the default T-type calcium conductance for calcium dynamics based on the target DICs.
        The default function is a linear fit performed in this work and is associated with the best set of conductances to compensate.
    iterations : int, optional
        Number of iterations for the compensation algorithm. Useless if g_CaS and g_CaT are not among the conductances to compensate.

    Returns
    -------
    np.ndarray
        Array of generated neuromodulated neuron conductances.

    Notes
    -----
    - The function first generates a population of spiking neurons and then modulates them to achieve the specified DICs.
    - The number of conductances to compensate should be equal to the number of target DICs that are None or NaN.
    - If g_CaS and g_CaT are among the conductances to be compensated, the system is non-linear and providing a default value for the conductances is necessary. The compensation is, in this case, approximate and the reached DICs may not be exactly the target DICs. Iterative compensation can be used to refine the compensation.
    - In practice, the method using neuromodulation introduced by A. Fyon et al. (2015) and improved by this work can be used to generate a population of neurons with specified DICs without the failure of the direct population generation method.

    References
    ----------
    - Fyon, A., Franci, A., Sacré, P., & Drion, G. (2024). Dimensionality reduction of neuronal degeneracy reveals two interfering physiological mechanisms. PNAS Nexus, 3(10), pgae415. https://doi.org/10.1093/pnasnexus/pgae415
    """
    g_f_target = None
    spiking_population = generate_spiking_population(n_cells, V_th_target)

    if set_to_compensate is None:
        set_to_compensate = get_best_set(g_s_target, g_u_target)

    if use_fitted_gCaS:
        d_gCaS = use_fitted_gCaS(g_s_target, g_u_target)

    if use_fitted_gCaT:
        d_gCaT = use_fitted_gCaT(g_s_target, g_u_target)

    neuromodulated_population = modulate_population(spiking_population, V_th_target, g_f_target, g_s_target, g_u_target, get_default_parameters(), set_to_compensate, default_g_CaS_for_Ca=d_gCaS, default_g_CaT_for_Ca=d_gCaT, iterations=iterations)

    if clean:
        # remove any neuron with < 0 conductances
        neuromodulated_population = neuromodulated_population[np.all(neuromodulated_population >= 0, axis=1)]

    return neuromodulated_population