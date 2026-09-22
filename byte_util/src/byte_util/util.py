import numpy as np

from min3p.input import InputFile

__all__ = [
    'all_sites',
    'all_forcing_types',
    'battaglia_sands',
    'calc_specific_discharge',
    'get_vsflow_parameters',
    'single_to_double_float',
    'states_per_site',
    'start_date',
    'van_genuchten_perm',
    'van_genuchten_vwc',
]

all_sites = ['Cecil',
             'Flanagan',
             'HoustonBlack',
             'Kalamazoo',
             'Kuma',
             'Palouse',
             'Pullman',
             'Yolo']

all_forcing_types = ['longterm', 'monthly', 'daily', 'hourly']

states_per_site = {'Cecil': 'Northern SC',
                   'Flanagan': 'Central IL',
                   'HoustonBlack': 'Central TX',
                   'Kalamazoo': 'Central MI',
                   'Kuma': 'Eastern CO',
                   'Palouse': 'Eastern WA',
                   'Pullman': 'Northern TX',
                   'Yolo': 'Central CA'}

# According to the growing cycle specified in transpiration forcing, maize is harvested on Oct 2
# Set a universal start date for simulations, assuming we apply rock and till one month later on Nov 1
start_date = '2001-11-01'


def van_genuchten_vwc(psi, theta_r, theta_s, alpha, n):
    """Given pressure, calculate soil water content according to van Genuchten (1980)

    Parameters
    ----------
    psi : array_like
        Pressure head (L), negative for tension, positive for saturation
    theta_r : float
        Residual water content (-)
    theta_s : float
        Saturated water content (-)
    alpha : float
        van Genuchten parameter (1/L)
    n : float
        van Genuchten parameter (-)

    Returns
    -------
    theta : array_like
        Soil water content (L^3/L^3)
    """

    # Ensure psi is a numpy array
    psi = np.asarray(psi)

    m = 1.0 - 1.0/n

    # Note the np.abs(psi); in Eq 3 in van Genuchten (1980), psi is assumed to be positive
    vwc = (theta_s - theta_r) / (1.0 + (alpha * np.abs(psi)) ** n) ** m + theta_r

    # Ensure saturation when psi is positive
    vwc = np.where(psi >= 0, theta_s, vwc)

    # Ensure water content is [theta_r, theta_s] to avoid numerically
    # overshooting/undershooting
    vwc = np.clip(vwc, theta_r, theta_s)

    return vwc


def single_to_double_float(estring):
    """Reformat an exponential float string in scientific notation (1.3e-1) to
    double precision float string (1.3d-1)."""

    # Swap e with d
    dstring = estring.replace('e', 'd')
    # Also replace leading zero in exponent
    dstring = dstring.replace('-0', '-')
    dstring = dstring.replace('+0', '+')

    return dstring

def double_to_single_float(dstring):
    """Reformat an exponential float string in double precision notation (1.3d-1) to
    single precision float string (1.3e-1)."""

    # Swap e with d
    estring = dstring.replace('d', 'e')

    return estring


def van_genuchten_perm(psi, theta_r, theta_s, alpha, n, l=0.5):
    """Given the pressure head, calculate relative permeability
    according to van Genuchten (1980).

    Parameters
    ----------
    psi : array_like
        Pressure head (L)
    theta_r : float
        Residual water content (-)
    theta_s : float
        Saturated water content (-)
    alpha : float
        van Genuchten parameter (1/L)
    n : float
        van Genuchten parameter (-)
    l : float
        van Genuchten pore connectivity exponent (-)

    Returns
    -------
    theta : array_like
        Water content (-)
    """

    m = 1 - 1/n

    vwc = van_genuchten_vwc(psi, theta_r, theta_s, alpha, n)
    srel = (vwc - theta_r) / (theta_s - theta_r)

    perm = srel**l * (1.0 - (1.0 - srel**(1.0/m))**m)**2.0

    return perm


def battaglia_sands(sat, satwlim, satwfield, rew0, p1):
    """Calculate root water uptake according to Battaglia and Sands (1987)

    Parameters
    ----------
    sat : array_like
        Soil saturation (0-1)
    satwlim : float
        Saturation at wilting point (0-1)
    satwfield : float
        Saturation at field capacity (0-1)
    rew0 : float
        Fitting parameter, root extractable water at which
        rootwat=0.5 ("w_0" in Battaglia and Sands, 1997)
    p1 : float
        Fitting parameter, the slope of the linear portion
        of the curve ("a_w" in Battaglia and Sands, 1997)

    Returns
    -------
    rootwat : array_like
        Relative root water uptake (0-1)"""

    # Caculate root extractable water
    rew = (sat - satwlim)/(satwfield - satwlim)
    rew[rew > 1] = 1

    # Calculate relative root water uptake
    rootwat = rew**2 * np.exp(p1*rew)/(rew0**2 * np.exp(p1*rew0) + rew**2 * np.exp(p1*rew))

    # Ensure that everything below wilting point is zero
    rootwat[sat < satwlim] = 0

    return rootwat


def get_vsflow_parameters(input_file, elevation, return_zone=False):
    """Given a MIN3P input file and elevation, return the variably
    saturated flow parameters at that elevation.

    Parameters
    ----------
    input_file : str
        Path to the input file
    elevation : float
        Elevation at which to retrieve saturated flow parameters
    return_zone : bool
        If True, return the zone name as well"""

    params = {}

    infile = InputFile.load(input_file)
    pppm = infile.physical_parameters_porous_medium
    zi, zone_name = None, None
    for i, zone in enumerate(pppm.zones):
        bot, top = zone.extent_of_zone[-2:]
        if bot <= elevation <= top:
            zone_name = zone.name
            zi = i
            params['theta_s'] = zone.porosity

    if zi is None:
        raise ValueError('Could not determine zone associated with this elevation.')

    # Get the van genuchten parameters associated with this zone
    ppvs = infile.physical_parameters_vsflow
    params['ks'] = ppvs.zones[zi].hydraulic_conductivity_z
    shfp = ppvs.zones[zi].soil_hydraulic_function_parameters
    params['alpha'] = shfp.alpha
    params['n'] = shfp.n
    params['theta_r'] = shfp.residual_saturation * params['theta_s']

    if return_zone:
        return params, zone_name
    else:
        return params


def calc_specific_discharge(gbp, input_file, ci_a, ci_b, elev_a, elev_b,
                            gbp_cols=None):
    """Calculate the specific discharge between two grid cells using
    upstream spatial weighting.

    Parameters:
    -----------
    gbp : np.ndarray
        The gbp array of pressure values read in via read_min3p_sequence.
    input_file : str
        Path to the input file for this simulation. Used to get variably
        saturated flow parameters.
    ci_a : int
        The index of the first grid cell in the gbp array.
    ci_b : int
        The index of the second grid cell in the gbp array.
    elev_a : float
        The elevation of the first grid cell.
    elev_b : float
        The elevation of the second grid cell.
    gbp_cols : list of str, optional
        The column names of the gbp array. If not provided, use the
        default column names (['time', 'h_w', 'p_w', 's_w', 'theta_a',
        's_a', 'theta_g', 'q_root']).

    Returns:
    --------
    q : np.ndarray
        The specific discharge between two grid cells."""

    # Assume default column values if not provided
    if gbp_cols is None:
        gbp_cols = ['time', 'h_w', 'p_w', 's_w', 'theta_a', 's_a', 'theta_g', 'q_root']

    # Get h_w in each zone
    hw_a = gbp[ci_a, gbp_cols.index('h_w')]
    hw_b = gbp[ci_b, gbp_cols.index('h_w')]

    # Double check that the shallow cell head is greater than the deeper cell head
    upstream_mask = hw_a > hw_b

    # Get the vsflow parameters associated with each grid cell
    params_a = get_vsflow_parameters(input_file, elev_a)
    params_b = get_vsflow_parameters(input_file, elev_b)

    # Get hydraulic conductivity of upstream node (note that upstream spatial weighting is default for vsflow in MIN3P)
    ph_a = gbp[ci_a, gbp_cols.index('p_w')]
    ph_b = gbp[ci_b, gbp_cols.index('p_w')]
    relperm_a = van_genuchten_perm(ph_a, params_a['theta_r'], params_a['theta_s'], params_a['alpha'], params_a['n'])
    relperm_b = van_genuchten_perm(ph_b, params_b['theta_r'], params_b['theta_s'], params_b['alpha'], params_b['n'])
    k_unsat_a = relperm_a * params_a['ks']
    k_unsat_b = relperm_b * params_b['ks']

    # Calculate head gradient
    dh_dz = (hw_b - hw_a) / (elev_b - elev_a)

    # Calculate specific discharge using upstream spatial weighting
    q = np.empty_like(relperm_a)
    q[upstream_mask] = -k_unsat_a[upstream_mask] * dh_dz[upstream_mask]
    q[~upstream_mask] = -k_unsat_b[~upstream_mask] * dh_dz[~upstream_mask]

    return q

