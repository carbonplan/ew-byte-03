import numpy as np

from min3p.output import write_min3p

__all__ = [
    'bsa_to_ssa',
    'co2_factors',
    'dualperm_co2_factors',
    'mineral_params',
    'ssa_to_bsa',
    'write_initial_cec'
]

mineral_params = {'calcite-ph': {'ssa': 0.001, 'density':2.71},
                  'quartz': {'ssa': 0.001, 'density': 2.65, 'vol_frac': 0.25},
                  'k-feld-d-ph': {'ssa': 0.1, 'density': 2.57, 'vol_frac': 0.04},
                  'na-montmor': {'ssa': 5, 'density': 2.74, 'vol_frac': 0.07},
                  'sio2(a,pt)': {'ssa': 50, 'density': 2.65, 'vol_frac': 1.0e-10},
                  'gibbsite-ph': {'ssa': 50, 'density': 2.35, 'vol_frac': 1.0e-10},
                  'forst-ph': {'ssa': 0.375, 'density': 3.2, 'vol_frac': 1.0e-10},}

co2_factors = {'Cecil': 0.01,
               'Flanagan': 0.1,
               'HoustonBlack': 0.016,
               'Kalamazoo': 0.05,
               'Kuma': 0.1,
               'Palouse': 0.1,
               'Pullman': 0.04,
               'Yolo': 0.1}

dualperm_co2_factors = {'Cecil': 0.8,
                        'Flanagan': 1.0,
                        'HoustonBlack': 1.0,
                        'Kalamazoo': 1.0,
                        'Kuma': 1.0,
                        'Palouse': 1.2,
                        'Pullman': 0.8,
                        'Yolo': 1.2}

def ssa_to_bsa(ssa, density, vol_frac):
    """Convert specific surface area of a mineral (in m2/g) to bulk surface
    area (in m2/L soil)"""

    # m2/L = (m2/g) * (g/cm3) * (100 cm/m)^3 * (m3 mineral/m3 soil) * (1 m3/1000 L)
    bsa = ssa * density * 100**3 * vol_frac / 1000
    return bsa


def bsa_to_ssa(bsa, density, vol_frac):
    """Convert bulk surface area of a mineral (in m2/m3 soil) to specific
    surface area (in m2/g mineral)"""

    # m2/g = (m2/m3 soil) * (m3 soil/m3 mineral) * (1 m/100 cm)^3 * (cm3/g)
    ssa = bsa * (1/vol_frac) * (1/100)**3 * (1/density)
    return ssa


def write_initial_cec(outpath, cec, bulk_density, elev_top, elev_bot,
                      nz=401, dz=0.01, return_array=False, write_to_file=True):
    """Write 'prefix.cec' file with distributed CEC and bulk densities.

    Parameters
    ----------
    outpath : pathlib.Path
        Output file path.
    cec : iterable of float
        Cation exchange capacities (in meq/100g) for each zone
    bulk_density : iterable of float
        Bulk densities (in g/cm3) for each zone
    elev_top : iterable of float
        Top elevation height (in meters) for each zone
    elev_bot : iterable of float
        Bottom elevation height (in meters) for each zone
    nz : int, optional
        Number of zones to use (defaults to 401)
    dz : float, optional
        Grid spacing (in meters) for each zone
    return_array : bool, optional
        Whether to return the numpy array or not.
    write_to_file : bool, optional
        Whether to write the array to a file or not.

    Returns
    -------
    arr : np.ndarray
        Numpy array of shape (5, nz) with columns for X, Y, Z, CEC, and bulk density."""

    # Construct numpy array to write to file
    # X, Y, Z, CEC, bulk density
    arr = np.zeros((5, nz), dtype=float)
    arr[2] = np.linspace(0, dz*(nz-1), nz)  # Z column

    for top, bot, c, rho in zip(elev_top, elev_bot, cec, bulk_density):
        idx = np.where((arr[2] >= bot) & (arr[2] <= top))[0]
        arr[3, idx] = c
        arr[4, idx] = rho

    columns = ['x', 'y', 'z', 'cec', 'rho']

    if write_to_file:
        write_min3p(arr, outpath, columns, prefix=outpath.stem, label='initial CEC')

    if return_array:
        return arr
    else:
        return None
