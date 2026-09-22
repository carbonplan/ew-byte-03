"""Feedstock application and re-application for chained MIN3P simulations.

A MIN3P run carries feedstock in two depth-resolved arrays: `*.min` for mineral 
volume fractions and `*.surf` for bulk surface areas (m2/L soil). Later applications
build on a partially dissolved profile, so we have to reconstruct the fate of the 
surface area from the feedstock that was already present.

We don't seem to have a surface area output, but we can derive it with the selected
surface area scaling relationship from the volume fraction. See `advance_surface_area`.

Re-application rock surface area comes from the "aged" rock via the selected 
relationship, and the fresh rock with a prescribed SA. The bulk surface areas add; 
the resulting specific surface area is then the mass-weighted mean of the two.

Everything here is numpy/pandas plus `min3p`. Functions should work for Coiled and 
HPC.

Note on column lists
--------------------
``*.min`` carries a trailing ``porosity`` column that ``*.surf`` does not, so the 
two files have different variable lists. Functions here look minerals
up by name in each array's own column list; pass ``bsa_columns`` whenever the
surface-area array does not share the volume-fraction array's columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from min3p.output import read_min3p, read_min3p_sequence, write_min3p

__all__ = [
    'DEFAULT_CONST_RF',
    'DEFAULT_MIX_RULE',
    'DEFAULT_ROUGHNESS_OPTION',
    'MIX_RULES',
    'PHI_FLOOR',
    'SA_EXPONENT',
    'advance_surface_area',
    'apply_feedstock',
    'depth_profile',
    'feedstock_properties',
    'feedstock_roughness',
    'implied_ssa',
    'load_feedstock',
    'load_schedule',
    'props_for_schedule',
    'read_mineral_database',
    'till_layer_mask',
    'till_soil',
    'write_restart_minerals',
]

# Minimum volume fraction for a mineral that is present but depleted
PHI_FLOOR = 1.0e-10

# Surface-area scaling exponent for the 'twothird' and 'twothird-mix' update types.
SA_EXPONENT = 2.0 / 3.0

DEFAULT_ROUGHNESS_OPTION = 'constant'
DEFAULT_CONST_RF = 10.0
DEFAULT_MIX_RULE = 'homogeneous'


# ---------------------------------------------------------------------------
# Mineral properties and feedstock definition
# ---------------------------------------------------------------------------

def read_mineral_database(dbs_path)->dict:
    """Parse molar mass and density for every mineral in a MIN3P ``mineral.dbs``.

    Each entry is a quoted mineral name, a surface-area-type line, then a line
    holding molar mass (g/mol) and density (g/cm3). Molar volume follows as
    ``molar_mass / density``. 

    Parameters
    ----------
    dbs_path : str or pathlib.Path
        Path to the MIN3P mineral database file.

    Returns
    -------
    dict
        Maps mineral name to a dict with ``molar_mass_g_mol``, ``density_g_cm3``
        and ``molar_volume_cm3_mol``.
    """
    lines = Path(dbs_path).read_text(errors='replace').splitlines()

    minerals = {}
    for i, line in enumerate(lines):
        name = line.strip()
        if not (name.startswith("'") and name.endswith("'") and len(name) > 2):
            continue
        if i + 2 >= len(lines) or lines[i + 1].strip() != "'surface'":
            continue
        parts = lines[i + 2].split()
        try:
            molar_mass, density = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        minerals[name.strip("'")] = {
            'molar_mass_g_mol': molar_mass,
            'density_g_cm3': density,
            'molar_volume_cm3_mol': molar_mass / density,
        }

    if not minerals:
        raise ValueError(f'No mineral entries parsed from {dbs_path}')

    return minerals


def load_feedstock(path, mineral_db, default_diameter_um, density_g_cm3=None):
    """Load a feedstock definition file and join it to mineral database properties.

    Mineral names must match the MIN3P database so their properties can be looked
    up, and must also be declared in the ``'minerals'`` block of the simulation's
    ``.dat`` deck -- a mineral with no column in ``.min``/``.surf`` has nowhere to
    receive volume fraction.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with columns ``mineral`` and ``weight_percent``, optionally
        ``diameter_um``.
    mineral_db : dict
        Output of `read_mineral_database`.
    default_diameter_um : float
        Grain diameter used for any mineral without its own ``diameter_um``.
    density_g_cm3 : float, optional
        Bulk feedstock density. When omitted it is derived from the composition
        as the mass-weighted harmonic mean of the components. 

    Returns
    -------
    feedstock : pandas.DataFrame
        One row per mineral, with weight fraction and molar properties.
    density_g_cm3 : float
        Bulk density of the feedstock (g/cm3).
    """
    feedstock = pd.read_csv(path, comment='#')

    missing_cols = {'mineral', 'weight_percent'} - set(feedstock.columns)
    if missing_cols:
        raise ValueError(f'{path} is missing required column(s): {sorted(missing_cols)}')
    if feedstock.empty:
        raise ValueError(f'{path} defines no minerals')

    unknown = sorted(set(feedstock['mineral']) - set(mineral_db))
    if unknown:
        raise KeyError(f'minerals not found in the MIN3P database: {unknown}')

    total = feedstock['weight_percent'].sum()
    if total <= 0:
        raise ValueError(f'{path}: weight percents must sum to a positive value')
    feedstock['wt_frac'] = feedstock['weight_percent'] / total

    for key in ('molar_mass_g_mol', 'density_g_cm3', 'molar_volume_cm3_mol'):
        feedstock[key] = [mineral_db[m][key] for m in feedstock['mineral']]

    if 'diameter_um' not in feedstock.columns:
        feedstock['diameter_um'] = np.nan
    feedstock['diameter_um'] = feedstock['diameter_um'].fillna(default_diameter_um)

    if density_g_cm3 is None:
        density_g_cm3 = 1.0 / (feedstock['wt_frac'] / feedstock['density_g_cm3']).sum()

    return feedstock, density_g_cm3


def load_schedule(path, n_years):
    """Read an application schedule file into a per-year array of rates.

    Only years with an application need a row; an absent year means no rock goes
    on. Multiple rows for the same year are summed, so split applications within
    a year work without a format change.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with columns ``year`` (1-indexed) and ``rate_t_ha``. Lines beginning
        with ``#`` are comments. A header-only file is a valid "never apply"
        schedule.
    n_years : int
        Length of the run.

    Returns
    -------
    numpy.ndarray
        Application rate (t/ha) for each year, length ``n_years``.
    """
    schedule = pd.read_csv(path, comment='#')

    missing_cols = {'year', 'rate_t_ha'} - set(schedule.columns)
    if missing_cols:
        raise ValueError(f'{path} is missing required column(s): {sorted(missing_cols)}')

    rates = np.zeros(int(n_years), dtype=float)
    for _, row in schedule.iterrows():
        year = int(row['year'])
        if not 1 <= year <= n_years:
            raise ValueError(f'{path}: year {year} is outside a {n_years}-year run')
        if row['rate_t_ha'] < 0:
            raise ValueError(f'{path}: negative application rate in year {year}')
        rates[year - 1] += float(row['rate_t_ha'])

    return rates


# ---------------------------------------------------------------------------
# Feedstock -> volume fraction and surface area
# ---------------------------------------------------------------------------

def feedstock_roughness(option=DEFAULT_ROUGHNESS_OPTION, radius_m=None,
                        const_rf=DEFAULT_CONST_RF):
    """Roughness factor (dimensionless) for a grain of the given radius.

    Parameters
    ----------
    option : str or float
        One of ``'smooth'`` (1.0), ``'constant'`` (uses ``const_rf``),
        ``'NSB07'`` (20.0), ``'BM00'`` (10**0.7 * r**-0.1) or ``'B20'``
        (10**3.3 * r**0.33). A bare number is used directly. ``'NBS07'`` is
        accepted as an alias of ``'NSB07'``, since we keep messing that up.
    radius_m : float, optional
        Grain radius (m). Required only for the size-dependent calibrations.
    const_rf : float
        Roughness factor used when ``option='constant'``.

    Returns
    -------
    float
        Roughness factor.
    """
    rules = {
        'smooth': lambda r: 1.0,
        'NSB07': lambda r: 20.0,
        'BM00': lambda r: (10 ** 0.7) * (r ** -0.1),
        'B20': lambda r: (10 ** 3.3) * (r ** 0.33),
    }
    rules['NBS07'] = rules['NSB07']

    if option == 'constant':
        if const_rf is None:
            raise ValueError("const_rf must be provided when option='constant'")
        return float(const_rf)

    if option in rules:
        if radius_m is None and option in ('BM00', 'B20'):
            raise ValueError(f'radius_m is required for the {option!r} calibration')
        return float(rules[option](radius_m))

    try:
        return float(option)
    except (TypeError, ValueError):
        raise ValueError(f'unrecognised roughness option: {option!r}') from None


def feedstock_properties(feedstock, density_g_cm3, app_rate_t_ha, till_depth_m,
                         roughness_option=DEFAULT_ROUGHNESS_OPTION,
                         const_rf=DEFAULT_CONST_RF):
    """Per-mineral volume fraction and bulk surface area for a well-mixed tilled layer.

    Bulk surface area is returned in m2 per litre of soil, matching the units of
    a ``.surf`` file.

    Parameters
    ----------
    feedstock : pandas.DataFrame
        Output of `load_feedstock`.
    density_g_cm3 : float
        Bulk feedstock density.
    app_rate_t_ha : float
        Application rate (metric tonnes per hectare).
    till_depth_m : float
        Depth the feedstock is mixed into.
    roughness_option : str or float
        Roughness rule, passed to `feedstock_roughness`.
    const_rf : float
        Roughness factor used when ``roughness_option='constant'``.

    Returns
    -------
    pandas.DataFrame
        ``feedstock`` with ``roughness_factor``, ``ssa_m2_g``,
        ``vol_frac_in_feedstock``, ``vol_frac_in_soil`` and ``bsa_m2_L`` added.
    """
    if app_rate_t_ha < 0:
        raise ValueError('app_rate_t_ha must be non-negative')
    if till_depth_m <= 0:
        raise ValueError('till_depth_m must be positive')

    out = feedstock.copy()

    # t/ha -> kg/m2 is x1e3/1e4 = x0.1; then / (kg/m3) / m gives m3 rock per m3 soil
    vol_frac_applied = (app_rate_t_ha * 0.1) / (density_g_cm3 * 1e3) / till_depth_m

    out['vol_frac_in_feedstock'] = (
        out['molar_volume_cm3_mol'] * out['wt_frac'] / out['molar_mass_g_mol'] * density_g_cm3
    )
    out['vol_frac_in_soil'] = out['vol_frac_in_feedstock'] * vol_frac_applied

    diam_m = out['diameter_um'] * 1e-6
    out['roughness_factor'] = [
        feedstock_roughness(roughness_option, d / 2, const_rf) for d in diam_m
    ]
    # SSA of a sphere is 6 / (rho * d), scaled by the roughness factor
    out['ssa_m2_g'] = 6.0 / (out['density_g_cm3'] * 1e3 * diam_m) * out['roughness_factor'] / 1e3
    out['bsa_m2_L'] = out['ssa_m2_g'] * out['density_g_cm3'] * 1e3 * out['vol_frac_in_soil']

    return out


def props_for_schedule(rates, feedstock, density_g_cm3, **kwargs):
    """Pre-compute `feedstock_properties` for each distinct non-zero rate in a schedule.

    Parameters
    ----------
    rates : array_like
        Per-year application rates, from `load_schedule`.
    feedstock : pandas.DataFrame
        Output of `load_feedstock`.
    density_g_cm3 : float
        Bulk feedstock density.
    **kwargs
        Passed through to `feedstock_properties` (``till_depth_m``,
        ``roughness_option``, ``const_rf``).

    Returns
    -------
    dict
        Maps application rate to the corresponding properties DataFrame.
    """
    return {
        rate: feedstock_properties(feedstock, density_g_cm3, app_rate_t_ha=rate, **kwargs)
        for rate in sorted({float(r) for r in np.asarray(rates).ravel() if r > 0})
    }


# ---------------------------------------------------------------------------
# Depth distribution
# ---------------------------------------------------------------------------

def till_layer_mask(z, till_depth_m):
    """Boolean mask selecting the top ``till_depth_m`` of the grid, counted in cells.

    Counting cells rather than thresholding elevation matters due to floating point:
    i.e., 4.0-3.7 could yield 0.2999... which adds a cell to a 0.3m till layer.

    Parameters
    ----------
    z : array_like
        Cell-centre elevations (m), ascending, as stored in MIN3P contour output.
        The ground surface is taken to be the last element.
    till_depth_m : float
        Depth of the tilled layer.

    Returns
    -------
    numpy.ndarray of bool
        True for cells inside the tilled layer.
    """
    z = np.asarray(z, dtype=float)
    n_till = int(round(till_depth_m / np.diff(z).mean()))
    if n_till < 1:
        raise ValueError('till_depth_m is smaller than one grid cell')
    # Cell index counted from the ground surface downward: 0 is the surface cell.
    return np.arange(len(z))[::-1] < n_till


def depth_profile(z, till_depth_m, distribution='uniform', steepness=0.4):
    """Relative depth-distribution weight for each grid cell.

    Parameters
    ----------
    z : array_like
        Cell-centre elevations (m), ascending.
    till_depth_m : float
        Depth over which the feedstock is mixed.
    distribution : {'uniform', 'sigmoidal'}
        Profile shape. Both integrate to the same total, so switching between
        them redistributes rock without changing how much was applied.
    steepness : float
        Sigmoid steepness, used only when ``distribution='sigmoidal'``.

    Returns
    -------
    numpy.ndarray
        Weight per cell, same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    uniform = till_layer_mask(z, till_depth_m).astype(float)

    if distribution == 'uniform':
        return uniform
    if distribution != 'sigmoidal':
        raise ValueError(f'unknown distribution: {distribution!r}')

    n_till = int(round(till_depth_m / np.diff(z).mean()))
    idx = np.arange(len(z))[::-1]
    raw = 1.0 / (1.0 + np.exp(steepness * (idx - n_till)))
    scaled = raw * uniform.sum() / raw.sum()
    return np.where(scaled < 1e-6, 0.0, scaled)


# ---------------------------------------------------------------------------
# Surface-area bookkeeping
# ---------------------------------------------------------------------------

def advance_surface_area(bsa_0, phi_0, phi_now, columns, minerals,
                         bsa_columns=None, exponent=SA_EXPONENT):
    """Surface area after dissolution, following MIN3P's ``twothird`` update law.

    MIN3P 2/3 relationship for surface area is: ``A = A0 * (phi/phi0)**(2/3)``
    Since ``A0`` and ``phi0`` were set when the run started, the current area is 
    recoverable from the reported volume fraction.

    Only the named minerals are updated. The law should be trusted for feedstock
    minerals only: soil minerals declared ``'twothird-mix'`` that precipitate
    rather than dissolve go through a nucleation branch this does not capture,
    and the arrays also carry x/y/z and porosity columns where it is meaningless.

    Parameters
    ----------
    bsa_0 : numpy.ndarray
        Bulk surface area written at the start of the run (m2/L), shape
        ``(n_variables, n_cells)``.
    phi_0 : numpy.ndarray
        Volume fraction written at the start of the run.
    phi_now : numpy.ndarray
        Volume fraction from the run's final output.
    columns : sequence of str
        Variable names for ``phi_0`` and ``phi_now``.
    minerals : sequence of str
        Minerals to update. Everything else is carried through unchanged.
    bsa_columns : sequence of str, optional
        Variable names for ``bsa_0``. Defaults to ``columns``; pass explicitly
        when the surface-area array has its own column list (a ``.surf`` file
        has no trailing ``porosity`` column, unlike a ``.min`` file).
    exponent : float
        Surface-area scaling exponent.

    Returns
    -------
    numpy.ndarray
        Copy of ``bsa_0`` with the named minerals advanced to ``phi_now``.
    """
    columns = list(columns)
    bsa_columns = columns if bsa_columns is None else list(bsa_columns)
    bsa_now = np.asarray(bsa_0, dtype=float).copy()

    for mineral in minerals:
        pi = columns.index(mineral)
        bi = bsa_columns.index(mineral)
        if np.any(phi_0[pi] <= 0):
            raise ValueError(
                f'phi_0 for {mineral!r} must be strictly positive (use the {PHI_FLOOR:g} floor)'
            )
        bsa_now[bi] = bsa_0[bi] * (phi_now[pi] / phi_0[pi]) ** exponent

    return bsa_now


def implied_ssa(phi, bsa, columns, minerals, mineral_db, bsa_columns=None):
    """Specific surface area (m2/g) implied by volume fraction and bulk surface area.

    Inverts ``BSA = SSA * rho * 1000 * phi``. Where a cell holds both aged and
    fresh grains this is the mass-weighted mean over the two cohorts, which is
    what summing their bulk surface areas implies.

    Parameters
    ----------
    phi, bsa : numpy.ndarray
        Volume fractions and bulk surface areas (m2/L).
    columns : sequence of str
        Variable names for ``phi``.
    minerals : sequence of str
        Minerals to report.
    mineral_db : dict
        Output of `read_mineral_database`, for mineral densities.
    bsa_columns : sequence of str, optional
        Variable names for ``bsa``. Defaults to ``columns``.

    Returns
    -------
    dict
        Maps mineral name to a per-cell array of specific surface area.
    """
    columns = list(columns)
    bsa_columns = columns if bsa_columns is None else list(bsa_columns)

    out = {}
    for mineral in minerals:
        pi = columns.index(mineral)
        bi = bsa_columns.index(mineral)
        denom = phi[pi] * mineral_db[mineral]['density_g_cm3'] * 1e3
        out[mineral] = np.divide(
            bsa[bi], denom, out=np.zeros_like(np.asarray(bsa[bi], dtype=float)), where=denom > 0
        )
    return out


# ---------------------------------------------------------------------------
# Mixing rules
# ---------------------------------------------------------------------------

def mix_homogeneous(phi, bsa, columns, bsa_columns, minerals, mix_mask):
    """Redistribute feedstock uniformly through the tilled layer.

    Volume fraction and bulk surface area are each averaged over the mixed cells,
    which conserves feedstock mass and total reactive surface area and therefore
    leaves the layer-integrated specific surface area unchanged. Material below
    the tilled layer is untouched.
    """
    phi_out = phi.copy()
    bsa_out = bsa.copy()
    for mineral in minerals:
        pi = columns.index(mineral)
        bi = bsa_columns.index(mineral)
        phi_out[pi, mix_mask] = phi[pi, mix_mask].mean()
        bsa_out[bi, mix_mask] = bsa[bi, mix_mask].mean()
    return phi_out, bsa_out


def mix_none(phi, bsa, columns, bsa_columns, minerals, mix_mask):
    """Leave the profile as-is: fresh feedstock layers onto the existing distribution."""
    return phi.copy(), bsa.copy()


# Mixing-rule registry. New rules (partial mixing, depth-decaying tillage, ...)
# are added here and become available to `apply_feedstock` and `till_soil`
# without either needing to change.
MIX_RULES = {
    'homogeneous': mix_homogeneous,
    'none': mix_none,
}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_feedstock(phi, bsa, columns, props, profile, mix_mask=None,
                    bsa_columns=None, mix_rule=DEFAULT_MIX_RULE, phi_floor=PHI_FLOOR):
    """Add a feedstock application to existing volume fractions and surface areas.

    Both quantities are added rather than overwritten, so this is valid for the
    first application (onto a floor-valued profile) and for every re-application
    onto partially dissolved rock.

    Parameters
    ----------
    phi : numpy.ndarray
        Mineral volume fractions, shape ``(n_variables, n_cells)``, as read from
        a ``.min`` file or a ``.gsv`` output file.
    bsa : numpy.ndarray
        Mineral bulk surface areas (m2/L), matching a ``.surf`` file.
    columns : sequence of str
        Variable names for ``phi``.
    props : pandas.DataFrame
        Output of `feedstock_properties`.
    profile : numpy.ndarray
        Relative depth weights from `depth_profile`, length ``n_cells``.
    mix_mask : numpy.ndarray of bool, optional
        Cells the mixing rule acts on, from `till_layer_mask`. Required unless
        ``mix_rule='none'``.
    bsa_columns : sequence of str, optional
        Variable names for ``bsa``. Defaults to ``columns``.
    mix_rule : str
        Key into `MIX_RULES`.
    phi_floor : float
        Minimum volume fraction written for a feedstock mineral.

    Returns
    -------
    phi_new, bsa_new : numpy.ndarray
        Copies of the inputs with the application added and mixing applied.
    """
    columns = list(columns)
    bsa_columns = columns if bsa_columns is None else list(bsa_columns)
    minerals = list(props['mineral'])

    missing = [m for m in minerals if m not in columns or m not in bsa_columns]
    if missing:
        raise KeyError(
            f'feedstock minerals absent from the mineral arrays: {missing}. '
            "They must be declared in the 'minerals' block of the .dat deck."
        )
    if mix_rule not in MIX_RULES:
        raise ValueError(f'unknown mix_rule {mix_rule!r}; options: {sorted(MIX_RULES)}')
    if mix_rule != 'none' and mix_mask is None:
        raise ValueError(f'mix_mask is required for mix_rule={mix_rule!r}')

    phi_new = np.asarray(phi, dtype=float).copy()
    bsa_new = np.asarray(bsa, dtype=float).copy()

    for _, row in props.iterrows():
        pi = columns.index(row['mineral'])
        bi = bsa_columns.index(row['mineral'])
        phi_new[pi] = phi_new[pi] + row['vol_frac_in_soil'] * profile
        bsa_new[bi] = bsa_new[bi] + row['bsa_m2_L'] * profile

    phi_new, bsa_new = MIX_RULES[mix_rule](
        phi_new, bsa_new, columns, bsa_columns, minerals, mix_mask
    )

    for mineral in minerals:
        phi_new[columns.index(mineral)] = np.maximum(
            phi_new[columns.index(mineral)], phi_floor
        )

    return phi_new, bsa_new


def till_soil(phi, bsa, columns, minerals, mix_mask, bsa_columns=None,
              mix_rule=DEFAULT_MIX_RULE):
    """Apply a mixing rule without adding any feedstock. (i.e., till but no new rock)

    Parameters
    ----------
    phi, bsa : numpy.ndarray
        Mineral volume fractions and bulk surface areas (m2/L).
    columns : sequence of str
        Variable names for ``phi``.
    minerals : sequence of str
        Feedstock minerals to mix. Soil minerals are left alone.
    mix_mask : numpy.ndarray of bool
        Cells the mixing rule acts on, from `till_layer_mask`.
    bsa_columns : sequence of str, optional
        Variable names for ``bsa``. Defaults to ``columns``.
    mix_rule : str
        Key into `MIX_RULES`.

    Returns
    -------
    phi_new, bsa_new : numpy.ndarray
        Copies of the inputs with the mixing rule applied.
    """
    columns = list(columns)
    bsa_columns = columns if bsa_columns is None else list(bsa_columns)

    if mix_rule not in MIX_RULES:
        raise ValueError(f'unknown mix_rule {mix_rule!r}; options: {sorted(MIX_RULES)}')

    return MIX_RULES[mix_rule](
        np.asarray(phi, dtype=float), np.asarray(bsa, dtype=float),
        columns, bsa_columns, list(minerals), mix_mask
    )


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------

def write_restart_minerals(prev_dir, prev_prefix, next_dir, next_prefix, minerals,
                           props=None, profile=None, mix_mask=None,
                           mix_rule=DEFAULT_MIX_RULE, till_without_application=False,
                           exponent=SA_EXPONENT, phi_floor=PHI_FLOOR,
                           return_arrays=False):
    """Write the ``.min`` and ``.surf`` files that start the next year of a chain.

    Reads the volume fractions and surface areas written at the start of the
    previous year, together with that year's final ``.gsv`` output, reconstructs
    the aged cohort's SA following MIN3p rule, optionally applies fresh
    feedstock and/or tillage, and writes the result.

    This handles minerals only. The flow and aqueous restart files (``.ivs`` from
    the last ``.gsp``, ``.aqt`` from the last ``.gst``) carry no bookkeeping and
    are the chain driver's responsibility.

    Parameters
    ----------
    prev_dir : str or pathlib.Path
        Directory holding the previous year's run.
    prev_prefix : str
        Run prefix for the previous year, as given in its ``root.dat``.
    next_dir : str or pathlib.Path
        Directory to write the next year's input files into. Created if absent.
    next_prefix : str
        Run prefix for the next year.
    minerals : sequence of str
        Feedstock minerals tracked across the restart.
    props : pandas.DataFrame, optional
        Output of `feedstock_properties`. When omitted no feedstock is applied.
    profile : numpy.ndarray, optional
        Relative depth weights from `depth_profile`. Required when ``props`` is
        given.
    mix_mask : numpy.ndarray of bool, optional
        Cells the mixing rule acts on, from `till_layer_mask`.
    mix_rule : str
        Key into `MIX_RULES`.
    till_without_application : bool
        Whether to mix the plough layer in a year with no application.
    exponent : float
        Surface-area scaling exponent for the aged cohort.
    phi_floor : float
        Minimum volume fraction written for a feedstock mineral.
    return_arrays : bool
        If True, also return the written arrays and their column lists.

    Returns
    -------
    None or tuple
        ``(phi_new, phi_columns, bsa_new, bsa_columns)`` when ``return_arrays``.
    """
    prev_dir, next_dir = Path(prev_dir), Path(next_dir)
    next_dir.mkdir(parents=True, exist_ok=True)

    # State we wrote at the start of the previous year
    phi_0, phi_columns, _ = read_min3p(f'{prev_prefix}.min', folder=prev_dir)
    bsa_0, bsa_columns, _ = read_min3p(f'{prev_prefix}.surf', folder=prev_dir)

    # Where MIN3P left the volume fractions at the end of that year
    gsv, gsv_columns, _ = read_min3p_sequence(f'{prev_prefix}_1.gsv', folder=prev_dir)
    phi_end = gsv[-1]
    if list(gsv_columns) != list(phi_columns):
        raise ValueError(
            f'{prev_prefix}_1.gsv columns do not match {prev_prefix}.min columns'
        )

    # Aged cohort: recover the surface area MIN3P had reached
    bsa_end = advance_surface_area(bsa_0, phi_0, phi_end, phi_columns, minerals,
                                   bsa_columns=bsa_columns, exponent=exponent)

    if props is not None:
        if profile is None:
            raise ValueError('profile is required when props is given')
        phi_new, bsa_new = apply_feedstock(
            phi_end, bsa_end, phi_columns, props, profile, mix_mask=mix_mask,
            bsa_columns=bsa_columns, mix_rule=mix_rule, phi_floor=phi_floor,
        )
    elif till_without_application:
        phi_new, bsa_new = till_soil(
            phi_end, bsa_end, phi_columns, minerals, mix_mask,
            bsa_columns=bsa_columns, mix_rule=mix_rule,
        )
    else:
        phi_new, bsa_new = phi_end.copy(), bsa_end.copy()

    write_min3p(phi_new, f'{next_prefix}.min', list(phi_columns), folder=next_dir,
                prefix=next_prefix, label='phi_i, T = initial')
    write_min3p(bsa_new, f'{next_prefix}.surf', list(bsa_columns), folder=next_dir,
                prefix=next_prefix, label='bsa_i (m2/L), T = initial')

    if return_arrays:
        return phi_new, list(phi_columns), bsa_new, list(bsa_columns)
    return None
