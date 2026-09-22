"""Build the ``.dat`` decks and transient forcing files for a chained run.

Deck generation is a *build-time* activity, not a run-time one: every year's deck
is fully determined before any of them has run, so the whole tree is generated
once, locally, where it can be inspected. Nothing here runs on the VM -- by the
time a chain starts, `byte_util.restart` only writes data files.

The helpers below wrap `min3p.input.InputFile` for the edits a chained run needs:

* **Fortran float formatting.** MIN3P reads ``1.3d-1``, not ``1.3e-1``. Anything
  written into a deck goes through `byte_util.util.single_to_double_float`.
* **The database path encodes directory depth.** ``'../../../../databases'`` is
  correct from ``min3p_runs/<site>/<scenario>/`` and wrong from a ``year01/``
  subdirectory, and the cluster's ``stage_min3p`` relocates the databases again.
  `set_database_dir` computes it rather than copying it.
* **The deck's scalar boundary value must match the first transient record.**
  The top-boundary flux and transpiration factor in the deck apply until the
  first ``.bcvs``/``.soi`` record takes over; if they disagree the first timestep
  is wrong. `year_forcing` returns the files *and* the two scalars together so
  they cannot drift apart.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .met_transport import create_bcvs_soi, write_transient
from .util import single_to_double_float

__all__ = [
    'DEFAULT_DAYS_PER_YEAR',
    'DEFAULT_TRANSIENT_CONTROL_PLANES_CM',
    'DEFAULT_TRANSIENT_MONITOR_DEPTHS_CM',
    'STEADY_MEANS',
    'collapse_initial_chemistry_zones',
    'control_volumes_for_depths',
    'enable_transient_forcing',
    'free_concentration_constraints',
    'horizon_zones',
    'relative_database_dir',
    'restart_from_files',
    'set_database_dir',
    'set_problem_title',
    'set_run_window',
    'set_surface_forcing',
    'set_transient_output',
    'write_year_forcing',
    'year_forcing',
    'year_windows',
]

# Uniform segment length. The forcing record is 20 years and a chain needs 10,
# so segments are cut to equal length rather than to calendar years.
DEFAULT_DAYS_PER_YEAR = 365

# How a steady (non-resampled) forcing window picks its constant value.
# ``'record'`` averages the whole forcing record ()"long-term mean")
# ``'window'`` averages only the segment (year), so interannual variability persists
STEADY_MEANS = ('record', 'window')

# Depths (cm) monitored as single points in the met_forcing decks.
DEFAULT_TRANSIENT_MONITOR_DEPTHS_CM = (0, 1, 5, 10, 30)

# Depths (cm) monitored as flux control planes -- each contributes the pair of
# cells straddling it, which is what `byte_util.met_transport.calc_tracer_flux`
# needs to compute specific discharge across the plane. These four match its
# default ``control_planes``.
DEFAULT_TRANSIENT_CONTROL_PLANES_CM = (50, 100, 200, 300)


# ---------------------------------------------------------------------------
# Deck edits
# ---------------------------------------------------------------------------

def set_problem_title(infile, title):
    """Set the problem title in the global control parameters block."""
    infile.global_control_parameters.problem_title = title
    return infile


def set_run_window(infile, final_time, output_times=None, maximum_time_step=None,
                   start_time=None):
    """Set how long a segment runs and when it writes spatial output.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    final_time : float
        End of the segment, in the deck's time unit (days for these decks).
    output_times : sequence of float, optional
        Spatial output times. Left alone when omitted.
    maximum_time_step : float, optional
        Maximum time step. Left alone when omitted.
    start_time : float, optional
        Start of the segment. Left alone when omitted; each segment of a chain
        normally restarts its clock at zero.

    Returns
    -------
    min3p.input.InputFile
    """
    tsc = infile.time_step_control
    tsc.final_time = float(final_time)
    if maximum_time_step is not None:
        tsc.maximum_time_step = float(maximum_time_step)
    if start_time is not None:
        tsc.start_time = float(start_time)

    if output_times is not None:
        times = [float(t) for t in output_times]
        if times and max(times) > final_time:
            raise ValueError(
                f'output time {max(times):g} is beyond final_time {final_time:g}'
            )
        infile.output_control.output_of_spatial_data = times

    return infile


def set_surface_forcing(infile, flux_m_s=None, transpiration_factor=None):
    """Set the initial top-boundary flux and transpiration factor.

    These are the values that apply until the first ``.bcvs``/``.soi`` record
    takes over, so they should be that record's values -- see `year_forcing`,
    which returns them alongside the files.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    flux_m_s : float, optional
        Infiltration at the top boundary (m/s). Written in Fortran double
        notation.
    transpiration_factor : float, optional
        Root water uptake factor (1/d), applied to every vsflow zone.

    Returns
    -------
    min3p.input.InputFile
    """
    if flux_m_s is not None:
        top = infile.boundary_conditions_vsflow.zones[0]
        top.boundary_value = single_to_double_float(f'{float(flux_m_s):0.3e}')

    if transpiration_factor is not None:
        for zone in infile.physical_parameters_vsflow.zones:
            zone.root_water_uptake.transpiration_factor = f'{float(transpiration_factor):0.6f}'

    return infile


def enable_transient_forcing(infile, boundary_conditions=True, transpiration=True):
    """Switch on transient boundary conditions and/or transient transpiration.

    Adds the ``'transient boundary conditions'`` keyword to the vsflow boundary
    block and sets the transient transpiration flag, which is what makes MIN3P
    look for the ``.bcvs`` and ``.soi`` files. Idempotent.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    boundary_conditions, transpiration : bool
        Which to enable.

    Returns
    -------
    min3p.input.InputFile
    """
    if boundary_conditions:
        bcvs = infile.boundary_conditions_vsflow
        if 'transient boundary conditions' not in bcvs.text:
            bcvs.text += "\n'transient boundary conditions'\n\n"

    if transpiration:
        infile.physical_parameters_vsflow.transient_transpiration = True

    return infile


def restart_from_files(infile, flow=True, aqueous=True, minerals=True, areas=True,
                       cec=True, comment='! Initial condition read from previous segment'):
    """Point the deck's initial conditions at files instead of inline values.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    flow : bool
        Read the variably-saturated initial condition from ``.ivs``. Unlike the
        reactive-transport flags this is one-way: False leaves the block alone
        rather than rebuilding zone-based initial conditions.
    aqueous : bool
        Read initial aqueous component concentrations from ``.aqt``.
    minerals : bool
        Read initial mineral volume fractions from ``.min``.
    areas : bool
        Read initial mineral areas from ``.surf``.
    cec : bool
        Read cation exchange capacity and bulk density from ``.cec``.

    Returns
    -------
    min3p.input.InputFile
    """
    if flow:
        infile.initial_conditions_vsflow.text = (
            f"{comment}\n'read initial condition from file'\n"
        )

    # Set explicitly rather than only-when-True: a base deck may already carry a
    # flag, and restart_from_files(cec=False) should clear it rather than be a
    # silent no-op.
    icrt = infile.initial_conditions_reactive_transport
    icrt.read_initial_aqueous_component_concentrations_from_file = bool(aqueous)
    icrt.read_initial_mineral_volume_fractions_from_file = bool(minerals)
    icrt.read_initial_mineral_areas_from_file = bool(areas)
    icrt.read_cec_from_file = bool(cec)

    return infile


def collapse_initial_chemistry_zones(infile, keep='A horizon chem'):
    """Collapse the reactive-transport *initial conditions* to a single zone.

    Removes the per-horizon *initial chemistry* -- starting pH, component 
    concentrations, mineral volume fractions, exchange capacity. Once
    `restart_from_files` is on those are supplied from ``.aqt``, ``.min``,
    ``.surf`` and ``.cec``, treading each cell individually. 

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    keep : str
        Name of the initial-condition zone to retain.

    Returns
    -------
    min3p.input.InputFile
    """
    icrt = infile.initial_conditions_reactive_transport
    names = [zone.name for zone in icrt.zones]
    if keep not in names:
        raise ValueError(f'zone {keep!r} not found; have {names}')

    for name in names:
        if name != keep:
            infile.delete_zone(name, block_names=['initial_conditions_reactive_transport'])

    return infile


def horizon_zones(infile):
    """Soil horizons declared by a deck, as depths below ground.

    Horizon boundaries are kept in the config data for interpreting results. 
    No affect on the simulation itself.

    Depths are measured down from the top of the domain, matching the
    ``top_m``/``bottom_m`` convention of ``soil_physical_parameters.parquet``.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to read.

    Returns
    -------
    list of dict
        One entry per horizon, shallowest first, with ``name``, ``top_m``,
        ``bottom_m``, ``z_top``, ``z_bottom`` and ``porosity``.
    """
    surface = float(infile.spatial_discretization.z_bounds[1])

    horizons = []
    for zone in infile.physical_parameters_porous_medium.zones:
        z_bottom, z_top = (float(v) for v in zone.extent_of_zone[-2:])
        horizons.append({
            'name': zone.name,
            'top_m': round(surface - z_top, 6),
            'bottom_m': round(surface - z_bottom, 6),
            'z_top': z_top,
            'z_bottom': z_bottom,
            'porosity': float(zone.porosity),
        })

    return sorted(horizons, key=lambda h: h['top_m'])


def free_concentration_constraints(infile, zone=0):
    """Convert ``'ph'`` and ``'pco2'`` constraints to ``'free'``.

    Required once initial aqueous concentrations are read from file: the values
    in the deck are overridden, but MIN3P still parses the block, and a ``'ph'``
    or ``'pco2'`` constraint there would be applied rather than the file value.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    zone : int
        Index of the initial-condition zone to convert.

    Returns
    -------
    min3p.input.InputFile
    """
    records = infile.initial_conditions_reactive_transport.zones[zone].concentration_input.records
    for record in records:
        value = record.value()
        if 'ph' in value:
            record.replace_content(f"{value[0]:0.2f}           'free'")
        elif 'pco2' in value:
            record.replace_content(f"{single_to_double_float(f'{value[0]:0.4e}')}      'free'")
    return infile


def control_volumes_for_depths(monitor_depths_cm=(), control_planes_cm=(),
                               nz=401, dz=0.01):
    """Control volume numbers for a set of monitoring depths.

    MIN3P identifies transient output locations by 1-based control volume number
    counted from the bottom of the domain, so a depth below ground has to be
    converted -- the committed decks carry hand-computed values like ``351`` with
    a comment explaining half of them. Deriving them removes an easy off-by-one.

    A *control plane* contributes two cells rather than one: the pair straddling
    the plane, which is what a flux calculation needs to form a gradient. A
    *monitor depth* contributes a single cell.

    Parameters
    ----------
    monitor_depths_cm : sequence of float
        Depths below ground recorded as single points.
    control_planes_cm : sequence of float
        Depths below ground recorded as straddling pairs.
    nz : int
        Number of control volumes in z. The ground surface is volume ``nz``.
    dz : float
        Cell thickness (m).

    Returns
    -------
    list of int
        Control volume numbers, shallowest first, deduplicated.
    """
    cell_cm = dz * 100

    def volume_at(depth_cm):
        cells_down = int(round(depth_cm / cell_cm))
        return nz - cells_down

    volumes = []
    for depth in monitor_depths_cm:
        volumes.append(volume_at(depth))
    for depth in control_planes_cm:
        upper = volume_at(depth)
        volumes.extend([upper, upper - 1])

    for volume in volumes:
        if not 1 <= volume <= nz:
            raise ValueError(
                f'control volume {volume} is outside the grid (1..{nz}); '
                'a requested depth is above the surface or below the domain'
            )

    return sorted(set(volumes), reverse=True)


def set_transient_output(infile, locations=None, interval=None,
                         monitor_depths_cm=None, control_planes_cm=None):
    """Set which control volumes write transient output, and how often.

    ``interval`` counts *timesteps*, and MIN3P's timestep is adaptive so I'm 
    not sure how to fix the cadence. Still, we may want to modify the interval
    for cases where the met boundary forcing comes in at a high resolution.

    Give either ``locations`` (explicit control volume numbers) or the two depth
    arguments, in which case the grid is read from the deck.

    Parameters
    ----------
    infile : min3p.input.InputFile
        Deck to modify, in place.
    locations : sequence of int, optional
        Control volume numbers. Left alone when omitted.
    interval : int, optional
        Timesteps between transient records. Left alone when omitted.
    monitor_depths_cm, control_planes_cm : sequence of float, optional
        Depths to convert via `control_volumes_for_depths`, using the deck's own
        grid. Ignored when ``locations`` is given.

    Returns
    -------
    min3p.input.InputFile
    """
    if locations is None and (monitor_depths_cm is not None or control_planes_cm is not None):
        sd = infile.spatial_discretization
        nz = int(sd.number_of_control_volumes_z)
        zmin, zmax = sd.z_bounds
        locations = control_volumes_for_depths(
            monitor_depths_cm or (), control_planes_cm or (),
            nz=nz, dz=(zmax - zmin) / (nz - 1),
        )

    if locations is not None:
        locations = [int(v) for v in locations]
        if not locations:
            raise ValueError('at least one transient output location is required')
        infile.output_control.output_of_transient_data = locations

    if interval is not None:
        if int(interval) < 1:
            raise ValueError('transient output interval must be at least 1')
        infile.output_control.transient_output_interval = int(interval)

    return infile


def relative_database_dir(run_dir, database_dir, max_parents=8):
    """Relative path from a run directory to the MIN3P database directory.

    The deck stores this as a relative path, so it depends on how deep the run
    directory sits. Adding a ``year01/`` level to the existing layout changes it,
    and it changes again when a job stages the run somewhere else.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Directory the deck will live in.
    database_dir : str or pathlib.Path
        Directory holding the MIN3P databases.
    max_parents : int or None
        Refuse to return a path that climbs more than this many levels. A long
        ``../../..`` chain means the run tree and the databases are on different
        branches of the filesystem, which produces a path that is technically
        correct, unreadable, and broken as soon as the run is staged elsewhere.
        Pass an absolute path to `set_database_dir` in that case, or generate the
        run tree alongside the databases. ``None`` disables the check.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        When the relative path would climb more than ``max_parents`` levels.
    """
    relative = os.path.relpath(Path(database_dir).resolve(), Path(run_dir).resolve())

    parents = relative.split(os.sep).count(os.pardir)
    if max_parents is not None and parents > max_parents:
        raise ValueError(
            f'database directory is {parents} levels above the run directory, which '
            f'gives an unusable relative path ({relative!r}). Generate the run tree '
            f'alongside the databases, or pass an absolute path to set_database_dir.'
        )
    return relative


def set_database_dir(infile, path):
    """Set the database directory. Quoting is handled by `min3p`."""
    infile.geochemical_system.database_directory = str(path)
    return infile


# ---------------------------------------------------------------------------
# Transient forcing
# ---------------------------------------------------------------------------

def year_windows(start, n_years, days_per_year=DEFAULT_DAYS_PER_YEAR):
    """Half-open ``[start, end)`` windows for each segment of a chain.

    Segments are equal length rather than calendar years. Ten 365-day segments
    total 3650 days, matching the existing unchained runs, and nothing
    downstream requires a segment to align with a calendar year. The cost is
    that windows drift by a day per leap year, which matters only for monthly
    resampling.

    Parameters
    ----------
    start : str or pandas.Timestamp
        Start of the first segment.
    n_years : int
        Number of segments.
    days_per_year : float
        Segment length in days.

    Returns
    -------
    list of (pandas.Timestamp, pandas.Timestamp)
    """
    start = pd.Timestamp(start)
    step = pd.Timedelta(days=days_per_year)
    return [(start + i * step, start + (i + 1) * step) for i in range(int(n_years))]


def year_forcing(met_forcing, start, end, freq=None, steady_mean='record',
                 dz=0.01, bottom_bc=0.0):
    """Slice one segment of met forcing into ``.bcvs``/``.soi`` records.

    Parameters
    ----------
    met_forcing : pandas.DataFrame
        Hourly forcing indexed by datetime, with ``surface_flux_mm.hr`` and
        ``transpiration_mm.hr`` columns. Not modified.
    start, end : str or pandas.Timestamp
        Half-open window ``[start, end)``.
    freq : {'h', 'D', 'MS'} or None
        Resampling frequency. ``None`` means steady forcing -- no transient
        files, just the two scalars for the deck.
    steady_mean : {'record', 'window'}
        Which mean a steady window uses. ``'record'`` averages the whole
        ``met_forcing`` passed in, so every segment gets the same value; if you
        hand it 20 years and chain 10, that is the 20-year mean, matching the
        existing unchained runs. ``'window'`` averages only ``[start, end)``.
        Ignored when ``freq`` is not None.
    dz : float
        Grid cell size, used to convert transpiration to a rate factor.
    bottom_bc : float
        Head applied at the bottom boundary.

    Returns
    -------
    bcvs : pandas.DataFrame or None
        Transient boundary conditions, with the first record removed -- that one
        lives in the deck instead, as ``initial_flux_m_s``.
    soi : pandas.DataFrame or None
        Transient transpiration, likewise.
    initial_flux_m_s : float
        Top-boundary flux for the deck.
    initial_transpiration_factor : float
        Transpiration factor for the deck (1/d).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = met_forcing.loc[(met_forcing.index >= start) & (met_forcing.index < end)].copy()
    if window.empty:
        raise ValueError(f'no forcing records in [{start}, {end})')

    if freq is None:
        if steady_mean not in STEADY_MEANS:
            raise ValueError(f'steady_mean must be one of {STEADY_MEANS}, got {steady_mean!r}')
        source = met_forcing if steady_mean == 'record' else window
        flux = float((source['surface_flux_mm.hr'] / 1000 / 60 / 60).mean())
        transpiration = float((source['transpiration_mm.hr'] / 1000 * 24).mean() / dz)
        return None, None, flux, transpiration

    # delete_first_record=False so the first record is available for the deck
    bcvs, soi = create_bcvs_soi(window, freq=freq, start_date=window.index[0],
                                bottom_bc=bottom_bc, dz=dz, delete_first_record=False)

    initial_flux = float(bcvs['surface_flux_m.s'].iloc[0])
    initial_transpiration = float(soi['transpiration_factor'].iloc[0])

    return bcvs.iloc[1:], soi.iloc[1:], initial_flux, initial_transpiration


def write_year_forcing(bcvs, soi, run_dir, prefix):
    """Write ``.bcvs`` and ``.soi`` files for one segment.

    Returns
    -------
    dict
        Maps ``'bcvs'``/``'soi'`` to the path written. Empty when both are None
        (the steady-forcing case).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    if bcvs is not None:
        path = run_dir / f'{prefix}.bcvs'
        write_transient(path, bcvs)
        written['bcvs'] = path
    if soi is not None:
        path = run_dir / f'{prefix}.soi'
        write_transient(path, soi)
        written['soi'] = path
    return written
