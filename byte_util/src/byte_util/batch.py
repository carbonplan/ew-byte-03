"""Expand a sweep specification into a batch of chained MIN3P runs.

A batch is described by a small dictionary -- the *spec* -- which a notebook
edits and this module turns into a design matrix, one row per simulation. 

The spec has four sections::

    SPEC = {
        'constant': {...},              # applied to every run
        'by_site':  {'spinup_dir': {'Cecil': ..., 'Yolo': ...}},
        'vary':     {'site':       {'values': [...]},        # categorical
                     'total_t_ha': {'range': (1, 40),        # continuous
                                    'scale': 'log'}},
        'design':   'factorial',        # 'factorial' | 'oat' | 'lhs'
    }

Factors declare either ``values`` (discrete levels) or ``range`` (continuous
bounds), and which of the two is allowed depends on the design:

``factorial``
    Every combination of ``values``. Continuous factors are rejected -- a
    product needs levels, so give the levels you want.
``oat``
    One factor moved off an explicit ``baseline`` at a time. Also needs
    ``values``, for the same reason.
``lhs``
    Latin hypercube over the ``range`` factors, crossed with every combination
    of the ``values`` factors. A Latin hypercube is a construction on a
    continuous unit cube, so it has nothing to say about ``forcing='hourly'``
    versus ``'daily'``; those are crossed instead. Run count is therefore
    ``n_categorical_cells x n_samples``, which grows quickly -- check it before
    writing anything.
"""

import json
import shutil
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from min3p.input import InputFile
from scipy.stats import qmc

from .deck import (
    DEFAULT_TRANSIENT_CONTROL_PLANES_CM,
    DEFAULT_TRANSIENT_MONITOR_DEPTHS_CM,
    enable_transient_forcing,
    free_concentration_constraints,
    collapse_initial_chemistry_zones,
    relative_database_dir,
    restart_from_files,
    set_database_dir,
    set_problem_title,
    set_run_window,
    set_surface_forcing,
    set_transient_output,
    write_year_forcing,
    year_forcing,
)
from .restart import seed_run_state, write_root_dat

__all__ = [
    'DESIGNS',
    'SCHEDULE_KINDS',
    'build_run_tree',
    'expand_design',
    'run_directory_name',
    'schedule_records',
    'write_provenance',
    'write_schedule',
]

#: Sweep designs `expand_design` understands.
DESIGNS = ('factorial', 'oat', 'lhs')

#: Application schedule shapes `schedule_records` can generate.
SCHEDULE_KINDS = ('pulse', 'annual', 'none')


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def _split_factors(vary):
    """Separate a ``vary`` mapping into categorical and continuous factors."""
    categorical, continuous = {}, {}

    for name, spec in vary.items():
        if not isinstance(spec, dict):
            raise TypeError(
                f'factor {name!r} must be a dict with "values" or "range", got {type(spec).__name__}'
            )
        has_values, has_range = 'values' in spec, 'range' in spec
        if has_values == has_range:
            raise ValueError(
                f'factor {name!r} must declare exactly one of "values" or "range"'
            )

        if has_values:
            levels = list(spec['values'])
            if not levels:
                raise ValueError(f'factor {name!r} has no values')
            categorical[name] = levels
        else:
            low, high = spec['range']
            if not low < high:
                raise ValueError(f'factor {name!r} needs range (low, high) with low < high')
            if spec.get('scale', 'linear') not in ('linear', 'log'):
                raise ValueError(f'factor {name!r} has unknown scale {spec["scale"]!r}')
            if spec.get('scale') == 'log' and low <= 0:
                raise ValueError(f'factor {name!r} needs a positive range for log scale')
            continuous[name] = dict(spec)

    return categorical, continuous


def _sample_continuous(continuous, n_samples, seed=None):
    """Latin hypercube sample of the continuous factors, as a DataFrame."""
    if not continuous:
        return pd.DataFrame(index=range(n_samples))

    names = list(continuous)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=n_samples)

    columns = {}
    for i, name in enumerate(names):
        spec = continuous[name]
        low, high = spec['range']
        if spec.get('scale', 'linear') == 'log':
            values = np.exp(np.log(low) + unit[:, i] * (np.log(high) - np.log(low)))
        else:
            values = low + unit[:, i] * (high - low)
        if spec.get('integer'):
            values = np.rint(values).astype(int)
        columns[name] = values

    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

def expand_design(vary, design='factorial', baseline=None, n_samples=None, seed=None,
                  constant=None, by_site=None, lhs_paired=True, run_id_template=None):
    """Expand a sweep specification into a design matrix.

    Parameters
    ----------
    vary : dict
        Factor name -> ``{'values': [...]}`` or ``{'range': (low, high), ...}``.
    design : {'factorial', 'oat', 'lhs'}
        See the module docstring for what each accepts.
    baseline : dict, optional
        Required for ``'oat'``: the value of every factor at the reference
        point. Explicit rather than "the first level of each".
    n_samples : int, optional
        Required for ``'lhs'``: hypercube samples per categorical cell.
    seed : int, optional
        Seed for the hypercube, recorded in the manifest for reproducibility.
    constant : dict, optional
        Columns applied unchanged to every row.
    by_site : dict, optional
        ``{column: {site: value}}``, looked up per row. Requires a ``site``
        column.
    lhs_paired : bool
        When True the same hypercube sample is reused in every categorical cell,
        so cells are directly comparable at identical continuous values -- which
        is what a pulse-versus-annual comparison wants. When False each cell is
        sampled independently.
    run_id_template : str, optional
        Format string over the row, e.g. ``'{site}_{forcing}'``. Defaults to
        joining the categorical values.

    Returns
    -------
    pandas.DataFrame
        One row per run, with a ``run_id`` column first.
    """
    if design not in DESIGNS:
        raise ValueError(f'design must be one of {DESIGNS}, got {design!r}')

    categorical, continuous = _split_factors(vary)

    if design in ('factorial', 'oat') and continuous:
        raise ValueError(
            f'{design!r} needs discrete levels, but {sorted(continuous)} declare "range". '
            'Give "values" instead, or use design="lhs".'
        )

    if design == 'factorial':
        rows = [dict(zip(categorical, combo))
                for combo in product(*categorical.values())]

    elif design == 'oat':
        if baseline is None:
            raise ValueError("design='oat' requires an explicit baseline")
        missing = sorted(set(categorical) - set(baseline))
        if missing:
            raise ValueError(f'baseline is missing factor(s): {missing}')

        rows = [dict(baseline)]
        for name, levels in categorical.items():
            if baseline[name] not in levels:
                raise ValueError(
                    f'baseline value {baseline[name]!r} for {name!r} is not among its levels'
                )
            for level in levels:
                if level != baseline[name]:
                    rows.append({**baseline, name: level})

    else:  # lhs
        if not n_samples:
            raise ValueError("design='lhs' requires n_samples")
        if not continuous:
            raise ValueError("design='lhs' needs at least one factor with a 'range'")

        cells = [dict(zip(categorical, combo))
                 for combo in product(*categorical.values())] or [{}]

        rows = []
        shared = _sample_continuous(continuous, n_samples, seed=seed) if lhs_paired else None
        for cell_index, cell in enumerate(cells):
            sample = shared if lhs_paired else _sample_continuous(
                continuous, n_samples, seed=None if seed is None else seed + cell_index)
            for _, values in sample.iterrows():
                rows.append({**cell, **values.to_dict()})

    frame = pd.DataFrame(rows)

    for column, value in (constant or {}).items():
        frame[column] = value

    for column, lookup in (by_site or {}).items():
        if 'site' not in frame.columns:
            raise ValueError(f'by_site column {column!r} needs a "site" factor')
        unknown = sorted(set(frame['site']) - set(lookup))
        if unknown:
            raise ValueError(f'by_site[{column!r}] has no entry for site(s): {unknown}')
        frame[column] = frame['site'].map(lookup)

    frame.insert(0, 'run_id', run_directory_name(frame, categorical, run_id_template))
    if frame['run_id'].duplicated().any():
        duplicated = sorted(frame.loc[frame['run_id'].duplicated(), 'run_id'].unique())
        raise ValueError(f'run_id is not unique: {duplicated[:3]}')

    return frame.reset_index(drop=True)


def run_directory_name(frame, categorical=None, template=None):
    """Build a readable, unique identifier for each row of a design.

    Uses the categorical factor values, which is what distinguishes runs at a
    glance, and appends a zero-padded index only when those are not enough --
    which is the usual case under LHS, where many rows share a cell.
    """
    if template is not None:
        names = [template.format(**row) for _, row in frame.iterrows()]
    elif categorical:
        names = ['_'.join(str(row[c]) for c in categorical) for _, row in frame.iterrows()]
    else:
        names = ['run'] * len(frame)

    counts = pd.Series(names).value_counts()
    if (counts > 1).any():
        width = len(str(len(frame)))
        names = [f'{name}_{i:0{width}d}' for i, name in enumerate(names)]

    return names


# ---------------------------------------------------------------------------
# Application schedules
# ---------------------------------------------------------------------------

def schedule_records(kind, total_t_ha=None, rate_t_ha=None, n_applications=None,
                     first_year=1, every=1):
    """Build an application schedule as a table of ``year, rate_t_ha``.

    Give either ``total_t_ha`` (split across the applications) or ``rate_t_ha``
    (applied at each one). Sweeping ``total_t_ha`` while crossing ``kind`` is
    what keeps a pulse and an annual arm at matched total mass, which is the
    comparison a timing experiment is built on; sweeping ``rate_t_ha`` lets
    total mass float instead.

    Parameters
    ----------
    kind : {'pulse', 'annual', 'none'}
        ``'pulse'`` applies once, ``'annual'`` applies ``n_applications`` times,
        ``'none'`` never applies.
    total_t_ha : float, optional
        Total mass across the whole schedule.
    rate_t_ha : float, optional
        Mass per application.
    n_applications : int, optional
        Number of applications; required for ``'annual'``.
    first_year : int
        Year of the first application.
    every : int
        Years between applications.

    Returns
    -------
    pandas.DataFrame
        Columns ``year`` and ``rate_t_ha``, one row per application.
    """
    if kind not in SCHEDULE_KINDS:
        raise ValueError(f'kind must be one of {SCHEDULE_KINDS}, got {kind!r}')

    if kind == 'none':
        return pd.DataFrame(columns=['year', 'rate_t_ha'])

    if (total_t_ha is None) == (rate_t_ha is None):
        raise ValueError('give exactly one of total_t_ha or rate_t_ha')

    if kind == 'pulse':
        n_applications = 1
    elif not n_applications:
        raise ValueError("kind='annual' requires n_applications")
    if n_applications < 1:
        raise ValueError('n_applications must be at least 1')
    if every < 1:
        raise ValueError('every must be at least 1')

    per_application = (float(rate_t_ha) if rate_t_ha is not None
                       else float(total_t_ha) / n_applications)
    if per_application < 0:
        raise ValueError('application rate must be non-negative')

    years = [first_year + i * every for i in range(n_applications)]
    return pd.DataFrame({'year': years, 'rate_t_ha': [per_application] * n_applications})


def write_schedule(records, path, comment=None):
    """Write an application schedule where `byte_util.feedstock.load_schedule` can read it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for line in (comment or '').splitlines():
        lines.append(f'# {line}')
    lines.append('year,rate_t_ha')
    for _, row in records.iterrows():
        lines.append(f'{int(row["year"])},{row["rate_t_ha"]:g}')

    path.write_text('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------------------------------
# Run tree
# ---------------------------------------------------------------------------

def _default_output_times(days_per_year, interval_days=30):
    times = [float(t) for t in range(interval_days, int(days_per_year) + 1, interval_days)]
    if times and times[-1] < days_per_year:
        times.append(float(days_per_year))
    return times


def build_run_tree(config, run_dir, met_forcing=None, start=None, output_times=None,
                   maximum_time_step=1.0, seed=False, spinup_ssa_m2_g=None,
                   cec_file=None, overwrite=False, year_prefix='year',
                   deck_database_dir=None):
    """Write every year directory for one chained run.

    Builds the ``.dat`` decks and transient forcing for all segments, and
    optionally seeds year one from the configured spin-up. Decks for later years
    are written now rather than during the chain, because each one is fully
    determined before any of them has run -- only the *restart data* has to wait.

    Parameters
    ----------
    config : byte_util.RunConfig
        Validated configuration for this run.
    run_dir : str or pathlib.Path
        Directory to build into. Must not already contain files unless
        ``overwrite``; nothing is ever deleted.
    met_forcing : pandas.DataFrame, optional
        Hourly forcing for this site. Required unless every segment is steady
        and ``start`` is None.
    start : str or pandas.Timestamp, optional
        Start of the first segment. Required when ``met_forcing`` is given.
    output_times : sequence of float, optional
        Spatial output times within each segment. Defaults to every 30 days.
    maximum_time_step : float
        Maximum time step for every segment.
    seed : bool
        Whether to seed year one from ``config.spinup_dir``. Needs the spin-up's
        output files, which are gitignored -- leave False to build decks and
        forcing only.
    spinup_ssa_m2_g : dict, optional
        Specific surface areas for rebuilding surface area at the seed. Required
        when ``seed``.
    cec_file : str or pathlib.Path, optional
        ``.cec`` to install for year one. A spin-up has none of its own.
    overwrite : bool
        Allow writing into a directory that already contains files.
    year_prefix : str
        Prefix for each segment's directory and run name.
    deck_database_dir : str, optional
        What to write as the deck's database directory. This is a *deployment*
        parameter (rather than build-time): the tree is often staged somewhere
        unrelated to the databases and then fetched next to a repo clone, so
        where they sit relative to each other at build time says nothing about
        run time. Defaults to the path relative to ``run_dir``, which is right
        when the tree is built in place, and raises rather than emitting an
        unusable ``../../../..`` chain when it is not.

    Returns
    -------
    dict
        Maps segment name to the paths written for it.
    """
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f'{run_dir} already contains files; pass overwrite=True to add to it. '
            'Nothing is deleted either way -- clear it yourself if that is what you want.'
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if output_times is None:
        output_times = _default_output_times(config.days_per_year)

    base_deck = config.resolved_base_deck
    windows = config.year_windows(start) if start is not None else [(None, None)] * config.n_years

    written = {}
    for index, (window_start, window_end) in enumerate(windows, start=1):
        prefix = f'{year_prefix}{index:02d}'
        year_dir = run_dir / prefix
        year_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        flux = transpiration = None
        if met_forcing is not None and window_start is not None:
            bcvs, soi, flux, transpiration = year_forcing(
                met_forcing, window_start, window_end,
                freq=config.resample_freq, steady_mean=config.steady_mean,
            )
            paths.update(write_year_forcing(bcvs, soi, year_dir, prefix))

        deck = InputFile.load(base_deck.name, path=base_deck.parent)
        set_problem_title(deck, f'{config.run_id} {prefix}')
        set_run_window(deck, final_time=config.days_per_year, output_times=output_times,
                       maximum_time_step=maximum_time_step)
        set_surface_forcing(deck, flux_m_s=flux, transpiration_factor=transpiration)
        if config.is_transient:
            enable_transient_forcing(deck)
        restart_from_files(deck)
        collapse_initial_chemistry_zones(deck)
        free_concentration_constraints(deck)
        set_transient_output(
            deck, interval=config.transient_output_interval,
            monitor_depths_cm=(config.transient_monitor_depths_cm
                               or DEFAULT_TRANSIENT_MONITOR_DEPTHS_CM),
            control_planes_cm=(config.transient_control_planes_cm
                               or DEFAULT_TRANSIENT_CONTROL_PLANES_CM),
        )
        if deck_database_dir is None:
            try:
                database_ref = relative_database_dir(year_dir, config.database_dir)
            except ValueError as err:
                raise ValueError(
                    f'{err} Pass deck_database_dir explicitly -- it is the path the '
                    'databases will have *from a run directory at run time*, which is '
                    'not knowable from where the tree is being staged.'
                ) from None
        else:
            database_ref = deck_database_dir
        set_database_dir(deck, database_ref)
        deck.save(year_dir / f'{prefix}.dat')
        paths['dat'] = year_dir / f'{prefix}.dat'
        paths['root'] = write_root_dat(year_dir, prefix)

        written[prefix] = paths

    if seed:
        if spinup_ssa_m2_g is None:
            raise ValueError('spinup_ssa_m2_g is required when seed=True')
        if config.spinup_dir is None:
            raise ValueError('config.spinup_dir is required when seed=True')

        first = f'{year_prefix}01'
        grid = _grid_elevations(base_deck)
        rates = config.application_rates()
        props = config.properties_by_rate().get(float(rates[0])) if rates[0] > 0 else None

        written[first].update(seed_run_state(
            config.spinup_dir, config.spinup_prefix, run_dir / first, first,
            minerals=config.minerals(), ssa_m2_g=spinup_ssa_m2_g,
            mineral_db=config.mineral_db, props=props,
            profile=config.depth_profile(grid) if props is not None else None,
            mix_mask=config.till_mask(grid), mix_rule=config.mix_rule,
            cec_file=cec_file, phi_floor=config.phi_floor,
            tracer_mode=config.tracer_mode, tracers=config.tracers, z=grid,
        ))

    return written


def _grid_elevations(deck_path):
    """Cell-centre elevations of a deck's z grid."""
    deck = InputFile.load(Path(deck_path).name, path=Path(deck_path).parent)
    sd = deck.spatial_discretization
    nz = int(sd.number_of_control_volumes_z)
    zmin, zmax = sd.z_bounds
    return np.linspace(zmin, zmax, nz)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def write_provenance(config, dest, git_shas=None, sources=None, extra=None,
                     copy_databases=True):
    """Record what a run was built from, next to where its results will land.

    Inputs that can change or cannot be reconstructed are copied rather than
    referenced: the thermodynamic databases above all, since ``mineral.dbs``
    evolves and a run is uninterpretable without the one it used. Anything too
    large to copy is recorded by URI in the manifest instead.

    ``run_config.json`` holds the *resolved* configuration, not the batch row.
    A row plus code at a pinned commit reproduces the derived values -- the
    transient interval, the steady mean, the spin-up prefix, the feedstock's
    bulk density -- but writing them means they can be read directly.

    Parameters
    ----------
    config : byte_util.RunConfig
        The run being described.
    dest : str or pathlib.Path
        Directory to write into, normally ``<output_dir>/provenance``.
    git_shas : dict, optional
        Repository name -> commit, e.g. ``{'min3p-model-bytes': ..., 'python-min3p': ...}``.
    sources : dict, optional
        Label -> URI for inputs too large to copy, such as the met forcing.
    extra : dict, optional
        Anything else worth recording, e.g. the MIN3P binary's hash.
    copy_databases : bool
        Whether to copy the database directory (about 430 KB).

    Returns
    -------
    dict
        Maps a short key to each path written.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = {}

    if copy_databases and config.database_dir.is_dir():
        target = dest / 'databases'
        target.mkdir(exist_ok=True)
        for source in sorted(config.database_dir.glob('*.dbs')):
            shutil.copy(source, target / source.name)
        written['databases'] = target

    for label, source in (('feedstock', config.feedstock), ('schedule', config.schedule)):
        if source.exists():
            target = dest / f'{label}{source.suffix}'
            shutil.copy(source, target)
            written[label] = target

    config_path = dest / 'run_config.json'
    config_path.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + '\n')
    written['run_config'] = config_path

    manifest = {
        'run_id': config.run_id,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'git_shas': dict(git_shas or {}),
        'sources': dict(sources or {}),
    }
    manifest.update(extra or {})
    manifest_path = dest / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    written['manifest'] = manifest_path

    return written
