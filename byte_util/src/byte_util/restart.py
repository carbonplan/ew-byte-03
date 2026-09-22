"""Turn a finished MIN3P year into the input files for the next one.

A chained run restarts from four things:

    *.ivs    flow state        <- verbatim copy of the last *.gsp
    *.aqt    aqueous totals    <- last *.gst, optionally with tracers reset
    *.min    mineral volume fractions   } see byte_util.feedstock,
    *.surf   mineral surface areas      } which handles the SA law

plus ``*.cec`` and ``*.rld``, which are time-invariant and are simply
carried forward. Everything else MIN3P needs is derived rather than carried. 

Tracers
-------
``psi01`` and ``psi02`` are conservative tracers in these decks: ``comp.dbs``
gives them zero charge, zero ion size and zero molar mass, and they aren't
referenced elsewhere.

Note: In ``sorption.dbs`` the ``psi<plane><surface>`` names are the electrostatic 
potential unknowns of MIN3P's triple-layer surface-complexation model. These decks
borrow the unused slots as tracers, which is fine as long as no surface
complexation is switched on. **If surface complexation is ever activated, these
names appear to stop being tracers.**

`write_aqueous_state` therefore offers three modes:

``'carry'``
    Pass the tracers through like any other component. The default, because
    we want to avoid a restart rewriting a state unknowingly.
``'reinitialise'``
    Reset each tracer to its background value and re-inject an elevated band,
    reproducing what ``MIN3P_BuildMetForcingERWSimulations.ipynb`` does at the
    spinup boundary. 
``'drop'``
    Remove the tracer columns entirely. Requires a matching edit to the deck's
    ``'components'`` block -- see `write_aqueous_state`.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from min3p.output import read_min3p, read_min3p_sequence, write_min3p

from .feedstock import write_restart_minerals

__all__ = [
    'MET_FORCING_TRACERS',
    'TRACER_MODES',
    'TracerBand',
    'copy_flow_state',
    'copy_static_files',
    'depth_band_mask',
    'last_sequence_file',
    'seed_run_state',
    'write_aqueous_state',
    'write_restart_state',
]

# Modes `write_aqueous_state` accepts for handling tracer components.
TRACER_MODES = ('carry', 'reinitialise', 'drop')

# Files that do not change between years and are carried forward untouched.
STATIC_EXTENSIONS = ('cec', 'rld')


@dataclass(frozen=True)
class TracerBand:
    """A tracer and the depth band it is re-injected into.

    Parameters
    ----------
    component : str
        Component name, as it appears in the deck's ``'components'`` block.
    background : float
        Concentration written everywhere outside the band.
    value : float
        Concentration written inside the band.
    top_m : float
        Depth below ground at which the band starts.
    bottom_m : float
        Depth below ground at which the band ends.
    """

    component: str
    background: float = 1.0e-3
    value: float = 1.0e-1
    top_m: float = 0.0
    bottom_m: float = 0.3


# The tracer layout used by the met_forcing decks: a shallow tracer over the
# top 30 cm and a deeper one from 30 cm to 1 m. Reproduces the initialisation
# in ``MIN3P_BuildMetForcingERWSimulations.ipynb``.
MET_FORCING_TRACERS = (
    TracerBand('psi01', top_m=0.0, bottom_m=0.3),
    TracerBand('psi02', top_m=0.3, bottom_m=1.0),
)


def depth_band_mask(z, top_m, bottom_m):
    """Boolean mask for a depth band, counted in cells from the ground surface.

    Counted rather than thresholded for the same reason as
    `byte_util.feedstock.till_layer_mask`.

    Parameters
    ----------
    z : array_like
        Cell-center "elevations" (m), ascending. The ground surface is the last
        element.
    top_m, bottom_m : float
        Depths below ground bounding the band. ``top_m`` is inclusive,
        ``bottom_m`` exclusive.

    Returns
    -------
    numpy.ndarray of bool
    """
    z = np.asarray(z, dtype=float)
    if bottom_m <= top_m:
        raise ValueError('bottom_m must be deeper than top_m')

    dz = np.diff(z).mean()
    n_top = int(round(top_m / dz))
    n_bottom = int(round(bottom_m / dz))

    idx = np.arange(len(z))[::-1]  # 0 at the ground surface, increasing downward
    return (idx >= n_top) & (idx < n_bottom)


def last_sequence_file(folder, prefix, extension):
    """Path to the highest-numbered file in a MIN3P output sequence.

    MIN3P writes one contour file per spatial output time, numbered
    ``prefix_1.ext``, ``prefix_2.ext``, ... so the last one is the final state.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory holding the run's output.
    prefix : str
        Run prefix, as given in ``root.dat``.
    extension : str
        Output extension without the dot, e.g. ``'gsp'``.

    Returns
    -------
    pathlib.Path
    """
    folder = Path(folder)
    files = list(folder.glob(f'{prefix}_*.{extension}'))
    if not files:
        raise FileNotFoundError(f'no {prefix}_*.{extension} files in {folder}')

    def index(path):
        return int(path.stem.rsplit('_', 1)[-1])

    return max(files, key=index)


def copy_flow_state(prev_dir, prev_prefix, next_dir, next_prefix):
    """Write the next year's ``.ivs`` from the previous year's final ``.gsp``.

    A verbatim copy -- MIN3P's flow output and its variably-saturated initial
    condition file are the same format, so nothing is transformed. The copied
    file keeps the previous run's title in its header, which is expected.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    source = last_sequence_file(prev_dir, prev_prefix, 'gsp')
    target = Path(next_dir) / f'{next_prefix}.ivs'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    return target


def write_aqueous_state(prev_dir, prev_prefix, next_dir, next_prefix,
                        tracer_mode='carry', tracers=MET_FORCING_TRACERS, z=None,
                        return_arrays=False):
    """Write the next year's ``.aqt`` from the previous year's final ``.gst``.

    Parameters
    ----------
    prev_dir : str or pathlib.Path
        Directory holding the previous year's run.
    prev_prefix : str
        Run prefix for the previous year.
    next_dir : str or pathlib.Path
        Directory to write into. Created if absent.
    next_prefix : str
        Run prefix for the next year.
    tracer_mode : {'carry', 'reinitialise', 'drop'}
        How to treat the tracer components. See the module docstring.
    tracers : sequence of TracerBand
        Tracer layout, used by ``'reinitialise'`` and ``'drop'``.
    z : array_like, optional
        Cell-centre elevations. Required for ``'reinitialise'``; taken from the
        output file's own ``z`` column when omitted.
    return_arrays : bool
        If True, also return the written array and its column list.

    Returns
    -------
    pathlib.Path, or tuple
        The file written, or ``(path, array, columns)`` when ``return_arrays``.

    Notes
    -----
    ``'drop'`` removes the tracer columns from the file. The deck must drop the same
    components or the read will be misaligned. 
    """
    if tracer_mode not in TRACER_MODES:
        raise ValueError(f'tracer_mode must be one of {TRACER_MODES}, got {tracer_mode!r}')

    gst, columns, _ = read_min3p_sequence(f'{prev_prefix}_1.gst', folder=Path(prev_dir))
    aqt = np.asarray(gst[-1], dtype=float).copy()
    columns = list(columns)

    if tracer_mode == 'reinitialise':
        grid = np.asarray(z) if z is not None else aqt[columns.index('z')]
        for band in tracers:
            if band.component not in columns:
                raise KeyError(
                    f'tracer {band.component!r} is not a component of {prev_prefix}'
                )
            row = columns.index(band.component)
            aqt[row, :] = band.background
            aqt[row, depth_band_mask(grid, band.top_m, band.bottom_m)] = band.value

    elif tracer_mode == 'drop':
        drop = {band.component for band in tracers}
        keep = [i for i, name in enumerate(columns) if name not in drop]
        aqt = aqt[keep]
        columns = [columns[i] for i in keep]

    next_dir = Path(next_dir)
    next_dir.mkdir(parents=True, exist_ok=True)
    target = next_dir / f'{next_prefix}.aqt'
    write_min3p(aqt, target.name, columns, folder=next_dir,
                prefix=next_prefix, label='initial aqueous concentrations')

    if return_arrays:
        return target, aqt, columns
    return target


def copy_static_files(prev_dir, prev_prefix, next_dir, next_prefix,
                      extensions=STATIC_EXTENSIONS, missing_ok=False):
    """Carry time-invariant input files forward to the next year.

    ``.cec`` (cation exchange capacity and bulk density) and ``.rld`` (root
    length density) are properties of the soil profile, not of the simulation
    state, so they are copied rather than rebuilt.

    Parameters
    ----------
    prev_dir, next_dir : str or pathlib.Path
        Run directories.
    prev_prefix, next_prefix : str
        Run prefixes.
    extensions : sequence of str
        Extensions to carry forward, without the dot.
    missing_ok : bool
        When True, a missing source file is skipped instead of raising.
        Defaults to False: a chain that silently drops a ``.cec`` produces a
        run with the wrong exchange capacity, which is not something to
        discover from the results.

    Returns
    -------
    list of pathlib.Path
        The files written.
    """
    prev_dir, next_dir = Path(prev_dir), Path(next_dir)
    next_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for extension in extensions:
        source = prev_dir / f'{prev_prefix}.{extension}'
        if not source.exists():
            if missing_ok:
                continue
            raise FileNotFoundError(f'{source} not found')
        target = next_dir / f'{next_prefix}.{extension}'
        shutil.copy(source, target)
        written.append(target)

    return written


def write_root_dat(run_dir, prefix):
    """Write the ``root.dat`` that tells MIN3P which prefix to run.

    Returns
    -------
    pathlib.Path
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / 'root.dat'
    target.write_text(f'{prefix}\n')
    return target


def seed_run_state(spinup_dir, spinup_prefix, next_dir, next_prefix, minerals,
                   ssa_m2_g, mineral_db, props=None, profile=None, mix_mask=None,
                   mix_rule='homogeneous', cec_file=None, phi_floor=None,
                   tracer_mode='carry', tracers=MET_FORCING_TRACERS, z=None,
                   static_extensions=('rld',), write_root=True):
    """Seed the first year of a chain from a spin-up.

    The first hop is not a restart, because a spin-up directory has a different
    shape. Its deck sets only ``'read root length density field from file'`` and
    ``'read initial mineral areas from file'``, taking mineral volume fractions
    from its ``'mineral input'`` blocks and exchange capacity from
    ``'sorption parameter input'`` -- so there is no ``prefix.min`` and no
    ``prefix.cec`` to read or carry forward.

    The two consequences:

    * Surface area is recomputed from volume fraction rather than advanced by
      the 2/3 law. That is *correct* here rather than a shortcut: feedstock
      minerals sit at ``phi_floor`` throughout a spin-up, so there is no aged
      cohort whose history could be lost. The 2/3 law only starts to matter from
      the second hop, once feedstock has actually dissolved.
    * ``.cec`` has to be supplied rather than copied.

    Parameters
    ----------
    spinup_dir : str or pathlib.Path
        Directory holding the finished spin-up.
    spinup_prefix : str
        Its run prefix.
    next_dir : str or pathlib.Path
        Directory to write year one into.
    next_prefix : str
        Year one's run prefix.
    minerals : sequence of str
        Feedstock minerals.
    ssa_m2_g : dict
        Specific surface area (m2/g) per mineral, used to rebuild bulk surface
        areas from volume fractions. Specific surface area is a modelling choice
        rather than a database property, so it is supplied rather than derived.
        Minerals absent from this mapping keep their spin-up column unchanged,
        which is what ``co2_resp`` needs -- its ``.surf`` column carries a
        depth-varying rate constant, not an area.
    mineral_db : dict
        Output of `byte_util.feedstock.read_mineral_database`, for densities.
    props : pandas.DataFrame, optional
        Feedstock properties for a year-one application.
    profile : numpy.ndarray, optional
        Depth weights; required when ``props`` is given.
    mix_mask : numpy.ndarray of bool, optional
        Plough-layer cells.
    mix_rule : str
        Key into `byte_util.feedstock.MIX_RULES`.
    cec_file : str or pathlib.Path, optional
        A ``.cec`` file to copy in. Generate it with
        `byte_util.reaction.write_initial_cec` when there isn't one.
    phi_floor : float, optional
        Minimum volume fraction.
    tracer_mode : {'carry', 'reinitialise', 'drop'}
        How to treat tracer components.
    tracers : sequence of TracerBand
        Tracer layout.
    z : array_like, optional
        Cell-centre elevations, required for ``'reinitialise'``.
    static_extensions : sequence of str
        Time-invariant files to carry forward. ``.rld`` only by default, since a
        spin-up has no ``.cec``.
    write_root : bool
        Whether to write ``root.dat``.

    Returns
    -------
    dict
        Maps a short key to each path written.
    """
    from .feedstock import PHI_FLOOR, apply_feedstock

    spinup_dir, next_dir = Path(spinup_dir), Path(next_dir)
    next_dir.mkdir(parents=True, exist_ok=True)
    phi_floor = PHI_FLOOR if phi_floor is None else phi_floor

    written = {}
    written['ivs'] = copy_flow_state(spinup_dir, spinup_prefix, next_dir, next_prefix)
    written['aqt'] = write_aqueous_state(spinup_dir, spinup_prefix, next_dir, next_prefix,
                                         tracer_mode=tracer_mode, tracers=tracers, z=z)

    # Final mineral volume fractions, and the spin-up's surface areas for reference
    gsv, phi_columns, _ = read_min3p_sequence(f'{spinup_prefix}_1.gsv', folder=spinup_dir)
    phi = np.asarray(gsv[-1], dtype=float).copy()
    phi_columns = list(phi_columns)

    bsa_spin, bsa_columns, _ = read_min3p(f'{spinup_prefix}.surf', folder=spinup_dir)
    bsa = np.asarray(bsa_spin, dtype=float).copy()
    bsa_columns = list(bsa_columns)

    # Rebuild bulk surface area from the final volume fractions. Minerals with no
    # SSA supplied (co2_resp) keep the spin-up column untouched.
    for mineral, ssa in ssa_m2_g.items():
        if mineral not in bsa_columns or mineral not in phi_columns:
            continue
        density = mineral_db[mineral]['density_g_cm3']
        bsa[bsa_columns.index(mineral)] = (
            ssa * density * 1e3 * phi[phi_columns.index(mineral)]
        )

    if props is not None:
        if profile is None:
            raise ValueError('profile is required when props is given')
        phi, bsa = apply_feedstock(phi, bsa, phi_columns, props, profile,
                                   mix_mask=mix_mask, bsa_columns=bsa_columns,
                                   mix_rule=mix_rule, phi_floor=phi_floor)

    write_min3p(phi, f'{next_prefix}.min', phi_columns, folder=next_dir,
                prefix=next_prefix, label='phi_i, T = initial')
    write_min3p(bsa, f'{next_prefix}.surf', bsa_columns, folder=next_dir,
                prefix=next_prefix, label='bsa_i (m2/L), T = initial')
    written['min'] = next_dir / f'{next_prefix}.min'
    written['surf'] = next_dir / f'{next_prefix}.surf'

    for path in copy_static_files(spinup_dir, spinup_prefix, next_dir, next_prefix,
                                  extensions=static_extensions):
        written[path.suffix.lstrip('.')] = path

    if cec_file is not None:
        target = next_dir / f'{next_prefix}.cec'
        shutil.copy(cec_file, target)
        written['cec'] = target

    if write_root:
        written['root'] = write_root_dat(next_dir, next_prefix)

    return written


def write_restart_state(prev_dir, prev_prefix, next_dir, next_prefix, minerals,
                        props=None, profile=None, mix_mask=None, mix_rule='homogeneous',
                        till_without_application=False, exponent=None, phi_floor=None,
                        tracer_mode='carry', tracers=MET_FORCING_TRACERS, z=None,
                        static_extensions=STATIC_EXTENSIONS, write_root=True):
    """Write every input file the next year of a chain needs.

    Composes `copy_flow_state`, `write_aqueous_state`,
    `byte_util.feedstock.write_restart_minerals` and `copy_static_files`.

    This does *not* write the ``.dat`` deck or the transient forcing files
    (``.bcvs``, ``.soi``) -- those are built once per batch rather than per
    restart, since each year's deck is known before any of them have run.

    Parameters
    ----------
    prev_dir, next_dir : str or pathlib.Path
        Run directories for the previous and next year.
    prev_prefix, next_prefix : str
        Run prefixes, as given in each ``root.dat``.
    minerals : sequence of str
        Feedstock minerals tracked across the restart.
    props : pandas.DataFrame, optional
        Output of `byte_util.feedstock.feedstock_properties`. Omit for a year
        with no application.
    profile : numpy.ndarray, optional
        Depth weights; required when ``props`` is given.
    mix_mask : numpy.ndarray of bool, optional
        Plough-layer cells, from `byte_util.feedstock.till_layer_mask`.
    mix_rule : str
        Key into `byte_util.feedstock.MIX_RULES`.
    till_without_application : bool
        Whether to mix the plough layer in a year with no application.
    exponent : float, optional
        Surface-area scaling exponent; defaults to the feedstock module's.
    phi_floor : float, optional
        Minimum volume fraction; defaults to the feedstock module's.
    tracer_mode : {'carry', 'reinitialise', 'drop'}
        How to treat tracer components.
    tracers : sequence of TracerBand
        Tracer layout for ``'reinitialise'`` and ``'drop'``.
    z : array_like, optional
        Cell-centre elevations, required for ``tracer_mode='reinitialise'``.
    static_extensions : sequence of str
        Time-invariant files to carry forward.
    write_root : bool
        Whether to write ``root.dat`` in the new directory.

    Returns
    -------
    dict
        Maps a short key (``'ivs'``, ``'aqt'``, ``'min'``, ``'surf'``,
        ``'root'``, plus each static extension) to the path written.
    """
    prev_dir, next_dir = Path(prev_dir), Path(next_dir)
    next_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    written['ivs'] = copy_flow_state(prev_dir, prev_prefix, next_dir, next_prefix)
    written['aqt'] = write_aqueous_state(prev_dir, prev_prefix, next_dir, next_prefix,
                                         tracer_mode=tracer_mode, tracers=tracers, z=z)

    mineral_kwargs = {}
    if exponent is not None:
        mineral_kwargs['exponent'] = exponent
    if phi_floor is not None:
        mineral_kwargs['phi_floor'] = phi_floor

    write_restart_minerals(
        prev_dir, prev_prefix, next_dir, next_prefix, minerals,
        props=props, profile=profile, mix_mask=mix_mask, mix_rule=mix_rule,
        till_without_application=till_without_application, **mineral_kwargs,
    )
    written['min'] = next_dir / f'{next_prefix}.min'
    written['surf'] = next_dir / f'{next_prefix}.surf'

    for path in copy_static_files(prev_dir, prev_prefix, next_dir, next_prefix,
                                  extensions=static_extensions):
        written[path.suffix.lstrip('.')] = path

    if write_root:
        written['root'] = write_root_dat(next_dir, next_prefix)

    return written
