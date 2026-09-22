"""Typed configuration for a single chained MIN3P simulation.

`RunConfig` sits between a batch-CSV row and the functions in
`byte_util.feedstock`. **DEFAULTS LIVE IN RunConfig**, and it's the one place 
strings from a CSV get coerced to real types, and the one place a run is 
validated -- so a bad configuration fails when the batch is generated.

The same object is built by a Coiled worker from its task dict and by hand in a
notebook to debug a single site, so there is one code path rather than two:

    csv row (dict of str)
        -> RunConfig.from_row()      # defaults, coercion, path resolution
            -> RunConfig.validate()  # enums, ranges, files, deck minerals
                -> byte_util.feedstock.*   # pure functions on plain arrays

Configuration here is scientific plus input-file locations. Orchestration
concerns belong to batch driver so coiled isn't baked in.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from min3p.input import InputFile

from .deck import DEFAULT_DAYS_PER_YEAR, horizon_zones, year_windows
from .feedstock import (
    DEFAULT_CONST_RF,
    DEFAULT_MIX_RULE,
    DEFAULT_ROUGHNESS_OPTION,
    MIX_RULES,
    PHI_FLOOR,
    SA_EXPONENT,
    depth_profile,
    load_feedstock,
    load_schedule,
    props_for_schedule,
    read_mineral_database,
    till_layer_mask,
)
from .restart import MET_FORCING_TRACERS, TRACER_MODES, write_restart_state

__all__ = ['DEFAULT_TRANSIENT_INTERVAL', 'FORCING_FREQUENCIES', 'FORCING_STEADY_MEANS',
           'FORCING_TRANSIENT_INTERVALS', 'FORCING_TYPES', 'RunConfig']

# Forcing type -> pandas resampling frequency, in order of increasing temporal
# resolution. ``None`` means steady forcing: no transient files, just a constant
# in the deck. The three transient values match `all_frequencies` in
# ``MIN3P_CreateMetForcingBCFiles.ipynb``, which is where they came from.
FORCING_FREQUENCIES = {
    'longterm': None,
    'annual': None,
    'monthly': 'MS',
    'daily': 'D',
    'hourly': 'h',
}

# Which mean the two steady forcing types use -- the only thing separating them.
# ``longterm`` holds one value across the whole chain; ``annual`` gives each
# segment its own mean, keeping interannual variability.
FORCING_STEADY_MEANS = {'longterm': 'record', 'annual': 'window'}

# Forcing types the decks support. Derived, so there is one list to maintain.
FORCING_TYPES = tuple(FORCING_FREQUENCIES)

# Timesteps between transient output records, by forcing type. Hourly forcing
# drives small timesteps, so recording every one is the dominant term in output
# size; everything coarser is cheap enough to keep at full resolution.
FORCING_TRANSIENT_INTERVALS = {'hourly': 24}
DEFAULT_TRANSIENT_INTERVAL = 1

_TRUE = {'true', 't', 'yes', 'y', '1'}
_FALSE = {'false', 'f', 'no', 'n', '0', ''}


def _as_bool(value):
    """Coerce a CSV cell to bool, since everything arrives as a string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f'cannot interpret {value!r} as a boolean')


def _as_depths(value):
    """Coerce a CSV cell to a tuple of depths; accepts commas or whitespace."""
    if value is None or isinstance(value, float):
        return None
    if not isinstance(value, str):
        return tuple(float(v) for v in value)
    text = value.strip()
    if not text:
        return None
    return tuple(float(v) for v in text.replace(',', ' ').split())


def _as_roughness(value):
    """Roughness option is either a rule name or a bare number."""
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text


@dataclass
class RunConfig:
    """Everything needed to build and chain one MIN3P simulation.

    Parameters
    ----------
    run_id : str
        Unique identifier; names the output directory tree.
    site : str
        Soil series name, keying the site parameter tables.
    feedstock : pathlib.Path
        Feedstock composition file (see `byte_util.feedstock.load_feedstock`).
    schedule : pathlib.Path
        Application schedule file (see `byte_util.feedstock.load_schedule`).
    database_dir : pathlib.Path
        Directory holding the MIN3P databases, including ``mineral.dbs``.
    forcing : str
        One of ``FORCING_TYPES``. ``'longterm'`` and ``'annual'`` are both steady
        -- they write no transient files and differ only in which mean goes into
        the deck; see `FORCING_STEADY_MEANS`.
    n_years : int
        Number of chained one-year segments.
    days_per_year : float
        Length of each segment. Segments are equal length rather than calendar
        years; ten 365-day segments total 3650 days, matching the existing
        unchained runs.
    spinup_dir : pathlib.Path, optional
        Finished spin-up this chain seeds from. Its prefix is read from
        ``root.dat`` rather than configured.
    base_deck : pathlib.Path, optional
        Template ``.dat`` for the year decks. Defaults to the spin-up's own deck,
        which is where the ERW pipeline takes it from and which already carries
        that site's soil physics -- porosity, Ksat, van Genuchten parameters and
        horizon count all vary by site.
    diameter_um : float
        Default grain diameter, used for any mineral without its own.
    till_depth_m : float
        Depth feedstock is mixed into, and the depth tillage acts over.
    roughness_option : str or float
        Roughness rule name or a bare factor.
    const_rf : float
        Roughness factor used when ``roughness_option='constant'``.
    depth_distribution : str
        ``'uniform'`` or ``'sigmoidal'``.
    sigmoidal_steepness : float
        Steepness for the sigmoidal distribution.
    mix_rule : str
        Key into `byte_util.feedstock.MIX_RULES`.
    till_without_application : bool
        Whether the plough layer is mixed in years with no application.
    tracer_mode : str
        One of ``'carry'``, ``'reinitialise'`` or ``'drop'`` -- see
        `byte_util.restart`. Defaults to ``'carry'``, because a restart that
        quietly rewrites state is a poor default; set ``'reinitialise'``
        explicitly to use the tracers as an annual water-age diagnostic.
    transient_output_interval : int, optional
        Timesteps between transient output records. Resolved from
        `FORCING_TRANSIENT_INTERVALS` when omitted, so hourly runs thin their
        output and coarser ones do not.
    transient_monitor_depths_cm : tuple, optional
        Depths (cm) recorded as single points. ``None`` leaves the deck's own
        locations alone.
    transient_control_planes_cm : tuple, optional
        Depths (cm) recorded as straddling pairs, for flux across the plane.
        ``None`` leaves the deck's own locations alone.
    tracers : tuple of byte_util.restart.TracerBand
        Tracer layout, used by ``'reinitialise'`` and ``'drop'``. Defaults to
        the met_forcing decks' shallow/deep pair, so a batch CSV only needs the
        ``tracer_mode`` column.
    feedstock_density_g_cm3 : float, optional
        Override for the bulk feedstock density; derived from composition when
        omitted.
    sa_exponent : float
        Surface-area scaling exponent for the aged cohort.
    phi_floor : float
        Minimum volume fraction written for a feedstock mineral.
    """

    run_id: str
    site: str
    feedstock: Path
    schedule: Path
    database_dir: Path

    forcing: str = 'longterm'
    n_years: int = 10
    days_per_year: float = DEFAULT_DAYS_PER_YEAR

    spinup_dir: Path | None = None
    base_deck: Path | None = None

    diameter_um: float = 50.0
    till_depth_m: float = 0.3
    roughness_option: str | float = DEFAULT_ROUGHNESS_OPTION
    const_rf: float = DEFAULT_CONST_RF

    depth_distribution: str = 'uniform'
    sigmoidal_steepness: float = 0.4

    mix_rule: str = DEFAULT_MIX_RULE
    till_without_application: bool = False

    tracer_mode: str = 'carry'
    tracers: tuple = MET_FORCING_TRACERS

    transient_output_interval: int | None = None
    transient_monitor_depths_cm: tuple | None = None
    transient_control_planes_cm: tuple | None = None

    feedstock_density_g_cm3: float | None = None
    sa_exponent: float = SA_EXPONENT
    phi_floor: float = PHI_FLOOR

    _mineral_db: dict = field(default=None, repr=False, compare=False)
    _spinup_prefix: str = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Resolve the transient interval so the field always holds the effective value."""
        if self.transient_output_interval is None:
            self.transient_output_interval = FORCING_TRANSIENT_INTERVALS.get(
                self.forcing, DEFAULT_TRANSIENT_INTERVAL)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_row(cls, row, root=None, **overrides):
        """Build a config from a batch-CSV row.

        Unknown columns are ignored, so a batch CSV may carry bookkeeping fields
        (row numbers, S3 destinations, notes) that are not part of a run's
        scientific definition.

        Parameters
        ----------
        row : dict
            One CSV row. Values may be strings; they are coerced here.
        root : str or pathlib.Path, optional
            Base directory that relative paths resolve against.
        **overrides
            Applied after the row, for one-off tweaks in a notebook.

        Returns
        -------
        RunConfig
        """
        known = {f.name for f in fields(cls) if not f.name.startswith('_')}
        merged = {k: v for k, v in dict(row).items() if k in known}
        merged.update(overrides)

        missing = {'run_id', 'site', 'feedstock', 'schedule', 'database_dir'} - set(merged)
        if missing:
            raise ValueError(f'row is missing required field(s): {sorted(missing)}')

        coercions = {
            'n_years': int,
            'days_per_year': float,
            'diameter_um': float,
            'till_depth_m': float,
            'const_rf': float,
            'sigmoidal_steepness': float,
            'sa_exponent': float,
            'phi_floor': float,
            'till_without_application': _as_bool,
            'roughness_option': _as_roughness,
            'transient_output_interval': int,
            'transient_monitor_depths_cm': _as_depths,
            'transient_control_planes_cm': _as_depths,
        }
        for key, cast in coercions.items():
            if key in merged and merged[key] is not None and merged[key] != '':
                merged[key] = cast(merged[key])

        if merged.get('feedstock_density_g_cm3') in ('', None):
            merged.pop('feedstock_density_g_cm3', None)
        elif 'feedstock_density_g_cm3' in merged:
            merged['feedstock_density_g_cm3'] = float(merged['feedstock_density_g_cm3'])

        base = Path(root) if root is not None else None

        def resolve(value):
            path = Path(value)
            return base / path if base is not None and not path.is_absolute() else path

        for key in ('feedstock', 'schedule', 'database_dir'):
            merged[key] = resolve(merged[key])
        for key in ('spinup_dir', 'base_deck'):
            if merged.get(key) in ('', None):
                merged.pop(key, None)
            else:
                merged[key] = resolve(merged[key])

        return cls(**merged)

    # -- validation --------------------------------------------------------

    def validate(self, deck_minerals=None):
        """Check the configuration, raising on the first problem found.

        Parameters
        ----------
        deck_minerals : sequence of str, optional
            The ``'minerals'`` block of the ``.dat`` deck this run will use.
            Read from `base_deck` automatically when that is resolvable; pass it
            explicitly only to override. Every feedstock mineral is checked
            against it -- a feedstock mineral with no column in ``.min``/``.surf``
            has nowhere to receive volume fraction, so we catch that here.

        Returns
        -------
        RunConfig
            ``self``, so this can be chained onto construction.
        """
        if self.forcing not in FORCING_TYPES:
            raise ValueError(f'forcing must be one of {FORCING_TYPES}, got {self.forcing!r}')
        if self.mix_rule not in MIX_RULES:
            raise ValueError(f'mix_rule must be one of {sorted(MIX_RULES)}, got {self.mix_rule!r}')
        if self.depth_distribution not in ('uniform', 'sigmoidal'):
            raise ValueError(f'unknown depth_distribution: {self.depth_distribution!r}')
        if self.days_per_year <= 0:
            raise ValueError('days_per_year must be positive')
        if self.transient_output_interval < 1:
            raise ValueError('transient_output_interval must be at least 1')
        if self.tracer_mode not in TRACER_MODES:
            raise ValueError(
                f'tracer_mode must be one of {TRACER_MODES}, got {self.tracer_mode!r}')
        if self.n_years < 1:
            raise ValueError('n_years must be at least 1')
        if self.till_depth_m <= 0:
            raise ValueError('till_depth_m must be positive')
        if self.diameter_um <= 0:
            raise ValueError('diameter_um must be positive')

        for label, path in (('feedstock', self.feedstock),
                            ('schedule', self.schedule),
                            ('database_dir', self.database_dir)):
            if not path.exists():
                raise FileNotFoundError(f'{label} not found: {path}')

        mineral_dbs = self.database_dir / 'mineral.dbs'
        if not mineral_dbs.exists():
            raise FileNotFoundError(f'mineral.dbs not found in {self.database_dir}')

        if self.spinup_dir is not None and not self.spinup_dir.is_dir():
            raise FileNotFoundError(f'spinup_dir not found: {self.spinup_dir}')
        if self.base_deck is not None and not self.base_deck.exists():
            raise FileNotFoundError(f'base_deck not found: {self.base_deck}')

        # Exercises the parsers, so a malformed file fails here too
        feedstock, _ = self.feedstock_table()
        self.application_rates()

        # Read the deck's mineral list rather than relying on the caller to
        # remember to pass it -- this check is the whole point of validating.
        if deck_minerals is None and (self.spinup_dir is not None or self.base_deck is not None):
            deck_minerals = self.deck_minerals()

        if deck_minerals is not None:
            absent = sorted(set(feedstock['mineral']) - set(deck_minerals))
            if absent:
                raise ValueError(
                    f'feedstock minerals absent from the deck: {absent}. '
                    "Add them to the 'minerals' block before running."
                )

        return self

    # -- derived quantities ------------------------------------------------

    @property
    def spinup_prefix(self):
        """Run prefix of the seed spin-up, read from its ``root.dat`` (cached)."""
        if self.spinup_dir is None:
            raise ValueError('spinup_dir is not set')
        if self._spinup_prefix is None:
            root = self.spinup_dir / 'root.dat'
            if not root.exists():
                raise FileNotFoundError(f'no root.dat in {self.spinup_dir}')
            self._spinup_prefix = root.read_text().strip()
        return self._spinup_prefix

    @property
    def resolved_base_deck(self):
        """Template deck for this run's year decks."""
        if self.base_deck is not None:
            return self.base_deck
        if self.spinup_dir is None:
            raise ValueError('set either base_deck or spinup_dir')
        return self.spinup_dir / f'{self.spinup_prefix}.dat'

    def deck_minerals(self):
        """Mineral names declared in the base deck's geochemical system."""
        deck = self.resolved_base_deck
        infile = InputFile.load(deck.name, path=deck.parent)
        return list(infile.geochemical_system.minerals)

    def year_windows(self, start):
        """Half-open ``[start, end)`` windows for every segment of this chain."""
        return year_windows(start, self.n_years, days_per_year=self.days_per_year)

    @property
    def resample_freq(self):
        """Pandas resampling frequency for this run's forcing, or None if steady."""
        return FORCING_FREQUENCIES[self.forcing]

    @property
    def steady_mean(self):
        """Which mean a steady forcing type uses. Unused when `resample_freq` is set."""
        return FORCING_STEADY_MEANS.get(self.forcing, 'record')

    @property
    def is_transient(self):
        """Whether this run needs ``.bcvs``/``.soi`` files and the transient keywords."""
        return self.resample_freq is not None

    @property
    def mineral_db(self):
        """Mineral properties parsed from ``mineral.dbs`` (cached)."""
        if self._mineral_db is None:
            self._mineral_db = read_mineral_database(self.database_dir / 'mineral.dbs')
        return self._mineral_db

    def feedstock_table(self):
        """Return ``(feedstock DataFrame, bulk density)`` for this run."""
        return load_feedstock(self.feedstock, self.mineral_db,
                              default_diameter_um=self.diameter_um,
                              density_g_cm3=self.feedstock_density_g_cm3)

    def application_rates(self):
        """Per-year application rate (t/ha), length ``n_years``."""
        return load_schedule(self.schedule, self.n_years)

    def minerals(self):
        """Feedstock mineral names, in file order."""
        feedstock, _ = self.feedstock_table()
        return list(feedstock['mineral'])

    def properties_by_rate(self):
        """`feedstock_properties` for each distinct non-zero rate in the schedule."""
        feedstock, density = self.feedstock_table()
        return props_for_schedule(
            self.application_rates(), feedstock, density,
            till_depth_m=self.till_depth_m,
            roughness_option=self.roughness_option,
            const_rf=self.const_rf,
        )

    def depth_profile(self, z):
        """Relative depth weights for the grid ``z``."""
        return depth_profile(z, self.till_depth_m,
                             distribution=self.depth_distribution,
                             steepness=self.sigmoidal_steepness)

    def till_mask(self, z):
        """Boolean mask for the plough layer on the grid ``z``."""
        return till_layer_mask(z, self.till_depth_m)

    # -- the chain step ----------------------------------------------------

    def write_restart(self, prev_dir, prev_prefix, next_dir, next_prefix, year, z):
        """Write every input file that starts ``year`` of the chain.

        Assembles the feedstock properties, depth profile and plough-layer mask
        from this config and the schedule, then defers to
        `byte_util.restart.write_restart_state`.

        Parameters
        ----------
        prev_dir, next_dir : str or pathlib.Path
            Run directories for the previous and next year.
        prev_prefix, next_prefix : str
            Run prefixes, as given in each ``root.dat``.
        year : int
            1-indexed year being written, used to look up the schedule.
        z : array_like
            Cell-centre elevations of the grid.

        Returns
        -------
        dict
            Maps a short key to each path written.
        """
        if not 1 <= year <= self.n_years:
            raise ValueError(f'year {year} is outside a {self.n_years}-year run')

        rate = float(self.application_rates()[year - 1])
        props = self.properties_by_rate().get(rate) if rate > 0 else None

        return write_restart_state(
            prev_dir, prev_prefix, next_dir, next_prefix,
            minerals=self.minerals(),
            props=props,
            profile=self.depth_profile(z) if props is not None else None,
            mix_mask=self.till_mask(z),
            mix_rule=self.mix_rule,
            till_without_application=self.till_without_application,
            exponent=self.sa_exponent,
            phi_floor=self.phi_floor,
            tracer_mode=self.tracer_mode,
            tracers=self.tracers,
            z=z,
        )

    def as_dict(self):
        """The resolved configuration, as JSON-serialisable plain data.

        Includes the values `RunConfig` derives -- transient interval, steady
        mean, spin-up prefix, base deck, feedstock bulk density -- so a
        provenance record can be read directly rather than re-derived from the
        batch row plus code at a pinned commit.
        """
        out = {}
        for f in fields(self):
            if f.name.startswith('_'):
                continue
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            elif f.name == 'tracers':
                value = [vars(t) for t in value]
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value

        _, density = self.feedstock_table()
        out['derived'] = {
            'resample_freq': self.resample_freq,
            'steady_mean': self.steady_mean,
            'is_transient': self.is_transient,
            'minerals': self.minerals(),
            'feedstock_density_g_cm3': float(density),
            'application_rates_t_ha': [float(r) for r in self.application_rates()],
        }
        # Horizon boundaries are needed to interpret and plot results, and are
        # otherwise only available from a soil table on S3. Recording them here
        # lets a run describe its own soil profile.
        try:
            deck = self.resolved_base_deck
        except ValueError:
            deck = None
        if deck is not None and deck.exists():
            out['derived']['base_deck'] = str(deck)
            out['derived']['horizons'] = horizon_zones(
                InputFile.load(deck.name, path=deck.parent))

        if self.spinup_dir is not None:
            out['derived']['spinup_prefix'] = self.spinup_prefix
        return out

    def summary(self):
        """One-line-per-field view, for logging at the top of a run."""
        rates = self.application_rates()
        applied = [(i + 1, float(r)) for i, r in enumerate(rates) if r > 0]

        if not applied:
            schedule_desc = 'no applications'
        elif len(applied) <= 4:
            schedule_desc = ', '.join(f'yr {y}: {r:g} t/ha' for y, r in applied)
        else:
            distinct = sorted({r for _, r in applied})
            rate_desc = (f'{distinct[0]:g}' if len(distinct) == 1
                         else f'{distinct[0]:g}-{distinct[-1]:g}')
            schedule_desc = (f'{len(applied)} applications, yr {applied[0][0]}-'
                             f'{applied[-1][0]}, {rate_desc} t/ha')

        lines = [f'{self.run_id}  ({self.site}, {self.forcing}, '
                 f'{self.n_years} x {self.days_per_year:g} d)']
        if self.spinup_dir is not None:
            lines.append(f'  seeds from      {self.spinup_dir}  '
                         f'(deck: {self.resolved_base_deck.name})')
        lines.append(f'  feedstock       {self.feedstock.name} -> {", ".join(self.minerals())}')
        lines.append(f'  schedule        {self.schedule.name} -> {schedule_desc}')
        lines.append(f'  total applied   {float(np.sum(rates)):g} t/ha')
        lines.append(f'  grain / rough   {self.diameter_um:g} um, '
                     f'{self.roughness_option} (rf={self.const_rf:g})')
        lines.append(f'  till            {self.till_depth_m:g} m, {self.depth_distribution}, '
                     f'mix={self.mix_rule}, till_without_application='
                     f'{self.till_without_application}')
        lines.append(f'  transient out   every {self.transient_output_interval} timestep(s)')
        tracer_names = ', '.join(t.component for t in self.tracers)
        lines.append(f'  tracers         {self.tracer_mode}'
                     + (f' ({tracer_names})' if self.tracer_mode != 'carry' else ''))
        return '\n'.join(lines)
