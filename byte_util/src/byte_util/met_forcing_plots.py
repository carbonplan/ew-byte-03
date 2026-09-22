from functools import lru_cache
from pathlib import Path
from tqdm.notebook import tqdm
from scipy.integrate import cumulative_trapezoid
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from min3p.output import read_min3p_sequence
from byte_util.util import all_sites, all_forcing_types, calc_specific_discharge

# All simulations run at 20 deg C
temperature_K = 293.15
top_elevation_m = 4.01
cell_size_m = 0.01

# Define variable names to plot, the plot labels, and the files in which they're found
default_variable_specs = {
    # Flow
    'p_w':            {'column': 'p_w',            'label': 'Pressure head (m)',                    'suffix': 'gsp'},
    's_w':            {'column': 's_w',            'label': 'Water saturation (-)',                 'suffix': 'gsp'},
    'theta_a':        {'column': 'theta_a',        'label': 'Water content (m$^3$/m$^3$)',          'suffix': 'gsp'},
    'q_root':         {'column': 'q_root',         'label': 'Root uptake (m$^3$/d)',                'suffix': 'gsp'},

    # Aqueous tracers
    'psi01':          {'column': 'psi01',          'label': 'Tracer 1 (M)',                         'suffix': 'gsc'},
    'psi02':          {'column': 'psi02',          'label': 'Tracer 2 (M)',                         'suffix': 'gsc'},
    'cl-1':           {'column': 'cl-1',           'label': 'Cl$^-$ (M)',                           'suffix': 'gsc'},

    # Carbonate system
    'h+1':            {'column': 'h+1',            'label': 'pH',                                   'suffix': 'gsc'},
    'h2co3aq':        {'column': 'h2co3aq',        'label': 'H$_2$CO$_3$ (M)',                      'suffix': 'gsc'},
    'hco3-':          {'column': 'hco3-',          'label': 'HCO$_3^{-}$ (M)',                      'suffix': 'gsc'},
    'co3-2':          {'column': 'co3-2',          'label': 'CO$_3^{2-}$ (M)',                      'suffix': 'gsc'},
    'ionic strength': {'column': 'ionic strength', 'label': 'Ionic Strength',                       'suffix': 'gsm'},
    'Alk [eq/L]':     {'column': 'Alk [eq/L]',     'label': 'Alk (meq/L)',                          'suffix': 'gsm'},
    'co2(g)':         {'column': 'co2(g)',         'label': 'CO$_2$ (atm)',                         'suffix': 'gsg'},
    'co2_resp':       {'column': 'co2_resp',       'label': 'CO$_2$ source (m$^3$/m$^3$)',          'suffix': 'gsv'},

    # Gas tracer
    'gtr(aq)':        {'column': 'gtr(aq)',        'label': 'Gas tracer (aq; M)',                   'suffix': 'gsc'},
    'gtr(g)':         {'column': 'gtr(g)',         'label': 'Gas tracer (g; atm)',                  'suffix': 'gsg'},
    'p_t_o_t':        {'column': 'p_t_o_t',        'label': 'Total gas pressure (atm)',             'suffix': 'gsg'},

    # Major cations
    'mg+2':           {'column': 'mg+2',           'label': 'Mg$^{2+}$ (M)',                        'suffix': 'gsc'},
    'ca+2':           {'column': 'ca+2',           'label': 'Ca$^{2+}$ (M)',                        'suffix': 'gsc'},
    'na+1':           {'column': 'na+1',           'label': 'Na$^{+}$ (M)',                         'suffix': 'gsc'},
    'k+1':            {'column': 'k+1',            'label': 'K$^{+}$ (M)',                          'suffix': 'gsc'},
    'al+3':           {'column': 'al+3',           'label': 'Al$^{3+}$ (M)',                        'suffix': 'gsc'},

    # Total concentrations
    'tot_h':          {'column': 'h+1',            'label': 'Total H (M)',                          'suffix': 'gst'},
    'tot_c':          {'column': 'co3-2',          'label': 'Total C (M)',                          'suffix': 'gst'},
    'tot_mg':         {'column': 'mg+2',           'label': 'Total Mg (M)',                         'suffix': 'gst'},
    'tot_ca':         {'column': 'ca+2',           'label': 'Total Ca (M)',                         'suffix': 'gst'},
    'tot_na':         {'column': 'na+1',           'label': 'Total Na (M)',                         'suffix': 'gst'},
    'tot_k':          {'column': 'k+1',            'label': 'Total K (M)',                          'suffix': 'gst'},
    'tot_al':         {'column': 'al+3',           'label': 'Total Al (M)',                         'suffix': 'gst'},
    'tot_si':         {'column': 'h4sio4',         'label': 'Total Si (M)',                         'suffix': 'gst'},

    # Cation exchange
    'base_sat':       {'column': 'x',              'label': 'Base saturation (%)',                  'suffix': 'gsb'},
    'ca_exch':        {'column': 'ca-x(na)',       'label': 'Exch. Ca (%)',                  'suffix': 'gsb'},
    'mg_exch':        {'column': 'mg-x(na)',       'label': 'Exch. Mg (%)',                  'suffix': 'gsb'},
    'k_exch':         {'column': 'k-x(na)',        'label': 'Exch. K (%)',                   'suffix': 'gsb'},
    'na_exch':        {'column': 'na-x(na)',       'label': 'Exch. Na (%)',                  'suffix': 'gsb'},
    'h_exch':         {'column': 'h-x(na)',        'label': 'Exch. H (%)',                   'suffix': 'gsb'},
    'al_exch':        {'column': 'al-x(na)',       'label': 'Exch. Al (%)',                  'suffix': 'gsb'},

    # Silica system
    'h4sio4':         {'column': 'h4sio4',         'label': 'H$_4$SiO$_4$ (M)',                     'suffix': 'gsc'},
    'h3sio4-':        {'column': 'h3sio4-',        'label': 'H$_3$SiO$_4^{-}$ (M)',                 'suffix': 'gsc'},
    'h2sio4-2':       {'column': 'h2sio4-2',       'label': 'H$_2$SiO$_4^{2-}$ (M)',                'suffix': 'gsc'},
    'sio2_vol':       {'column': 'sio2(a,pt)',     'label': 'SiO$_{2(am)}$ (m$^3$/m$^3$)',          'suffix': 'gsv'},
    'gibbsite_vol':   {'column': 'gibbsite-ph',    'label': 'Al(OH)$_{3(am)}$ (m$^3$/m$^3$)',       'suffix': 'gsv'},

    # K-feldspar
    'k_feld_neutral': {'column': 'k-feld-d-ph_1', 'label': 'Neutral diss.',                        'suffix': 'gsd'},
    'k_feld_acid':    {'column': 'k-feld-d-ph_2', 'label': 'Acid diss.',                           'suffix': 'gsd'},
    'k_feld_base':    {'column': 'k-feld-d-ph_3', 'label': 'Base diss.',                           'suffix': 'gsd'},
    'k_feld_SI':      {'column': 'k-feld-d-ph',   'label': 'SI',                                   'suffix': 'gss'},
    'k_feld_vol':     {'column': 'k-feld-d-ph',   'label': 'Vol. frac.',                           'suffix': 'gsv'},

    # Na-montmorillonite
    'namont_neutral': {'column': 'na-montmor_1',  'label': 'Neutral diss.',                        'suffix': 'gsd'},
    'namont_acid':    {'column': 'na-montmor_2',  'label': 'Acid diss.',                           'suffix': 'gsd'},
    'namont_base':    {'column': 'na-montmor_3',  'label': 'Base diss.',                           'suffix': 'gsd'},
    'namont_SI':      {'column': 'na-montmor',    'label': 'SI',                                   'suffix': 'gss'},
    'namont_vol':     {'column': 'na-montmor',    'label': 'Vol. frac.',                           'suffix': 'gsv'},

    # Calcite
    'calcite_neutral': {'column': 'calcite-ph_1', 'label': 'Neutral diss.',                        'suffix': 'gsd'},
    'calcite_acid':    {'column': 'calcite-ph_2', 'label': 'Acid diss.',                           'suffix': 'gsd'},
    'calcite_base':    {'column': 'calcite-ph_3', 'label': 'Base diss.',                           'suffix': 'gsd'},
    'calcite_SI':      {'column': 'calcite-ph',   'label': 'SI',                                   'suffix': 'gss'},
    'calcite_vol':     {'column': 'calcite-ph',   'label': 'Vol. frac.',                           'suffix': 'gsv'},

    # Forsterite
    'forst_neutral':  {'column': 'forst-ph_1',    'label': 'Neutral diss.',                        'suffix': 'gsd'},
    'forst_acid':     {'column': 'forst-ph_2',    'label': 'Acid diss.',                           'suffix': 'gsd'},
    'forst_SI':       {'column': 'forst-ph',      'label': 'SI',                                   'suffix': 'gss'},
    'forst_vol':      {'column': 'forst-ph',      'label': 'Vol. frac.',                           'suffix': 'gsv'},
}

# Default profiles to plot for spin-up and ERW simulations
profiles_to_plot = {
    1: {'vars': ['p_w', 'theta_a', 'q_root'], 'title': 'Water content and root uptake'},
    2: {'vars': ['psi01', 'psi02', 'cl-1'], 'title': 'Tracer profiles'},
    3: {'vars': ['h+1', 'hco3-', 'co2(g)', 'co2_resp', 'Alk [eq/L]'], 'title': 'pH, alkalinity, and CO2 respiration'},
    4: {'vars': ['gtr(aq)', 'gtr(g)', 'co2(g)', 'p_t_o_t', 'co2_resp'], 'title': 'CO2 and gas tracer'},
    5: {'vars': ['h2co3aq', 'hco3-', 'co3-2'], 'title': 'Carbonate speciation'},
    6: {'vars': ['tot_h', 'tot_c', 'tot_mg', 'tot_ca', 'tot_na',
                 'tot_al', 'tot_si'], 'title': 'Total concentrations'},
    7: {'vars': ['mg+2', 'ca+2', 'na+1', 'k+1', 'al+3'], 'title': 'Major cations'},
    8: {'vars': ['base_sat', 'ca_exch', 'mg_exch', 'na_exch', 'k_exch'],
        'title': 'Base saturation and exchange composition'},
    9: {'vars': ['base_sat', 'al_exch', 'h_exch'], 'title': 'Base saturation and exchangeable acidity'},
    10: {'vars': ['plot_exchange_profiles'], 'title': 'Final exchanger composition'},
    11: {'vars': ['h+1', 'ca+2', 'hco3-', 'co3-2', 'calcite_vol'], 'title': 'Calcite buffering'},
    12: {'vars': ['calcite_neutral', 'calcite_acid', 'calcite_base', 'calcite_SI',
                  'calcite_vol'], 'title': 'Calcite dissolution'},
    13: {'vars': ['h4sio4', 'h3sio4-', 'h2sio4-2', 'sio2_vol', 'gibbsite_vol'], 'title': 'Silica system'},
    14: {'vars': ['k_feld_neutral', 'k_feld_acid', 'k_feld_base', 'k_feld_SI',
                  'k_feld_vol'], 'title': 'K-feldspar dissolution'},
    15: {'vars': ['namont_neutral', 'namont_acid', 'namont_base', 'namont_SI',
                  'namont_vol'], 'title': 'Na-montmorillonite'},
    16: {'vars': ['forst_neutral', 'forst_acid', 'forst_SI', 'forst_vol'], 'title': 'Forsterite dissolution'},
}

@lru_cache(maxsize=None)
def read_spatial(site, forcing, scenario, suffix, sim_root=Path('../min3p_runs')):
    if forcing == 'spinup' or scenario == 'spinup':
        folder = sim_root / site / 'spinup'
        data, cols, times = read_min3p_sequence(folder / f'spinup_1.{suffix}')
    else:
        folder = sim_root / site / f'{forcing}_{scenario}'
        data, cols, times = read_min3p_sequence(folder / f'{forcing}_1.{suffix}')
    return data, cols, np.asarray(times)


@lru_cache(maxsize=None)
def read_transient(site, forcing, scenario, suffix, sim_root=Path('../min3p_runs')):
    if forcing == 'spinup' or scenario == 'spinup':
        folder = sim_root / site / 'spinup'
        return read_min3p_sequence(f'spinup_1.{suffix}', folder=folder, ftype='transient')
    else:
        folder = sim_root / site / f'{forcing}_{scenario}'
        return read_min3p_sequence(f'{forcing}_1.{suffix}', folder=folder, ftype='transient')


def spatial_variable(site, forcing, scenario, variable, variable_specs=None,
                     dualperm=False, domain=0):
    if variable_specs is None:
        variable_specs = default_variable_specs
    suffix = variable_specs[variable]['suffix']
    column = variable_specs[variable]['column']
    data, cols, times = read_spatial(site, forcing, scenario, suffix)
    if dualperm:
        values = data[:, cols.index(column), domain, :]
        depth = data[0, 2, domain].max() - data[0, 2, domain]
    else:
        values = data[:, cols.index(column), :]
        depth = data[0, 2].max() - data[0, 2]
    values = -np.log10(values) if variable == 'h+1' else 1e3 * values if variable == 'Alk [eq/L]' else values
    return values, times, depth


def export_series(site, forcing, scenario, control_plane_cm=300, sim_root=Path('../min3p_runs')):
    folder = sim_root / site / f'{forcing}_{scenario}'
    gbp, gbp_cols, cells = read_transient(site, forcing, scenario, 'gbp')
    gbm, gbm_cols, _ = read_transient(site, forcing, scenario, 'gbm')

    top_cell = round(top_elevation_m / cell_size_m)
    offset = round(control_plane_cm / 100 / cell_size_m)
    ci_up = list(cells).index(top_cell - offset)
    ci_dn = list(cells).index(top_cell - offset - 1)
    elev_up = top_elevation_m - control_plane_cm / 100
    elev_dn = elev_up - cell_size_m

    q = np.asarray(calc_specific_discharge(gbp, input_file=folder / f'{forcing}.dat', ci_a=ci_up, ci_b=ci_dn, elev_a=elev_up, elev_b=elev_dn))
    n = len(q)
    upstream = gbp[ci_up, gbp_cols.index('h_w'), -n:] > gbp[ci_dn, gbp_cols.index('h_w'), -n:]
    alkalinity = np.where(upstream, gbm[ci_up, gbm_cols.index('Alk [eq/L]'), -n:], gbm[ci_dn, gbm_cols.index('Alk [eq/L]'), -n:])
    time = gbp[ci_up, gbp_cols.index('time'), -n:]
    drainage = np.clip(-q, 0, None)
    rate = 1000 * drainage * alkalinity

    return pd.DataFrame({
        'time_d': time,
        'drainage_m_d': drainage,
        'alkalinity_eq_L': alkalinity,
        'alkalinity_export_eq_m2_d': rate,
        'cumulative_drainage_m': cumulative_trapezoid(drainage, time, initial=0),
        'cumulative_alkalinity_eq_m2': cumulative_trapezoid(rate, time, initial=0),
    })


def plot_profiles(vars_to_plot, site, forcing, scenario='ctrl', figsize=None, starting_timestep=-1.0,
                  label_unit='y', label_freq=2, colorbar=True, variable_specs=None, dualperm=False):
    # List of scenarios if both ctrl and erw are requested
    scenarios = ['ctrl', 'erw'] if scenario == 'both' else [scenario]
    domains = [('Macropore', 0), ('Matrix', 1)] if dualperm else [(None, None)]

    if variable_specs is None:
        variable_specs = default_variable_specs

    if forcing == 'spinup':
        ncols = len(vars_to_plot)
        nrows = len(domains)
    else:
        ncols = len(vars_to_plot) * len(domains)
        nrows = len(scenarios)

    if figsize is None:
        width = 16 if dualperm and forcing != 'spinup' else 8
        height = 4 * nrows
        figsize = (width, height)

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize,
                           sharex='col', sharey=True, constrained_layout=True)
    ax = np.atleast_2d(ax)

    # Set up consistent colors across all scenarios
    timesteps_to_plot = []
    for scenario in scenarios:
        _, timesteps, _ = spatial_variable(site, forcing, scenario, 'theta_a',
                                           dualperm=dualperm, domain=0)
        test_timesteps_to_plot = [t for t in timesteps if t >= starting_timestep]
        if len(test_timesteps_to_plot) > len(timesteps_to_plot):
            timesteps_to_plot = test_timesteps_to_plot
    cmap = plt.colormaps['cividis']
    colorbar_vals = np.array(timesteps_to_plot) if label_unit == 'd' else np.array(timesteps_to_plot) / 365
    norm = plt.Normalize(colorbar_vals.min(), colorbar_vals.max())
    colors = cmap(norm(colorbar_vals))

    for k, scenario in enumerate(scenarios):
        for d, (domain_name, domain) in enumerate(domains):
            for j, var_name in enumerate(vars_to_plot):
                if forcing == 'spinup':
                    col = j
                    row = d
                else:
                    col = d * len(vars_to_plot) + j
                    row = k
                arr, timesteps, depth = spatial_variable(site, forcing, scenario, var_name,
                                                         dualperm=dualperm,
                                                         domain=domain if dualperm else 0)

                # Handle cation exchange columns differently
                exchange_vars = ['base_sat', 'ca_exch', 'mg_exch', 'k_exch',
                                 'na_exch', 'h_exch', 'al_exch']
                if var_name in exchange_vars:
                    q = np.empty((6, *arr.shape), dtype=arr.dtype)
                    for i, exchange_var in enumerate(exchange_vars[1:]):
                        q[i], _, _ = spatial_variable(site, forcing, scenario, exchange_var,
                                                      dualperm=dualperm,
                                                      domain=domain if dualperm else 0)
                    total = q.sum(axis=0)
                    frac = 100 * np.divide(q, total[None, :, :],
                                           out=np.full_like(q, np.nan),
                                           where=total[None, :, :] > 0)
                    if var_name == 'base_sat':
                        plot_arr = frac[:4].sum(axis=0)
                    else:
                        plot_arr = frac[exchange_vars[1:].index(var_name)]
                else:
                    plot_arr = arr

                for i, timestep in enumerate(timesteps_to_plot):
                    if timestep not in list(timesteps):
                        continue
                    time_idx = list(timesteps).index(timestep)
                    if label_unit == 'd':
                        label = f'{timestep:.0f} d' if (i + 1) % label_freq == 0 else None
                    else:
                        label = f'{timestep / 365:.0f} y' if (i + 1) % label_freq == 0 else None

                    values = plot_arr[time_idx]
                    ls = '--' if domain_name == 'Macropore' else '-'
                    ax[row, col].plot(values, depth, color=colors[i], label=label, ls=ls)
                if row == nrows - 1:
                    ax[row, col].set(xlabel=variable_specs[var_name]['label'])

                if dualperm:
                    title = domain_name if len(scenarios) == 1 else f'{scenario}: {domain_name}'
                    ax[row, col].set(title=title)
                elif len(scenarios) > 1:
                    ax[row, col].set(title=scenario)

                ax[row, 0].set(ylabel='Depth', ylim=(depth.max(), depth.min()))
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02, shrink=0.7)
        cbar.set_label('Time [d]' if label_unit == 'd' else 'Time [y]')
    else:
        ax[0, 0].legend()

    return fig, np.squeeze(ax)


def plot_exchange_profiles(site, forcing, scenario='spinup', itime=-1, sim_root='../min3p_runs/',
                           dualperm=False):
    species = {'Ca': 'ca-x(na)',
               'Mg': 'mg-x(na)',
               'K': 'k-x(na)',
               'Na': 'na-x(na)',
               'H': 'h-x(na)',
               'Al': 'al-x(na)'}

    # Define sim_folder
    if forcing == 'spinup':
        sim_folder = Path(sim_root) / site / f'{forcing}'
    else:
        sim_folder = Path(sim_root) / site / f'{forcing}_{scenario}'

    gsb, gsb_cols, timesteps = read_min3p_sequence(f'{forcing}_1.gsb', folder=sim_folder)
    domains = [('Macropore', 0), ('Matrix', 1)] if dualperm else [(None, None)]
    figsize = (8, 10) if dualperm and forcing == 'spinup' else (8 * len(domains), 5)
    fig, ax = plt.subplots(len(domains), 2, figsize=figsize, sharey=True, tight_layout=True)
    ax = np.atleast_2d(ax)
    colors = plt.colormaps['cividis'](np.linspace(0, 1, len(timesteps)))

    for d, (domain_name, domain) in enumerate(domains):
        if dualperm:
            q = np.stack([gsb[:, gsb_cols.index(s), domain, :] for s in species.values()], axis=1)
            depth = gsb[0, 2, domain].max() - gsb[0, 2, domain]
        else:
            q = np.stack([gsb[:, gsb_cols.index(s), :] for s in species.values()], axis=1)
            depth = gsb[0, 2].max() - gsb[0, 2]

        total = q.sum(axis=1, keepdims=True)
        frac = 100 * np.divide(q, total, out=np.full_like(q, np.nan), where=total > 0)
        base_sat = frac[:, :4].sum(axis=1)
        ax_base, ax_exch = ax[d]

        for i, timestep in enumerate(timesteps):
            ax_base.plot(base_sat[i], depth, color=colors[i],
                         label=f'{timestep:.0f} d' if i % 4 == 0 else None)

        left = np.zeros_like(depth)
        for label, values in zip(species, frac[itime]):
            ax_exch.fill_betweenx(depth, left, left + values, label=label)
            left += values

        ax_base.set(xlabel='Base saturation (%)', ylim=(depth.max(), depth.min()))
        ax_exch.set(xlabel='Exchange-site occupancy (%)', title=f'{timesteps[itime]:.0f} d')
        if dualperm:
            ax_base.set_title(domain_name)
            ax_exch.set_title(f'{domain_name}: {timesteps[itime]:.0f} d')

        ax[d, 0].set(ylabel='Depth')
    ax[0, 0].legend()
    ax[0, 1].legend()

    return fig, ax


def plot_cumulative_export(site, figsize=(12, 10), control_plane_cm=300, scenarios=None,
                           forcing_types=None, forcing_colors=None):
    if scenarios is None:
        scenarios = ['ctrl', 'erw']
    if forcing_types is None:
        forcing_types = all_forcing_types
    if forcing_colors is None:
        forcing_colors = dict(zip(forcing_types, plt.colormaps['Set2'](np.linspace(0.05, 0.95, 4))))

    fig, ax = plt.subplots(3, 3, figsize=figsize, sharex='col', tight_layout=True)

    exports = {}

    # Plot CTRL and ERW
    for i, scenario in enumerate(scenarios):
        for forcing in forcing_types:
            export = export_series(site, forcing, scenario, control_plane_cm=control_plane_cm)
            exports[forcing, scenario] = export
            years = export['time_d'] / 365
            ax[i, 0].plot(years, 1000 * export['drainage_m_d'], color=forcing_colors[forcing], label=forcing)
            ax[i, 1].plot(years, 1000 * export['alkalinity_eq_L'], color=forcing_colors[forcing])
            ax[i, 2].plot(years, export['cumulative_alkalinity_eq_m2'], color=forcing_colors[forcing])
        ax[i, 0].set(ylabel=f'{scenario.upper()}\nDrainage (mm/d)')
        ax[i, 1].set(ylabel='Alkalinity (meq/L)')
        ax[i, 2].set(ylabel='Cumulative export (eq/m$^2$)')

    # Plot ERW - CTRL
    for forcing in forcing_types:
        ctrl = exports[forcing, 'ctrl']
        erw = exports[forcing, 'erw']
        time = erw['time_d'].to_numpy()
        years = time / 365

        drainage_diff = erw['drainage_m_d'].to_numpy() - np.interp(time, ctrl['time_d'], ctrl['drainage_m_d'])
        alkalinity_diff = erw['alkalinity_eq_L'].to_numpy() - np.interp(time, ctrl['time_d'], ctrl['alkalinity_eq_L'])
        export_diff = erw['cumulative_alkalinity_eq_m2'].to_numpy() - np.interp(time, ctrl['time_d'], ctrl['cumulative_alkalinity_eq_m2'])

        ax[2, 0].plot(years, 1000 * drainage_diff, color=forcing_colors[forcing])
        ax[2, 1].plot(years, 1000 * alkalinity_diff, color=forcing_colors[forcing])
        ax[2, 2].plot(years, export_diff, color=forcing_colors[forcing])

    ax[2, 0].set_ylabel('ERW - CTRL\nDrainage (mm/d)')
    ax[2, 1].set_ylabel('Alkalinity (meq/L)')
    ax[2, 2].set_ylabel('Cumulative export (eq/m$^2$)')

    for axis in ax[2]:
        axis.axhline(0, color='k', lw=0.8)
        axis.set_xlabel('Time (yr)')

    ax[0, 0].legend()
    fig.suptitle(f'{site}: drainage and alkalinity export at {control_plane_cm} cm')
    return fig, ax


def plot_final_profiles(site, variables, figsize=None, scenarios=None, forcing_types=None, forcing_colors=None,
                        variable_specs=None):
    if scenarios is None:
        scenarios = ['ctrl', 'erw']
    if forcing_types is None:
        forcing_types = all_forcing_types
    if forcing_colors is None:
        forcing_colors = dict(zip(forcing_types, plt.colormaps['Set2'](np.linspace(0.05, 0.95, 4))))
    if figsize is None:
        figsize = (2.5 * len(variables), 7)
    if variable_specs is None:
        variable_specs = default_variable_specs

    fig, ax = plt.subplots(2, len(variables), figsize=figsize, sharey=True,
                           sharex='col', squeeze=False, tight_layout=True)

    for i, scenario in enumerate(scenarios):
        for j, variable in enumerate(variables):
            for forcing in forcing_types:
                values, times, depth = spatial_variable(site, forcing, scenario, variable)
                if times[-1] < 3640:
                    label = f'{forcing} ({times[-1]:.0f} d)'
                else:
                    label = f'{forcing} (10 y)'
                ax[i, j].plot(values[-1], depth, color=forcing_colors[forcing], label=label)
                if variable == 'forst_vol' and scenario == 'erw':
                    # Add initial forsterite
                    ax[i, j].plot(values[0], depth, color='k', linestyle='--', label='Initial', alpha=0.5)
            ax[i, j].set(xlabel=variable_specs[variable]['label']) if i == 1 else None
            ax[i, j].set(ylabel=f'{scenario.upper()}\nDepth (m)', ylim=(depth.max(), depth.min())) if j == 0 else None
    ax[1, 0].legend(loc='lower left')
    fig.suptitle(f'{site}: final profiles')
    return fig, ax


def plot_final_erw_difference(site, variables, figsize=None, forcing_types=None, forcing_colors=None,
                              variable_specs=None):
    if forcing_types is None:
        forcing_types = all_forcing_types
    if forcing_colors is None:
        forcing_colors = dict(zip(forcing_types, plt.colormaps['Set2'](np.linspace(0.05, 0.95, 4))))
    if variable_specs is None:
        variable_specs = default_variable_specs

    fig, ax = plt.subplots(1, len(variables), figsize=figsize or (2.8 * len(variables), 5), sharey=True, squeeze=False, tight_layout=True)
    ax = ax[0]

    for j, variable in enumerate(variables):
        for forcing in forcing_types:
            ctrl, times, depth = spatial_variable(site, forcing, 'ctrl', variable)
            erw, times, erw_depth = spatial_variable(site, forcing, 'erw', variable)
            order = np.argsort(erw_depth)
            ax[j].plot(np.interp(depth, erw_depth[order], erw[-1, order]) - ctrl[-1], depth, color=forcing_colors[forcing], label=forcing)
        ax[j].axvline(0, color='k', linewidth=0.8)
        ax[j].set(xlabel=f'ERW - control\n{variable_specs[variable]['label']}')

    ax[0].set(ylabel='Depth (m)', ylim=(depth.max(), depth.min()))
    ax[0].legend()
    fig.suptitle(f'{site}: Difference in final profiles')
    return fig, ax


def time_mean(values, time, axis=0):
    return np.trapezoid(values, time, axis=axis) / (time[-1] - time[0])


def plot_mean_spatial_gas_profiles(site, figsize=(6, 8), scenarios=None, forcing_types=None,
                                   forcing_colors=None, variable_specs=None):
    if scenarios is None:
        scenarios = ['ctrl', 'erw']
    if forcing_types is None:
        forcing_types = all_forcing_types
    if forcing_colors is None:
        forcing_colors = dict(zip(forcing_types, plt.colormaps['Set2'](np.linspace(0.05, 0.95, 4))))
    if variable_specs is None:
        variable_specs = default_variable_specs
    variables = ['theta_a', 'co2(g)']
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True, sharex='col', tight_layout=True)

    for i, scenario in enumerate(scenarios):
        for j, variable in enumerate(variables):
            for forcing in forcing_types:
                values, times, depth = spatial_variable(site, forcing, scenario, variable)
                mean_values = time_mean(values, times)
                ax[i, j].plot(mean_values, depth, color=forcing_colors[forcing], label=forcing)
            ax[i, j].set(xlabel=variable_specs[variable]['label'])
        ax[i, 0].set(ylabel=f'{scenario.upper()}\nDepth (m)', ylim=(depth.max(), depth.min()))
    ax[0, 0].legend()
    fig.suptitle(f'{site}: mean water content and gas profiles')
    return fig, ax


def plot_mean_transient_gas_profiles(site, figsize=(6, 8), scenarios=None, forcing_types=None, forcing_colors=None,
                                     variable_specs=None):
    if scenarios is None:
        scenarios = ['ctrl', 'erw']
    if forcing_types is None:
        forcing_types = all_forcing_types
    if forcing_colors is None:
        forcing_colors = dict(zip(forcing_types, plt.colormaps['Set2'](np.linspace(0.05, 0.95, 4))))
    if variable_specs is None:
        variable_specs = default_variable_specs

    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True, sharex='col', tight_layout=True)

    for i, scenario in enumerate(scenarios):
        for forcing in forcing_types:
            gbp, gbp_cols, cells = read_transient(site, forcing, scenario, 'gbp')
            gbg, gbg_cols, _ = read_transient(site, forcing, scenario, 'gbg')
            gbg = gbg[:, :, 1:]  # Trim zeroth time step (initial concentrations)
            times = gbp[0, gbp_cols.index('time')]
            depth = top_elevation_m - np.asarray(cells) * cell_size_m
            order = np.argsort(depth)
            profiles = [time_mean(gbp[:, gbp_cols.index('theta_a')], times, axis=1),
                        time_mean(gbg[:, gbg_cols.index('co2(g)')], times, axis=1)]

            for j, profile in enumerate(profiles):
                ax[i, j].plot(profile[order], depth[order], marker='o',
                              color=forcing_colors[forcing], label=forcing)

        for j, variable in enumerate(['theta_a', 'co2(g)']):
            ax[i, j].set(xlabel=variable_specs[variable]['label'])
        ax[i, 0].set(ylabel=f'{scenario.upper()}\nDepth (m)', ylim=(depth.max(), depth.min()))
    ax[0, 0].legend()
    fig.suptitle(f'{site}: mean water content and gas profiles, from transient output')
    return fig, ax


def plot_feedstock_dissolution(figsize=(10, 7), timestep=3650, forcing_types=None, forcing_colors=None,):
    if forcing_types is None:
        forcing_types = all_forcing_types
    if forcing_colors is None:
        forcing_colors = dict(zip(forcing_types, plt.colormaps['Set2'](np.linspace(0.05, 0.95, 4))))

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, tight_layout=True)
    width = 0.18
    x = np.arange(len(all_sites))
    dissolution = {}

    for site in all_sites:
        for forcing in forcing_types:
            forsterite, time, _ = spatial_variable(site, forcing, 'erw', 'forst_vol')
            dissolution[site, forcing] = 100 * (1 - forsterite[-1].sum() / forsterite[0].sum()) if time[-1] == timestep else np.nan

    for j, forcing in enumerate(forcing_types):
        values = np.array([dissolution[site, forcing] for site in all_sites])
        hourly = np.array([dissolution[site, 'hourly'] for site in all_sites])
        error = 100 * (values - hourly) / hourly

        pos = x + (j - (len(forcing_types) - 1) / 2) * width
        ax[0].bar(pos, values, width, color=forcing_colors[forcing], label=forcing, edgecolor='k')
        ax[1].bar(pos, error, width, color=forcing_colors[forcing], edgecolor='k')

    ax[0].set_ylabel('Forsterite dissolved (%)')
    ax[1].set(ylabel='Error relative to hourly (%)', xticks=x, xticklabels=all_sites)
    ax[1].axhline(0, color='k', lw=0.8)
    ax[0].legend()

    return fig, ax


def plot_bcvs_soi(site, forcing, scenario, figsize=None,
                  time_unit='y', sim_root='../min3p_runs'):

    if figsize is None:
        figsize = (8, 6)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, tight_layout=True)

    # Open gbp file to get last successful timestep
    sim_folder = Path(sim_root) / site / f'{forcing}_{scenario}'
    gbp, gbp_cols, cells = read_min3p_sequence(f'{forcing}_1.gbp', folder=sim_folder, ftype='transient')
    last_timestep = gbp[0, gbp_cols.index('time'), -1]

    soi = pd.read_csv(sim_folder / f'{forcing}.soi', sep=r'\s+', names=['time', 'T'])
    soi.set_index('time', inplace=True)
    bcvs = pd.read_csv(sim_folder / f'{forcing}.bcvs', sep=r'\s+', names=['time', 'top', 'bottom'])
    bcvs.set_index('time', inplace=True)

    # Limit to last_timestep
    soi = soi[soi.index <= last_timestep]
    bcvs = bcvs[bcvs.index <= last_timestep]

    time = soi.index.values
    if time_unit == 'y':
        time /= 365

    ax[0].plot(time, bcvs['top'], color='steelblue')
    ax[1].plot(time, soi['T'], color='forestgreen')
    ax[0].set(ylabel='Top boundary (m/s)')
    ax[1].set(ylabel='Transpiration (1/s)')

    return fig, ax


def plot_histories(vars, site, forcing, scenario, plot_cells=None, figsize=None,
                   time_unit='y', variable_specs=None, colorbar=True,
                   sim_root='../min3p_runs'):

    if figsize is None:
        figsize = (8, 8)

    if variable_specs is None:
        variable_specs = default_variable_specs

    fig, ax = plt.subplots(len(vars), 1, figsize=figsize, sharex=True, constrained_layout=True)

    # Open gbp file to get list of available cells to plot
    if forcing == 'spinup':
        sim_folder = Path(sim_root) / site / 'spinup'
    else:
        sim_folder = Path(sim_root) / site / f'{forcing}_{scenario}'
    gbp, gbp_cols, cells = read_min3p_sequence(f'{forcing}_1.gbp', folder=sim_folder, ftype='transient')

    if plot_cells is None:
        plot_cells = cells
    else:
        plot_cells = np.array(plot_cells)
        # Check that each of plot cells is in cells
        for c in plot_cells:
            if c not in cells:
                raise ValueError(f'Cell {c} not found in cells: {cells}')

    cmap = plt.colormaps['cividis']
    norm = plt.Normalize(plot_cells.min(), plot_cells.max())
    colors = cmap(norm(plot_cells))

    for j, var_name in enumerate(vars):
        suffix = variable_specs[var_name]['suffix']
        transient_suffix = suffix.replace('gs', 'gb')  # Assume transient output is same suffix with gb
        col = variable_specs[var_name]['column']
        xlabel = variable_specs[var_name]['label']
        arr, arr_cols, cells = read_min3p_sequence(f'{forcing}_1.{transient_suffix}', folder=sim_folder, ftype='transient')
        time = arr[0, arr_cols.index('time')]
        if time_unit == 'y':
            time /= 365

        # Handle cation exchange columns differently
        exchange_cols = ['ca-x(na)', 'mg-x(na)', 'k-x(na)', 'na-x(na)',
                         'h-x(na)', 'al-x(na)']
        exchange_vars = ['base_sat', 'ca_exch', 'mg_exch', 'k_exch',
                         'na_exch', 'h_exch', 'al_exch']
        if var_name in exchange_vars:
            q = np.stack([arr[:, arr_cols.index(s), :] for s in exchange_cols], axis=1)
            total = q.sum(axis=1)
            frac = 100 * np.divide(q, total[:, None, :],
                                   out=np.full_like(q, np.nan),
                                   where=total[:, None, :] > 0)

            if var_name == 'base_sat':
                plot_arr = frac[:, :4].sum(axis=1)
            else:
                plot_arr = frac[:, exchange_vars[1:].index(var_name)]

        else:
            plot_arr = arr[:, arr_cols.index(col), :]

        for i, cell in enumerate(plot_cells):
            arr_idx = list(cells).index(cell)

            if var_name == 'h+1':
                values = -np.log10(plot_arr[arr_idx])
            elif col == 'Alk [eq/L]':
                values = 1e3 * plot_arr[arr_idx]
            else:
                values = plot_arr[arr_idx]

            ax[j].plot(time, values, color=colors[i], label=f'{cell}')
            ax[j].set(ylabel=xlabel)

        ax[-1].set(xlabel=f'Time ({time_unit})')
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02, shrink=0.7)
        cbar.set_label('Cell #')
    else:
        ax[0].legend('Cell #')

    return fig, ax
