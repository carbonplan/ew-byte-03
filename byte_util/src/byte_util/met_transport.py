"""Miscellaneous functions used in post-processing of met forcing transport tests."""

import pandas as pd
import numpy as np
from .util import calc_specific_discharge, single_to_double_float
from min3p.output import read_min3p_sequence


def create_bcvs_soi(met_forcing, freq, start_date='2001-01-01', bottom_bc=0.0,
                    dz=0.01, delete_first_record=True, dt=1e-3):
    """Create dataframes for specifying transient boundary conditions
    and transient transpiration in MIN3P.

    Parameters
    ----------
    met_forcing: pd.DataFrame
        A DataFrame containing hourly forcing data
    freq: str
        The frequency of the output forcing data ('h', 'D', 'MS')
    start_date: str
        The start date of the output forcing data (default: '2001-01-01')
    bottom_bc: float
        The value to apply to bottom BC of the output forcing data (default: 0.0)
    dz: float
        The vertical grid cell size, used to convert transpiration (default: 0.01 m)
    delete_first_record: bool
        Whether to delete the output forcing data (default: True)
    dt: float
        The time step to use for the first record of the output forcing data. Only used
        if delete_first_record is False.

    Returns
    -------
    bcvs: pd.DataFrame
        A DataFrame containing hourly boundary condition data that can be written
        to a *.bcvs file
    soi: pd.DataFrame
        A DataFrame containing hourly transpiration values that can be written
        to a *.soi file"""

    if freq not in ['h', 'D', 'MS', 'hourly', 'daily', 'monthly']:
        raise ValueError("freq must be 'hourly'/'h', 'daily'/'D', or 'monthly'/'MS'")

    if freq == 'hourly':
        freq = 'h'
    elif freq == 'daily':
        freq = 'D'
    elif freq == 'monthly':
        freq = 'MS'

    # First, convert from mm/hr to m/s (surface_flux) and m/d (transpiration)
    met_forcing['surface_flux_m.s'] = met_forcing['surface_flux_mm.hr'] / 1000 / 60 / 60
    met_forcing['transpiration_m.d'] = met_forcing['transpiration_mm.hr'] / 1000 * 24

    # Subsample (ie, start at start_date) and resample (ie, calc temporal means)
    subsampled = met_forcing.loc[start_date:met_forcing.index[-1], :]
    resampled = subsampled.resample(freq).mean()

    # Create new index
    idx = (resampled.index - resampled.index[0])
    bcvs = pd.DataFrame(index=idx, dtype='float',
                        columns=['time', 'surface_flux_m.s', 'head_m'])
    bcvs['time'] = idx.total_seconds() / 60 / 60 / 24  # Convert time to days
    bcvs['surface_flux_m.s'] = resampled['surface_flux_m.s'].values
    bcvs['head_m'] = bottom_bc

    # For transpiration, calculate transpiration_factor as 1/d
    soi = pd.DataFrame(index=idx, dtype='float',
                       columns=['time', 'transpiration_factor'])
    soi['time'] = idx.total_seconds() / 60 / 60 / 24  # Convert to days
    soi['transpiration_factor'] = resampled['transpiration_m.d'].values / dz

    # If requested, drop the first record, since that is entered directly
    # into the MIN3P input file
    if delete_first_record:
        bcvs.drop(idx[0], inplace=True)
        soi.drop(idx[0], inplace=True)
    # Otherwise, change the first timestep to dt
    # If dt is small enough, the value entered into the MIN3P input file
    # doesn't matter, as it will be quickly replaced by the value in the
    # *.bcvs or *.soi file
    else:
        bcvs.loc[idx[0], 'time'] = dt
        soi.loc[idx[0], 'time'] = dt

    return bcvs, soi


def write_transient(filepath, df, formatters=None):
    """Write a DataFrame with transient boundary conditions either to a *.bcvs or
    *soi file formatted for use in MIN3P.

    Parameters
    df: pd.DataFrame
        A DataFrame containing hourly boundary condition or transpiration data
    filepath: str
        The filepath to write the output to
    formatters: list
        A list of formats for the output columns"""

    if formatters is None:
        if 'transpiration_factor' in df.columns:
            formatters = ["{:.3f}".format, "  {:.4e}".format]
        else:
            formatters = ["{:.3f}".format, "  {:.4e}".format, "  {:.2f}".format]

    with open(filepath, 'w') as f:
        f.write(df.to_string(header=False, index=False, formatters=formatters))


def calc_tracer_flux(sim_folder, sim_name, control_planes=None, tracers=None):
    """For a given simulation, calculate the flux of tracer across a series
    of control planes.

    Parameters
    ----------
    sim_folder : str or pathlib.Path
        The Path to the folder containing the simulation data
    sim_name : str
        Name of the simulation file
    control_planes : list of int, optional
        Depths, in cm, at which to calculate tracer flux
    tracers : list of str, optional
        List of tracer names to calculate tracer flux

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the specific discharge, tracer concentration,
        and tracer flux across each control plane"""

    if control_planes is None:
        control_planes = [50, 100, 200, 300]
    if tracers is None:
        tracers = ['psi01', 'psi02', 'psi03']

    # Read in time-varying pressure and concentration files
    gbp, gbp_cols, grid_cells = read_min3p_sequence(f'{sim_name}_1.gbp', folder=sim_folder, ftype='transient')
    gbt, gbt_cols, _ = read_min3p_sequence(f'{sim_name}_1.gbt', folder=sim_folder, ftype='transient')

    # Instantiate output dataframe
    cols = ['time']
    for c in control_planes:
        cols.append(f'q_{c}cm_m.d')
        for t in tracers:
            cols.append(f'{t}_conc_{c}cm_mol.L')
            cols.append(f'{t}_flux_{c}cm_mol.m2.d')
    df = pd.DataFrame(index=range(gbp.shape[-1]), columns=cols, dtype=float)
    df['time'] = gbp[0, gbt_cols.index('time')]

    for control_plane in control_planes:
        # Get indices of the grid cells above and below this control plane
        ci_up = list(grid_cells).index(401 - control_plane)
        ci_dn = list(grid_cells).index(400 - control_plane)
        # Get elevations of the grid cells above and below this control plane
        elev_up = 4.01 - control_plane / 100  # Elevation of the upper grid cell in m
        elev_dn = 4.00 - control_plane / 100  # Elevation of the lower grid cell in m
        q = calc_specific_discharge(gbp, input_file=sim_folder / f'{sim_name}.dat',
                                    ci_a=ci_up, ci_b=ci_dn, elev_a=elev_up, elev_b=elev_dn)
        df[f'q_{control_plane}cm_m.d'] = q

        # Calculate an upstream mask for choosing inflow concentration
        upstream_mask = gbp[ci_up, gbp_cols.index('h_w')] > gbp[ci_dn, gbp_cols.index('h_w')]

        for tracer in tracers:
            # Get inflow concentration
            c_up = gbt[ci_up, gbt_cols.index(tracer), 1:]  # `1:` to remove 0th time step (initial conditions)
            c_dn = gbt[ci_dn, gbt_cols.index(tracer), 1:]
            c = np.empty_like(q)
            c[upstream_mask] = c_up[upstream_mask]
            c[~upstream_mask] = c_dn[~upstream_mask]
            df[f'{tracer}_conc_{control_plane}cm_mol.L'] = c

            # For consistency with q (m/d), convert c from mol/L to mol/m3
            c *= 1000
            df[f'{tracer}_flux_{control_plane}cm_mol.m2.d'] = -q * c

    return df


def update_physical_params(infile, site, soil_params, root_params, dualperm=False):
    """Update the physical parameters (porosity, hydraulic conductivity,
    van Genuchten parameters, root water uptake, etc.) for each soil horizon."""
    y_start = '0.05' if dualperm else '0.0'
    idx_start = 3 if dualperm else 0  # Zone index

    ### First, adjust physical parameters
    for i, horizon in enumerate(['A', 'B', 'C']):
        ks = soil_params.loc[(site, horizon), 'Ks_m.d']
        theta_r = soil_params.loc[(site, horizon), 'theta_r']
        theta_s = soil_params.loc[(site, horizon), 'theta_s']
        alpha = soil_params.loc[(site, horizon), 'alpha_m']
        n = soil_params.loc[(site, horizon), 'n']

        # Get individual zones within each block
        pppm = infile.physical_parameters_porous_medium.zones[idx_start+i]
        ppvs = infile.physical_parameters_vsflow.zones[idx_start+i]

        # Set porosity
        pppm.porosity = f'{theta_s:.4f}'

        # Set Ksat and van genuchten parameters
        ks_ms = ks / 60 / 60 / 24  # Convert from m/d to m/s
        ppvs.hydraulic_conductivity_z = single_to_double_float(f'{ks_ms:.3e}')

        # Set van genuchten parameters
        shfp = ppvs.soil_hydraulic_function_parameters
        res_sat = theta_r / theta_s
        shfp.residual_saturation = f'{res_sat:0.4f}'
        shfp.alpha = f'{alpha:0.4f}'
        shfp.n = f'{n:0.4f}'

        # Define root water uptake function
        rwu = ppvs.root_water_uptake
        rwu.saturation_wilting_point = f'{root_params.loc[(site, horizon), 'satwlim']:0.4f}'
        rwu.saturation_field_capacity = f'{root_params.loc[(site, horizon), 'satwfield']:0.4f}'
        rwu.rew0 = f'{root_params.loc[(site, horizon), 'rew0']:0.4f}'
        rwu.p1 = f'{root_params.loc[(site, horizon), 'p1']:0.4f}'

        # Horizon thicknesses
        a_bot = 4.0 - soil_params.loc[(site, 'A'), 'bottom_m']
        b_bot = 4.0 - soil_params.loc[(site, 'B'), 'bottom_m']

        # Yolo doesn't have a C horizon
        if site == 'Yolo':
            b_bot = a_bot
        # Pullman b horizon extends well beyond 5 m
        elif site == 'Pullman':
            b_bot = 0.00
        if horizon == 'A':
            pppm.extent_of_zone = f'0.0 1.0  {y_start} 1.0  {a_bot:0.2f} 4.00'
        elif horizon == 'B':
            pppm.extent_of_zone = f'0.0 1.0  {y_start} 1.0  {b_bot:0.2f} {a_bot:0.2f}'
        elif horizon == 'C':
            pppm.extent_of_zone = f'0.0 1.0  {y_start} 1.0  0.00 {b_bot:0.2f}'

    # Delete C horizon for Pullman and B horizon for Yolo
    delete_from_blocks = [
            "physical_parameters_porous_medium",
            "physical_parameters_vsflow",
            "physical_parameters_reactive_transport"
        ]
    if site == 'Yolo':
        infile.delete_zone('B horizon', block_names=delete_from_blocks)
    elif site == 'Pullman':
        infile.delete_zone('C horizon', block_names=delete_from_blocks)
