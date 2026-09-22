import argparse
from pathlib import Path

import pandas as pd
from byte_util import all_sites
from byte_util.met_transport import calc_tracer_flux


def calc_peak_tracer(parent_folder, outfile=None, sites=None, scenarios=None,
                     control_planes=None, tracers=None, return_stats=False):
    """For a given folder containing multiple MIN3P simulations, calculate the time of
    peak tracer concentrations across all tracers and control planes in each simulation.

    Parameters
    ----------
    parent_folder : str or pathlib.Path
        The path to the folder containing MIN3P simulations.
    outfile : str, optional
        Path to the file to write output to. If not provided, the output will be
        parent_folder.name + '_TracerBreakthrough.csv' in the parent_folder.
    sites : list of str, optional
        List of sites to calculate peak tracer concentrations for. If not provided,
        default to using all sites in byte_util.
    scenarios : list of str, optional
        List of scenarios to calculate peak tracer concentrations for. If not provided,
        default to using hourly, daily, monthly and longterm.
    control_planes : list of int, optional
        List of control planes to calculate peak tracer concentrations for. If not provided,
        default to 50, 100, 200 and 300 cm depth
    tracers : list of str, optional
        List of tracers to calculate peak tracer concentrations for. If not provided,
        default to 'psi01' and 'psi02'
    return_stats : bool, optional
        If True, return the stats dataframe instead of saving to a file. Default is False.
    """

    # Set default arguments
    if not isinstance(parent_folder, Path):
        parent_folder = Path(parent_folder)

    if outfile is None:
        outfile = parent_folder / f'{parent_folder.name}_TracerBreakthrough.csv'

    if sites is None:
        sites = all_sites
    if control_planes is None:
        control_planes = [50, 100, 200, 300]
    if tracers is None:
        tracers = ['psi01', 'psi02']
    if scenarios is None:
        scenarios = ['hourly', 'daily', 'monthly', 'longterm']

    stats = pd.DataFrame(columns=['site', 'scenario', 'tracer', 'control_plane', 'peak_time'])
    stats.set_index(['site', 'scenario', 'tracer', 'control_plane'], inplace=True)

    for site in sites:
        for scenario in scenarios:
            sim_folder = parent_folder / site / scenario
            df = calc_tracer_flux(sim_folder, scenario, control_planes=control_planes,
                                  tracers=tracers)

            cp_str = [col.split('cm')[0].split('_')[-1] for col in df.columns if 'time' not in col]
            control_planes = list({int(cp) for cp in cp_str})
            control_planes.sort()
            tracers = list(set([col.split('_')[0] for col in df.columns if 'conc' in col]))

            for tracer in tracers:
                for cp in control_planes:
                    idx = df[f'{tracer}_conc_{cp}cm_mol.L'].argmax()
                    peak_conc_time = df.loc[idx, 'time']
                    stats.loc[(site, scenario, tracer, cp), 'peak_time'] = peak_conc_time

    if return_stats:
        return stats
    else:
        stats.reset_index(inplace=True)
        stats.to_csv(outfile, index=False)
        print(f'Saved peak tracer concentrations to {outfile}!')

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate peak tracer concentrations across all tracers and control planes")
    parser.add_argument('parent_folder',
                        type=str, help="Path to the folder containing MIN3P simulations")
    parser.add_argument('--outfile', dest='outfile', type=str, help="Path to the output csv file")
    parser.add_argument('--sites', dest='sites', nargs='+',
                        type=str, help="List of sites to calculate peak tracer concentrations for")
    parser.add_argument('--scenarios', dest='scenarios', nargs='+',
                        type=str, help="List of scenarios to calculate peak tracer concentrations for")
    parser.add_argument('--control_planes', dest='control_planes', nargs='+',
                        type=str, help="List of control planes to calculate peak tracer concentrations for")
    parser.add_argument('--tracers', dest='tracers', nargs='+',
                        type=str, help="List of tracers to calculate peak tracer concentrations for")
    args = parser.parse_args()

    calc_peak_tracer(**vars(args))