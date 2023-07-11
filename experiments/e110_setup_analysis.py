from os.path import exists
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fabricatio_controls.heuristics import HeuristicControl
from fabricatio_controls.heuristics import LQT
from fabricatio_controls.heuristics import SPT
from fabricatio_controls.comparison_utils import SchedulingSetup
from commons import load_fjc_env, load_jm_env
from fabricatio_rl.env_utils import create_folders

if TYPE_CHECKING:
    from fabricatio_rl.core_state import State


def get_wip_measurements(
        setup: SchedulingSetup, experiment_seeds: np.ndarray,
        n_steps: int = 100, wip_growth: int = 1):
    utls, ftis, dftis, wips, ts, seed_log = [], [], [], [], [], []
    path = f'./1_data/setup/wip_metrics_{setup.value}.csv'
    create_folders(path)
    if exists(path):
        df = pd.read_csv(path, index_col=0, float_precision='high')
        max_wip_size = df['wip_size'].astype(int).max()
    else:
        max_wip_size = 1
        df = None
    for wip_size in range(max_wip_size + 1, n_steps * wip_growth + 1,
                          wip_growth):
        for seed in experiment_seeds:
            if setup == SchedulingSetup.FLEXIBLEJOBSHOP:
                env = load_fjc_env(seed, wip_size, wip_size, 'balanced',
                                   basedir=f"../../benchmarks/fjssp_all")
            else:
                env = load_jm_env(seed, wip_size, wip_size, 'balanced',
                                  basedir=f"../../benchmarks/jssp_all")
            control = HeuristicControl(SPT(), LQT())
            end_state: 'State' = control.play_game(env)
            utl_ave = (end_state.trackers.utilization_times.mean() /
                       end_state.system_time)
            fti_ave = (end_state.trackers.flow_time.mean() /
                       end_state.system_time)
            duration_relative_fti = (
                    end_state.trackers.initial_durations.sum(axis=1) /
                    end_state.trackers.flow_time).mean()
            print(f"Average flow time and utilization "
                  f"for wip of size {wip_size}: "
                  f"\n\t{fti_ave}, {utl_ave}")
            utls.append(utl_ave)
            ftis.append(fti_ave)
            dftis.append(duration_relative_fti)
            seed_log.append(seed)  # :)
            wips.append(wip_size)
            ts.append(end_state.system_time)
    wip_metrics_new = pd.DataFrame(
        data={
            'wip_size': wips,
            'ave_utilization': utls,
            'cmax_rel_flow_time': ftis,
            'flow_time_rel_d_ave': dftis,
            'seed': seed_log,
            't': ts
        },
        dtype=int
    )
    if df is not None:
        wip_metrics_all = pd.concat([wip_metrics_new, df]).reset_index(
            drop=True)
        wip_metrics_all.to_csv(path)
    else:
        wip_metrics_new.to_csv(path)


np.random.seed(42)
seeds = np.random.randint(1, 10000, size=20)

get_wip_measurements(SchedulingSetup.JOBSHOP, seeds, 100, 2)
get_wip_measurements(SchedulingSetup.FLEXIBLEJOBSHOP, seeds, 100, 2)
