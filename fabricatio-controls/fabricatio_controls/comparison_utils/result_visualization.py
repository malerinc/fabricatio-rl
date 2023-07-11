import pandas as pd
import seaborn as sns
import numpy as np
from typing import List, Dict, Union
from copy import deepcopy


class ResultVisualization:
    def __init__(self, res_data: Union[str, pd.DataFrame, List[str]],
                 index_col: str = 'instance', control_col: str = 'control_name',
                 vbhs=False, palette=None, colors=None):
        # TODO: as arg!!!
        # self.ctrl_csv_to_plt_names = control_csv_to_plot_name_map
        # data info
        self.res_data = ResultVisualization.__prep_result_df(res_data)
        self.n_experiments = len(self.res_data.groupby(index_col).size())
        self.index_col = index_col
        self.cntrl_col = control_col
        self.vbhs = vbhs
        self.palette_string = 'cubehelix' if not palette else palette
        # a call select_data will instantiate this field;
        # no data prep before that!
        self.controls = None
        self.score_col = None
        self.control_renaming = None
        # a call to __prep_data will instantiate these fields;
        # no plotting before that!
        self.data_prepped = False
        self.res_wide, self.res_wide_rel = None, None
        self.res_long, self.res_long_rel = None, None
        self.control_order, self.colors = None, colors

    @staticmethod
    def __prep_result_df(res_data):
        if type(res_data) == str:
            res_data = pd.read_csv(res_data, index_col=0)
        elif type(res_data) == list:
            rdf = []
            for rdata in res_data:
                rdf.append(pd.read_csv(rdata, index_col=0))
            res_data = pd.concat(rdf)
        res_data.rename({
            'throughput_op': 'operation_throughput',
            'throughput_j': 'job_throughput',
        }, axis=1, inplace=True)
        res_data['utilization'] = (res_data['utilization']
                                   / res_data['makespan'])
        res_data['slack_time'] = 1 - res_data['utilization']
        return res_data

    def get_available_controls(self):
        return np.unique(self.res_data[self.cntrl_col])

    @staticmethod
    def get_available_targets():
        return [
            'makespan',
            'operation_throughput',
            'job_throughput',
            'tardiness',
            'utilization',
            'slack_time'
        ]

    def __rename_controls(self, name_dict):
        def rename_control(name):
            if name in name_dict:
                return name_dict[name]
            else:
                return name
        self.res_data[self.cntrl_col] = self.res_data[self.cntrl_col].apply(
            rename_control)

    def select_data(self, target: str = 'makespan',
                    control_names: Dict[str, str] = None,
                    control_subset: List['str'] = None):
        assert type(self.res_data) == pd.DataFrame
        self.score_col = target
        self.control_renaming = control_names
        if control_names:
            self.__rename_controls(name_dict=control_names)
        if not control_subset:
            controls = list(
                set(np.unique(self.res_data[self.cntrl_col]))
                - {self.index_col})
            self.controls = controls
        else:
            controls_in_data = self.get_available_controls()
            for c in control_subset:
                try:
                    assert c in controls_in_data
                except AssertionError:
                    print(f"The requested control  {c} is not available. " +
                          "Please select only one of the following: " +
                          '\n'.join(controls_in_data))
                    return
            self.controls = deepcopy(control_subset)
        self.__prep_data()

    def switch_frame(self, res_data):
        self.res_data = ResultVisualization.__prep_result_df(res_data)
        self.n_experiments = len(self.res_data.groupby(self.index_col).size())

    def __prep_data(self):
        assert self.controls is not None
        self.res_wide = self.__get_wide_df(self.res_data)
        self.res_wide['VBS'] = self.__get_best_control(self.controls)
        self.controls.append('VBS')
        if self.vbhs:
            hs = [
                'SPT', 'LPT', 'LOR', 'MOR', 'SRPT',
                'LRPT', 'LTPO', 'MTPO', 'EDD', 'LUDM'
            ]
            h_names = [h for h in hs if h in self.controls]
            self.res_wide['VBHS'] = self.__get_best_control(h_names)
        self.res_wide_rel = self.__get_relative_scores('VBS', self.controls)
        # normalize
        self.res_long = self.__get_long_df(self.res_wide)
        self.res_long_rel = self.__get_long_df(self.res_wide_rel)
        self.control_order = self.__get_control_order()
        if not self.colors:
            palette_colors = sns.color_palette(
                self.palette_string, len(self.control_order)).as_hex()
            self.colors = {self.control_order.index[i]: palette_colors[i]
                           for i in range(len(self.controls))}
        self.data_prepped = True

    def get_colors(self):
        return self.colors

    def __get_control_order(self):
        order = (self.res_long_rel.groupby(self.cntrl_col)[self.score_col]
                 .mean().sort_values()).reset_index()
        if order.loc[0, self.cntrl_col] != 'VBS':
            order_2nd = order.iloc[0, :]
            order_1st = order.iloc[1, :]
            order.iloc[0, :] = order_1st
            order.iloc[1, :] = order_2nd
        order.index = order[self.cntrl_col]
        order = order.drop(self.cntrl_col, axis=1)
        return order[self.score_col]

    def __get_wide_df(self, res_data):
        res_data = res_data[res_data[self.cntrl_col].isin(self.controls)]
        df_res = res_data.pivot_table(
            index=self.index_col, columns=self.cntrl_col, values=self.score_col)
        df_res = df_res.reset_index()
        df_res.columns.name = ''
        return df_res

    # 1: Virtual Best Selectors
    def __get_long_df(self, res_wide):
        idx = self.index_col
        df_result = pd.melt(res_wide, id_vars=[idx],
                            value_vars=self.controls)
        return df_result.rename(columns={
            'value': self.score_col,
            '': self.cntrl_col}
        )

    def __get_best_control(self, control_subset: List):
        return self.res_wide.apply(lambda row: min(row[control_subset]), axis=1)

    @staticmethod
    def __get_relative_row_scores(row, comparison_cols, baseline):
        s = row[baseline]
        r = row[comparison_cols]
        return np.divide(r + np.ones_like(r),
                         s + np.ones_like(s))

    def __get_relative_scores(self, baseline_control, control_subset):
        df_normalized = self.res_wide.apply(
            lambda r: ResultVisualization.__get_relative_row_scores(
                r, control_subset, baseline_control),
            axis=1
        ).reset_index(drop=True)
        df_normalized[self.index_col] = self.res_wide[self.index_col]
        return df_normalized

    def __get_cumsums(self):
        # reform df to contain one row per experiment;
        # create the cumulative countingdistribution function defined on seeds;
        # make seed index into a column
        # threshhold is the average minimum makespan value over all experiments
        col_names = set(self.res_wide.columns) - {self.index_col}
        thresh = self.res_wide.apply(
            lambda row: row.loc[col_names].mean(), axis=1).mean()
        better_than_thresh_cumsums = self.res_wide.loc[:, col_names].apply(
            lambda x: x <= thresh).cumsum().reset_index()
        better_than_thresh_cumsums['instance'] = self.res_wide['instance']
        # bring the df back into the original form where a row corresponds to
        # a single scheduling episode labeled by the according control
        df_long = pd.melt(better_than_thresh_cumsums, id_vars=[self.index_col],
                          value_vars=list(set(self.res_wide.columns) -
                                          {self.index_col}))
        experiment_numbers = list(range(
            len(np.unique(df_long[self.index_col]))))
        unique_seeds = np.unique(df_long[self.index_col])
        seed_mapping = {
            unique_seeds[i]: experiment_numbers[i] for i in
            range(len(unique_seeds))}
        df_long['Experiment Numbers'] = df_long[self.index_col].map(
            lambda x: seed_mapping[x])
        return df_long.drop(self.index_col, axis=1).rename(
            columns={'': 'control_name'}), thresh

    def plot_control_cumulative_counting_distribution_functions(
            self, title='', filename='fig', ax=None):
        assert self.data_prepped
        data, thresh = self.__get_cumsums()
        ax = sns.lineplot(data=data,
                          x='Experiment Numbers',
                          y='value', hue='control_name', ax=ax,
                          hue_order=self.control_order.index,
                          palette=self.colors)
        ax.set_ylabel(
            f'{ResultVisualization.split_capitalize(self.score_col)} Results'
            f'\nBelow Threshhold {round(thresh, 2)} '
        )
        ax.set_xlabel('Experiment Number')
        ax.set_title(title, loc='left')
        ax.legend(loc=2).texts[0].set_text(
            ResultVisualization.split_capitalize(self.cntrl_col)
        )
        return ax

    @staticmethod
    def _show_on_single_plot(ax, space, h_v):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 1.5
                _y = p.get_y() + p.get_height() * 1.01
                value = round(p.get_height(), 3)
                ax.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = round(p.get_width(), 3)
                ax.text(_x, _y, value, ha="left")

    @staticmethod
    def show_values_on_bars(axs, h_v="v", space=0.4):
        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                ResultVisualization._show_on_single_plot(ax, space, h_v)
        else:
            ResultVisualization._show_on_single_plot(axs, space, h_v)

    @staticmethod
    def split_capitalize(string):
        return ' '.join(map(lambda x: x.capitalize(), string.split("_")))

    def get_results_barplot(self, title='', y_lim=0.8,
                            filename='fig', ax=None):
        assert self.data_prepped
        ax = sns.barplot(data=self.res_long_rel,
                         y=self.score_col, x=self.cntrl_col,
                         order=self.control_order.index,
                         ci=None, estimator=np.mean,
                         palette=self.colors,
                         ax=ax)
        largest_bar_height = self.control_order.max()
        y_lo = self.control_order.min() - 0.1 * largest_bar_height
        # y_hi = bar_values.max() + 0.05 * largest_bar_height
        ax.set(ylim=(y_lo, None))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(ResultVisualization.split_capitalize(self.cntrl_col))
        ax.set_ylabel(
            f'Average VBS Relative '
            f'{ResultVisualization.split_capitalize(self.score_col)}')
        ax.set_title(title, loc='left')
        ResultVisualization.show_values_on_bars(ax)
        return ax

    def get_results_boxplot(self, title='', y_lim=0.8,
                            filename='fig', ax=None):
        assert self.data_prepped
        ax = sns.boxplot(data=self.res_long_rel,
                         y=self.score_col, x=self.cntrl_col,
                         order=self.control_order.index,
                         showmeans=True,
                         # meanline=True,
                         palette=self.colors,
                         ax=ax)
        # y_hi = bar_values.max() + 0.05 * largest_bar_height
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(ResultVisualization.split_capitalize(self.cntrl_col))
        ax.set_ylabel(
            f'VBS Relative '
            f'{ResultVisualization.split_capitalize(self.score_col)}')
        ax.set_title(title, loc='left')
        return ax

    def get_best_k(self, cols_subset, k_best):
        return self.res_wide[cols_subset].mean().T.sort_values().index[0:k_best]

    def __get_cofs(self, threshs, controls_to_compare: List = None):
        """
        Counting objective function (cof)of the different control results in df
        and the threshold values parameter.
        """
        # reform df to contain one row per experiment;
        # create the cumulative counting distribution function defined on
        # instancees;
        # make seed index into a column
        # threshhold is the average minimum score value over all experiments
        results = []
        for thresh in threshs:
            df_thresh = (self.res_wide[self.controls].apply(
                lambda x: x < thresh).sum()
                         / self.res_wide.shape[0])
            df_thresh['v'] = thresh
            results.append(df_thresh)
        df_orpe = pd.DataFrame(results)
        # bring the df back into the original form where a row corresponds to
        # a single scheduling episode labeled by the according control
        df_mlrpe = pd.melt(df_orpe, id_vars=['v'],
                           value_vars=list(set(df_orpe.columns) - {'v'}),
                           var_name=self.cntrl_col, value_name="$F_{ALG(v)}$")
        return df_mlrpe

    def plot_control_cumulative_cdf(self, title='', filename='fig',
                                    controls_to_compare=None,
                                    ax=None):
        assert self.data_prepped
        data = self.__get_cofs(
            np.linspace(self.res_long[self.score_col].min(),
                        self.res_long[self.score_col].max(), 300),
            controls_to_compare=controls_to_compare)
        ax = sns.lineplot(data=data,
                          x='v',
                          y="$F_{ALG(v)}$", hue=self.cntrl_col, ax=ax,
                          legend='brief', hue_order=self.control_order.index,
                          palette=self.colors)
        ax.set_ylabel('Below Threshhold \n Experiment Ratio: $F_{ALG(v)}$')
        ax.set_xlabel(
            f'Threshhold '
            f'{ResultVisualization.split_capitalize(self.score_col)} ($v$)')
        ax.set_title(title, loc='left')
        ax.legend().texts[0].set_text(
            ResultVisualization.split_capitalize(self.cntrl_col)
        )
        return ax

    def count_winns(self, ax=None):
        assert self.data_prepped
        winners = self.res_wide.iloc[:, 1:].idxmin(axis=1)
        winner_order = winners.value_counts().index
        plot_order = []
        for ctrl_name in self.control_order.index:
            if ctrl_name in winner_order:
                plot_order.append(ctrl_name)
        ax = sns.countplot(y=winners, order=plot_order, ax=ax,
                           palette=self.colors)
        ax.set_ylabel(
            ResultVisualization.split_capitalize(self.cntrl_col)
        )
        ax.set_xlabel("Number of Winns")
        return ax
