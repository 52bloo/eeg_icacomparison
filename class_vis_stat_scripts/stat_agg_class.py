import scipy
from scipy.stats import stats as scistat
import numpy as np
import pickle



class MetricContainer:
    def __init__(self, container_type=None, container_info=None, cv_result=None):
        self.container_type=container_type

        self.metrics = {}
        self.metric_names = []
        if container_info is None:
            self.container_info = {}
        else:
            self.container_info = container_info
        self.cv_result = cv_result

    def add_metrics(self, metric):
        for key, item in metric.items():
            self.metrics[key] = item
        return

    def add_container_info(self, infod):
        for key, item in infod.items():
            self.container_info[key] = item

    def set_cv_result(self, cv_result):
        self.cv_result = cv_result

    def set_paramsets(self, params_grid, cv_best_param):
        self.param_grid = params_grid
        self.cv_best_param = cv_best_param

class MetricAggregation(MetricContainer):
    def __init__(self, container_type=None, container_info=None, member_type=None):
        super().__init__(container_type=container_type, container_info=container_info)
        self.members = []
        self.member_type = member_type
        self.members_n = 0

    def add_members(self, member, member_type='individual'):
        # at this point it doesn't care whether members are more aggregates or a MetricContainer
        if type(member) is list:
            self.members += member
            self.members_n = len(self.members)
        elif type(member) is MetricAggregation:
            self.members.append(member)
            self.members_n += 1
        elif type(member) is MetricContainer:
            self.members.append(member)
            self.members_n += 1
        # and set member type if it's not set yet
        if self.member_type is None:
            self.member_type = member_type

    def collect_metric_values_for_plotting(self, target_metric, target_member_level=1):
        exit_arr = []
        for member in self.members:
            exit_arr.append(member.metrics[target_metric])
        # collect values in array form for easier plotting
        return exit_arr

    def collect_cvres_value_for_plotting(self, target_value):
        exit_arr = []
        for member in self.members:
            exit_arr.append(member.cv_result[target_value])

        return exit_arr

    def collect_cvres(self):
        exit_arr = []
        for member in self.members:
            exit_arr.append(member.cv_result)
        return exit_arr

    def run_metric_stats(self, target_metrics):

        for target_metric in target_metrics:
            member_mets = [mb.metrics[target_metric] for mb in self.members]
            self.metrics[target_metric] = np.mean(member_mets)
            self.metrics[f'median_{target_metric}'] = np.median(member_mets)
            self.metrics[f'std_{target_metric}'] = np.std(member_mets)
            self.metrics[f'sem_{target_metric}'] = scistat.sem(member_mets)

            self.metrics[f'all_{target_metric}'] = member_mets

        return
    def aggregate_container_info(self, target_infos):
        for target_info in target_infos:
            member_infos = [mb.container_info[target_info] for mb in self.members]
            self.container_info[f'all_{target_info}'] = member_infos
        return


# put statistic operations outside of class definition to avoid compatibility breaking between updates of statistics code

def per_parameter_statistics(metset: MetricAggregation, target_param):

    global_param_stats = []
    uniq_param_values = []
    for member in metset.members:
        # apply for each participant
        uniq_param_values = metset.members[0].param_grid[target_param]
        uniq_param_stats = []
        for uniq_param in uniq_param_values:
            uniq_param_loc = np.where(np.asarray(member.cv_result[f'param_{target_param}'])==uniq_param)[0]
            param_val_stats = np.asarray(member.cv_result['mean_test_score'])[uniq_param_loc]
            uniq_param_stats.append(param_val_stats)
        global_param_stats.append(uniq_param_stats)

    return {'param_name': target_param, 'param_values': uniq_param_values, 'param_scores': global_param_stats}