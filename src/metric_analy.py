import pickle

from visualization.metric_vis import *

# with open('data/data_pickle/pw_rw_rs_k13510_wosplit/metric_pr_wosplit.data', 'rb') as file:
#     metrics_dict_pr_wo = pickle.load(file)
# file.close()

# with open('data/data_pickle/pw_rw_rs_k13510_wosplit/metric_rw_wosplit.data', 'rb') as file:
#     metrics_dict_rw_wo = pickle.load(file)
# file.close()

# with open('data/data_pickle/pw_rw_rs_k13510_wosplit/metric_rs_wosplit.data', 'rb') as file:
#     metrics_dict_rs_wo = pickle.load(file)
# file.close()

with open('data/metric_dwug_wosplit.data', 'rb') as file:
    metrics_dict_dwug_wo = pickle.load(file)
file.close()

# with open('data/data_pickle/pw_rw_rs_k13510_wsplit/metric_pr_wsplit.data', 'rb') as file:
#     metrics_dict_pr_w = pickle.load(file)
# file.close()

# with open('data/data_pickle/pw_rw_rs_k13510_wsplit/metric_rw_wsplit.data', 'rb') as file:
#     metrics_dict_rw_w = pickle.load(file)
# file.close()

# with open('data/data_pickle/pw_rw_rs_k13510_wsplit/metric_rs_wsplit.data', 'rb') as file:
#     metrics_dict_rs_w = pickle.load(file)
# file.close()

# with open('data/data_pickle/pw_rw_rs_k13510_wsplit/metric_dwug_wsplit.data', 'rb') as file:
#     metrics_dict_dwug_w = pickle.load(file)
# file.close()

# sampling = {'metric1': {'k1': [], 'k2':[], ...}, }
judgements = [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]
com = [1, 3, 5, 10]

def three_metric_ploter(title, judgements, com, metric_dic, legend_flag):
    for k, v in metric_dic.items():
        for _k, _v in v.items():
            _v.extend([_v[-1]] * (len(judgements) - len(_v))) 

    threed_line_ploter(judgements, '#Judgements', com, '#Communities', 'Performance', title, legend_flag=True, 
        adj_randIndex=metric_dic['adjusted_randIndex'], purity=metric_dic['purity'], accuracy=metric_dic['accuracy'], jensenshannon=metric_dic['inverse_jensen_shannon_distance'])

def line_plot_metric(judgements, com, title: str, x_label: str, y_label: str, metric_dic = None):
    for k, v in metric_dic.items():
        for _k, _v in v.items():
            _v.extend([_v[-1]] * (len(judgements) - len(_v))) 

    for kom in com:
        for k, v in metric_dic.items():
            line_ploter(judgements, '{} - k = {}'.format(title, kom), x_label, y_label, save_flag=True, save_path='data/figs/other/tmp/{}_k_{}'.format(title, kom),
            adj_randIndex=metric_dic['adjusted_randIndex'][kom], purity=metric_dic['purity'][kom], accuracy=metric_dic['accuracy'][kom], jensenshannon=metric_dic['inverse_jensen_shannon_distance'][kom])

def bargraph_j_k(labels, kom, index, judgement, title: str, y_label: str, **metric_dics):
    adjusted_randIndex = []
    purity = []
    accuracy = []
    inverse_jensen_shannon_distance = []

    for k, v in metric_dics.items():
        try:
            adjusted_randIndex.append(v['adjusted_randIndex'][kom][index])
        except IndexError as identifier:
            adjusted_randIndex.append(v['adjusted_randIndex'][kom][-1])

        try:
            purity.append(v['purity'][kom][index])
        except IndexError as identifier:
            purity.append(v['purity'][kom][-1])

        try:
            accuracy.append(v['accuracy'][kom][index])
        except IndexError as identifier:
            accuracy.append(v['accuracy'][kom][-1])

        try:
            inverse_jensen_shannon_distance.append(v['inverse_jensen_shannon_distance'][kom][index])
        except IndexError as identifier:
            inverse_jensen_shannon_distance.append(v['inverse_jensen_shannon_distance'][kom][-1])

    bar_metric(labels, '{} for k = {} & Judgements = {}'.format(title, kom, judgement), y_label, True, 
        save_path='data/figs/other/tmp/{}_k_{}_j_{}'.format(title, kom, judgement), 
        adjusted_randIndex=adjusted_randIndex, purity=purity, accuracy=accuracy, inverse_jensen_shannon_distance=inverse_jensen_shannon_distance)

# three_metric_ploter('Simulation PageRank without Split', judgements, com, metrics_dict_pr_wo, True)
# three_metric_ploter('Simulation RandomWalk without Split', judgements, com, metrics_dict_rw_wo, True)
# three_metric_ploter('Simulation RandomSampling without Split', judgements, com, metrics_dict_rs_wo, True)
three_metric_ploter('Simulation DWUG without Split', judgements, com, metrics_dict_dwug_wo, True)

# line_plot_metric(judgements, com, 'DWUG without Split', 'Judgements', 'Performance', metrics_dict_dwug_wo)
# line_plot_metric(judgements, com, 'DWUG with Split', 'Judgements', 'Performance', metrics_dict_dwug_w)


# line_plot_metric(judgements, com, 'PageRank without Split', 'Judgements', 'Performance', metrics_dict_pr_wo)
# line_plot_metric(judgements, com, 'PageRank with Split', 'Judgements', 'Performance', metrics_dict_pr_w)


# line_plot_metric(judgements, com, 'RandomWalk without Split', 'Judgements', 'Performance', metrics_dict_rw_wo)
# line_plot_metric(judgements, com, 'RandomWalk with Split', 'Judgements', 'Performance', metrics_dict_rw_w)


# line_plot_metric(judgements, com, 'RandomSampling without Split', 'Judgements', 'Performance', metrics_dict_rs_wo)
# line_plot_metric(judgements, com, 'RandomSampling with Split', 'Judgements', 'Performance', metrics_dict_rs_w)

# for k in com:
#     for i, judgement in enumerate(judgements):
#         bargraph_j_k(['PageRank wS', 'PageRank woS', 'RandomWalk wS', 'RandomWalk woS'], k, i, judgement, 'Performance', 'Performance',
#         prs=metrics_dict_pr_w, pr=metrics_dict_pr_wo, rws=metrics_dict_rw_w, rw=metrics_dict_rw_wo)