import matplotlib.pyplot as plt
import os

# Data from the ablation on |W| table
data_w = {
    'w': [1, 5, 10, 15, 20],
    'sst2_aua': [75.76, 68.69, 63.64, 65.66, 66.67],
    'sst2_asr': [11.76, 20.0, 27.59, 23.53, 22.35],
    'ag_news_aua': [61.0, 57.0, 50.0, 49.0, 47.0],
    'ag_news_asr': [8.96, 17.39, 27.54, 28.99, 33.8],
    'strategyqa_aua': [48.0, 31.0, 32.0, 32.0, 32.0],
    'strategyqa_asr': [23.81, 52.31, 50.0, 50.0, 50.0],
    'sst2_succ_att_queries_avg': [12.40, 28.52, 31.70, 37.55, 36.73],
    'sst2_total_attack_time': ['001:49:00', '003:23:17', '004:14:46', '004:22:39', '004:12:58'],
    'ag_news_succ_att_queries_avg': [20.00, 38.50, 72.73, 99.50, 113.58],
    'ag_news_total_attack_time': ['001:59:58', '005:47:23', '010:15:10', '014:17:19', '018:35:28'],
    'strategyqa_succ_att_queries_avg': [8.53, 11.35, 11.78, 11.78, 11.78],
    'strategyqa_total_attack_time': ['000:09:21', '000:20:29', '000:20:49', '000:20:49', '000:20:49']
}

# Data from the ablation on |S| table
data_s = {
    's': [1, 5, 10, 20, 50],
    'sst2_aua': [78.79, 69.7, 68.69, 63.64, 58.59],
    'sst2_asr': [8.24, 20.69, 20.0, 25.88, 32.56],
    'ag_news_aua': [62.0, 57.0, 57.0, 52.0, 47.0],
    'ag_news_asr': [10.14, 16.18, 17.39, 24.64, 30.88],
    'strategyqa_aua': [55.0, 39.0, 31.0, 26.0, 24.0],
    'strategyqa_asr': [12.7, 39.06, 52.31, 60.0, 61.9],
    'sst2_succ_att_queries_avg': [4.71, 16.11, 28.52, 49.36, 78.85],
    'sst2_total_attack_time': ['000:44:32', '002:11:18', '003:22:45', '005:19:58', '009:30:16'],
    'ag_news_succ_att_queries_avg': [5.71, 23.45, 38.5, 75.35, 130.19],
    'ag_news_total_attack_time': ['000:48:54', '003:02:26', '005:47:23', '009:41:50', '016:30:55'],
    'strategyqa_succ_att_queries_avg': [2.75, 6.68, 11.35, 18.66, 35.74],
    'strategyqa_total_attack_time': ['000:05:25', '000:12:03', '000:20:29', '000:32:50', '000:58:51']
}

# Helper function to convert 'HHH:MM:SS' to total hours
def convert_time_to_hours(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600

data_w['sst2_total_attack_time_hours'] = [convert_time_to_hours(t) for t in data_w['sst2_total_attack_time']]
data_w['ag_news_total_attack_time_hours'] = [convert_time_to_hours(t) for t in data_w['ag_news_total_attack_time']]
data_w['strategyqa_total_attack_time_hours'] = [convert_time_to_hours(t) for t in data_w['strategyqa_total_attack_time']]
data_s['sst2_total_attack_time_hours'] = [convert_time_to_hours(t) for t in data_s['sst2_total_attack_time']]
data_s['ag_news_total_attack_time_hours'] = [convert_time_to_hours(t) for t in data_s['ag_news_total_attack_time']]
data_s['strategyqa_total_attack_time_hours'] = [convert_time_to_hours(t) for t in data_s['strategyqa_total_attack_time']]

# Set the default font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Garamond', 'Palatino', 'Charter', 'serif']

# Create directory for plots if not exists
directory = 'table_plots'
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to plot data
def plot_data(x, y1, y2, y3, xlabel, ylabel, title, file_name):

    fig, ax1 = plt.subplots(figsize=(15, 15))

    ax1.set_xlabel(xlabel, fontsize=35)
    ax1.set_ylabel(ylabel, fontsize=35)
    ax1.plot(x, y1, 'o-', color='tab:red', label='SST2')
    ax1.plot(x, y2, 's-', color='tab:orange', label='AG-News')
    ax1.plot(x, y3, '^-', color='tab:purple', label='StrategyQA')
    ax1.tick_params(axis='y', labelsize=25)
    ax1.tick_params(axis='x', labelsize=25)

    plt.title(title, fontsize=45, pad=25)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=3, prop={'size': 35})

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.85, bottom=0.2)

    file_path_png = os.path.join(directory, f'{file_name}.png')
    file_path_pdf = os.path.join(directory, f'{file_name}.pdf')
    plt.savefig(file_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(file_path_pdf, format='pdf', dpi=300, bbox_inches='tight')

    plt.show()

# Plot for ablation on |W|
plot_data(data_w['w'], data_w['sst2_aua'], data_w['ag_news_aua'], data_w['strategyqa_aua'], '|W|', 'AUA [%]', 'Ablation on |W| - AUA', 'ablation_ceattack_plot_w_aua')
plot_data(data_w['w'], data_w['sst2_asr'], data_w['ag_news_asr'], data_w['strategyqa_asr'], '|W|', 'ASR [%]', 'Ablation on |W| - ASR', 'ablation_ceattack_plot_w_asr')
plot_data(data_w['w'], data_w['sst2_succ_att_queries_avg'], data_w['ag_news_succ_att_queries_avg'], data_w['strategyqa_succ_att_queries_avg'], '|W|', 'Succ Att Queries Avg', 'Ablation on |W| - Succ Att Queries Avg', 'ablation_ceattack_plot_w_succ_att_queries_avg')
plot_data(data_w['w'], data_w['sst2_total_attack_time_hours'], data_w['ag_news_total_attack_time_hours'], data_w['strategyqa_total_attack_time_hours'], '|W|', 'Total Attack Time [Hours]', 'Ablation on |W| - Total Attack Time [Hours]', 'ablation_ceattack_plot_w_total_attack_time')

# Plot for ablation on |S|
plot_data(data_s['s'], data_s['sst2_aua'], data_s['ag_news_aua'], data_s['strategyqa_aua'], '|S|', 'AUA [%]', 'Ablation on |S| - AUA', 'ablation_ceattack_plot_s_aua')
plot_data(data_s['s'], data_s['sst2_asr'], data_s['ag_news_asr'], data_s['strategyqa_asr'], '|S|', 'ASR [%]', 'Ablation on |S| - ASR', 'ablation_ceattack_plot_s_asr')
plot_data(data_s['s'], data_s['sst2_succ_att_queries_avg'], data_s['ag_news_succ_att_queries_avg'], data_s['strategyqa_succ_att_queries_avg'], '|S|', 'Succ Att Queries Avg', 'Ablation on |S| - Succ Att Queries Avg', 'ablation_ceattack_plot_s_succ_att_queries_avg')
plot_data(data_s['s'], data_s['sst2_total_attack_time_hours'], data_s['ag_news_total_attack_time_hours'], data_s['strategyqa_total_attack_time_hours'], '|S|', 'Total Attack Time [Hours]', 'Ablation on |S| - Total Attack Time [Hours]', 'ablation_ceattack_plot_s_total_attack_time')

print(f'Plots saved to {directory}')