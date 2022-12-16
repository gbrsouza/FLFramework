import os
from statistics import mean, stdev
import re
from scipy import stats

RESULTS_PATH = "./results/avg/hospital_unb/"

def read_results(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()

    loss, acc, pre, time = [], [], [], []
    for line in lines:
        values = line.split('&')
        loss.append(values[2])
        acc.append(values[3])
        pre.append(values[4])
        time.append(re.sub(r"\\\\ \n","", values[5]))

    loss = [float(x.strip(' "')) for x in loss]
    acc  = [float(x.strip(' "')) for x in acc]
    pre  = [float(x.strip(' "')) for x in pre]
    time = [float(x.strip(' "')) for x in time]

    return {'loss':loss, 'acc':acc, 'pre': pre, 'time':time}

def read_results_by_algorithm(files):
    fedavg, fedavgm, qfedavg, fedadam, fedyogi, centralized = {},{},{},{},{},{}
    for f in files:
        fedalg = f.split('-')[-3]
        if fedalg == 'fedavg':
            fedavg = read_results(os.path.join(RESULTS_PATH, f))
        if fedalg == 'fedavgm':
            fedavgm = read_results(os.path.join(RESULTS_PATH, f))
        if fedalg == 'qfedavg':
            qfedavg = read_results(os.path.join(RESULTS_PATH, f))
        if fedalg == 'fedadam':
            fedadam = read_results(os.path.join(RESULTS_PATH, f))
        if fedalg == 'fedyogi':
            fedyogi = read_results(os.path.join(RESULTS_PATH, f))
        if fedalg == 'centralized':
            centralized = read_results(os.path.join(RESULTS_PATH, f))

    return fedavg, fedavgm, qfedavg, fedadam, fedyogi, centralized


def calculate_metrics(set_data):
    min_value = round(min(set_data),3)
    max_value = round(max(set_data),3)
    mean_value = round(mean(set_data),3)
    stdev_value = round(stdev(set_data),3)

    return {'min': min_value, 'max': max_value, 'mean': mean_value, 'stdev': stdev_value}

def metrics(data):
    loss_m, acc_m, pre_m, time_m = {}, {}, {}, {}

    loss_m = calculate_metrics(data['loss'])
    acc_m = calculate_metrics(data['acc'])
    pre_m = calculate_metrics(data['pre'])
    time_m = calculate_metrics(data['time'])

    return {'loss': loss_m, 'acc': acc_m, 'pre': pre_m, 'time': time_m}


def print_result(metrics):
    print("metr & min & max & avg & stdev")
    print("loss &", metrics['loss']['min'], "&", metrics['loss']['max'], "&", metrics['loss']['mean'], "&", metrics['loss']['stdev'])
    print("acc &", metrics['acc']['min'], "&", metrics['acc']['max'], "&", metrics['acc']['mean'], "&", metrics['acc']['stdev'])
    print("pre &", metrics['pre']['min'], "&", metrics['pre']['max'], "&", metrics['pre']['mean'], "&", metrics['pre']['stdev'])
    print("time &", metrics['time']['min'], "&", metrics['time']['max'], "&", metrics['time']['mean'], "&", metrics['time']['stdev'])

def print_mean(metrics):
    print("loss & acc & pre & time ")
    print(metrics['loss']['mean'], "&", metrics['acc']['mean'], "&", metrics['pre']['mean'], "&", metrics['time']['mean'])

def main() -> None:
    files = os.listdir(RESULTS_PATH)
    cnn_files, vgg_files, eff_files = [], [], []
    
    for f in files:
        for x in f.split('-'):
            if x == 'cnn':
                cnn_files.append(f)
            elif x == 'vgg16':
                vgg_files.append(f)
            elif x ==  'efinet':
                eff_files.append(f)

    # calculate for cnn
    fedavg, fedavgm, qfedavg, fedadam, fedyogi, centralized = read_results_by_algorithm(eff_files)

    # Perform the two sample t-test with equal variances
    # result_avg  = stats.ttest_ind(a=centralized['acc'], b=fedavg['acc'], equal_var=True)
    # result_avgm = stats.ttest_ind(a=centralized['acc'], b=fedavgm['acc'], equal_var=True)
    # result_qavg = stats.ttest_ind(a=centralized['acc'], b=qfedavg['acc'], equal_var=True)
    # result_adam = stats.ttest_ind(a=centralized['acc'], b=fedadam['acc'], equal_var=True)
    # result_yogi = stats.ttest_ind(a=centralized['acc'], b=fedyogi['acc'], equal_var=True)
    # print('fedavg', result_avg)
    # print('fedavgm', result_avgm)
    # print('qfedavg', result_qavg)
    # print('fedadam', result_adam)
    # print('fedyogi', result_yogi)

    # calculate metrics 
    m_fedavg = metrics(centralized)

    print_result(m_fedavg)
    # print_mean(m_fedavg)

    # Conduct the Kruskal-Wallis Test 
    # result = stats.kruskal(
    #     centralized['acc'], 
    #     fedavg['acc'],
    #     fedavgm['acc'],
    #     fedadam['acc'],
    #     fedyogi['acc'])   

    # Print the result
    # print(result) 

if __name__ == "__main__":
    main()