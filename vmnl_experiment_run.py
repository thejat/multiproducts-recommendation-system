from estim_experiments.vmnl_experiments import run_vmnl_experiment


data_dir = 'results/vmnl_estim'
variable_datasets = ['yc-items.txt']
traindatafiles = [f'{data_dir}/train_data_{dataset}' for dataset in variable_datasets]
testdatafiles = [f'{data_dir}/test_data_{dataset}' for dataset in variable_datasets]

run_vmnl_experiment(traindatafiles, testdatafiles, 'results/vmnl_estim')