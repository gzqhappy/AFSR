import argparse
import os
import AFSR.AFSR_train_utils_AT_visda_009 as train_utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='./MDD/config/dann.yml')
    parser.add_argument('--dataset', default='office', type=str,
                        help='which dataset')
    args = parser.parse_args()

    config = {}



    config["data_prefix"] = '/Data_SSD/user/path_dataset'
    config["project_output_path"] = '/Data_SSD/user/path_save'
    config["imagenet_linf_8"] = '/Data_SSD/user/path_robust_pretrain/imagenet_linf_8.pt'


    config["dataset"] = 'visda-2017'
    config["names"] = ['train', 'validation']
    config["class_num"] =12
    config["suffix"] = "_list.txt"

    config["source"] = 'train'
    config["target"] = 'validation'
    config["model"] = 'MDD'
    config["at"] = 'AT'
    config["random"] = '0'
    config["fold"] = '0'
    config["gpus"] = '2'
    config["paradigm"] = 'suda'
    config["train_with_AFSR"] = True
    config["AFSR_coefficient"] = 0.09

    num_gpu = len(config["gpus"].split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    config["seed"] = train_utils.all_seed(5388)
    config["num_worker"] = 4 * num_gpu

    config = train_utils.get_ruda_config(config)

    if config["model"] == 'MDD':
        config = train_utils.get_MDD(config)
    elif config["model"] == 'DANN':
        config = train_utils.get_DANN(config)


    config = train_utils.loaders_init(config)

    config = train_utils.eval_reference_model(config)

    config = train_utils.get_classification_model_and_optimizer_twins(config)

    train_utils.train_twin(config)