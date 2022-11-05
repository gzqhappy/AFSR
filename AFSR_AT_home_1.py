import argparse
import os
import AFSR.AFSR_train_utils_AT_office_home_009 as train_utils

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

    config["dataset"] = 'office-home'

    if config["dataset"] == 'office-home':
        config["names"] = ['Art', 'Clipart', 'Product', 'Real_World']
        config["class_num"] = 65
        config["suffix"] = ".txt"

    if config["dataset"] == 'office':
        config["names"] = ['amazon', 'dslr', 'webcam']
        config["class_num"] =31
        config["suffix"] = "_list.txt"

    if config["dataset"] == 'visda-2017':
        config["names"] = ['train', 'validation']
        config["class_num"] =12
        config["suffix"] = "_list.txt"

    config["source"] = 'Art'
    config["target"] = 'Clipart'
    config["model"] = 'DANN'
    config["at"] = 'AT'
    config["random"] = '0'
    config["fold"] = '2'
    config["gpus"] = '1'
    config["paradigm"] = 'uda'
    config["train_with_AFSR"] = True
    config["AFSR_coefficient"] = 0.05


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

    config = train_utils.get_classification_model_and_optimizer(config)

    train_utils.train(config)