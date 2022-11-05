from MDD.model.MDD import MDD_fake
from CDAN import network as cdan_network

from AFSR.data_provider import load_images
from AFSR import center, attack_generator as attack, lr_schedule, robust_network
import random
import os.path as osp
import numpy as np
from sklearn.metrics import confusion_matrix

import pickle
import os
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torch

AFSR_config = {
    'train': True,
    'epsilon': 0.031,
    'num_steps': 20,
    'step_size': 0.031 / 4,
    'rand_init': True,
    'loss_fn': "cent"
}


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def get_ae_AFSR(aux_net, criterion_cent, inputs_target, pseudo_labels):
    fea_nat, logits_pred_nat, softmax_outputs_tgt = aux_net(inputs_target, multi_outputs=True)
    if pseudo_labels is None:
        pred = softmax_outputs_tgt.data.detach().max(1, keepdim=False)[1]
        pseudo_labels = Variable(pred)

    x = inputs_target.detach()
    x_org = x.detach()
    x = x + torch.zeros_like(x).uniform_(-AFSR_config['epsilon'], AFSR_config['epsilon'])
    x = torch.clamp(x, 0.0, 1.0)
    x.requires_grad_()
    zero_gradients(x)
    x_2 = x
    if x.grad is not None:
        x.grad.data.fill_(0)
    step_size = 0.4
    num_steps = 3
    # AFSR
    for ii in range(num_steps):
        ####
        fea_2, logits_pred_2, _ = aux_net(x_2, multi_outputs=True)
        fea_sum_2 = logits_pred_2 - logits_pred_nat
        loss_cent = criterion_cent(fea_sum_2, pseudo_labels)
        aux_net.zero_grad()
        adv_loss_2 = loss_cent
        adv_loss_2.backward(retain_graph=True)
        x_adv_2 = x_2.data + step_size * AFSR_config['epsilon'] * torch.sign(x_2.grad.data)
        x_adv_2 = torch.min(torch.max(x_adv_2, inputs_target.detach() - AFSR_config['epsilon']),
                            inputs_target.detach() + AFSR_config['epsilon'])
        x_adv_2 = torch.clamp(x_adv_2, 0.0, 1.0)
        x_2 = Variable(x_adv_2)
        x_2.requires_grad_()
        zero_gradients(x_2)
    return x_2, x_org, pseudo_labels


def AFSR_forward(basic_net, criterion_cent, pseudo_labels, x_2, x_org):
    # AFSR
    fea_2, logits_pred_out_2, _ = basic_net(x_2, multi_outputs=True)
    # NAT
    fea_org, logits_pred_out_org, _ = basic_net(x_org, multi_outputs=True)
    # loss
    fea_sum = logits_pred_out_2 - logits_pred_out_org
    loss_cent = criterion_cent(fea_sum, pseudo_labels, reduction='sum')

    return loss_cent, fea_2, logits_pred_out_2, fea_org, logits_pred_out_org


def get_ae_pgd7(aux_net, inputs_target, pseudo_labels):
    with torch.enable_grad():
        x_adv = attack.pgd(aux_net, inputs_target, pseudo_labels, epsilon=0.031, step_size=0.031 / 4, num_steps=7,
                           loss_fn="cent", category="Madry", rand_init=True)
    return x_adv


def all_seed(manual_seed=None):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))

    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ruda_config(config):
    config["batch_size_ae"] = 32
    config["batch_size_train"] = 16
    config["batch_size_nat"] = 128

    if config["train_with_AFSR"]:
        config["at"] = config["at"] + '_AFSR'
        config["batch_size_ae"] = 32
        config["batch_size_train"] = 16
        config["batch_size_nat"] = 128

    config["lr_decay_rate"] = 1  # or None
    config["decay_epoch_1"] = 5
    config["decay_epoch_2"] = 10
    config["epoch"] = 0
    config["best_acc"] = 0
    config["num_iterations"] = 1000000000
    config["max_epochs"] = 100000000
    config["is_multi_output"] = False

    config["dataset_cv"] = config["dataset"] + '_' + config["random"]
    config["task_name"] = config["source"][0].upper() + config["target"][0].upper() + '_fold_' + config["fold"]

    datalist_path_cv = osp.join(config["data_prefix"], 'data', '3_fold', config["dataset_cv"])
    datalist_path = osp.join(config["data_prefix"], 'data', config["dataset"])

    config["s_data_list"] = config["source"] + config["suffix"]
    config["t_data_list"] = config["target"] + config["suffix"]
    config["s_data_list_path"] = osp.join(datalist_path, config["s_data_list"])
    config["t_data_list_path"] = osp.join(datalist_path, config["t_data_list"])

    config["t_data_list_train"] = config["target"] + '_train_' + config["fold"] + config["suffix"]
    config["t_data_list_test"] = config["target"] + '_test_' + config["fold"] + config["suffix"]
    config["t_data_list_train_path"] = osp.join(datalist_path_cv, config["t_data_list_train"])
    config["t_data_list_test_path"] = osp.join(datalist_path_cv, config["t_data_list_test"])

    # config["script_name"] = os.path.basename(sys.argv[0]) + '_' + str(time.time())
    output_path_Gener_UDA = os.path.join(config["project_output_path"], config["dataset_cv"], config["task_name"],
                                         config["model"])
    output_path_Robust_Gener_UDA = os.path.join(config["project_output_path"], config["dataset_cv"],
                                                config["task_name"],
                                                config["model"] + '_' + config["at"])

    config["output_path"] = output_path_Robust_Gener_UDA
    config["reference_model_path"] = os.path.join(output_path_Gener_UDA, 'best_acc_model.pth.tar')

    config["resume_path"] = config["imagenet_linf_8"]   # pre-trained robust imagenet
    # config["resume_path"] = os.path.join(config["output_path"], 'best_acc_model.pth.tar')  # latest best model

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])

    config["out_file"] = open(osp.join(config["output_path"], config["task_name"] + "_log.txt"), "w")
    config["out_file"].write(str(config))
    config["out_file"].flush()

    print(str(config))

    return config


def get_MDD(config):
    if config["dataset"] == 'office-31':
        class_num = 31
        width = 1024
        srcweight = 4
        is_cen = False
    elif config["dataset"] == 'office-home':
        class_num = 65
        width = 2048
        srcweight = 2
        is_cen = False
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    elif config["dataset"] == 'visda-2017':
        class_num = 12
        width = 2048
        srcweight = 4
        is_cen = False
    else:
        width = -1

    model_instance = MDD_fake(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)
    model_instance.c_net.load_state_dict(torch.load(config["reference_model_path"]))
    reference_model = model_instance.c_net

    config["classifier"] = model_instance.c_net.classifier_layer
    config["model_instance"] = model_instance

    config["output_index"] = 1
    config["bottleneck_dim"] = width
    config["width"] = width
    config["reference_model"] = torch.nn.DataParallel(reference_model).cuda()

    return config


def get_DANN(config):
    net_config = {"name": cdan_network.ResNetFc,
                  "params": {"resnet_name": "ResNet50", "use_bottleneck": True, "bottleneck_dim": 256,
                             "new_cls": True, "class_num": config["class_num"]}}
    base_network = net_config["name"](**net_config["params"])
    base_network.load_state_dict(torch.load(config["reference_model_path"]))
    reference_model = base_network

    ad_net_mada_parameters = []
    ad_net_mada = []
    # for _ in range(config["class_num"]):
    #     d = robust_network.AdversarialNetwork_MADA(256, 1024)
    #     p = d.parameter_list
    #     ad_net_mada_parameters.extend(p)
    #
    #     d = torch.nn.DataParallel(d).cuda()
    #     ad_net_mada.append(d)
    config["other_parameters"] = ad_net_mada_parameters
    config["ad_net"] = ad_net_mada

    config["classifier"] = base_network.fc
    config["output_index"] = 1
    config["bottleneck_dim"] = 256
    config["width"] = 256
    config["reference_model"] = torch.nn.DataParallel(reference_model).cuda()

    return config


def loaders_init(config):
    num_worker = config["num_worker"]
    config["s_train_loader"] = load_images(config["s_data_list_path"], batch_size=config["batch_size_train"],
                                           prefix=config["data_prefix"], is_train=True, num_worker=num_worker)
    config["s_train_loader_eval"] = load_images(config["s_data_list_path"], batch_size=config["batch_size_nat"],
                                                prefix=config["data_prefix"], is_train=False, num_worker=num_worker)

    config["t_train_loader"] = load_images(config["t_data_list_train_path"], batch_size=config["batch_size_train"],
                                           prefix=config["data_prefix"], is_train=True, num_worker=num_worker)

    config["t_train_loader_eval"] = load_images(config["t_data_list_train_path"], batch_size=config["batch_size_nat"],
                                                prefix=config["data_prefix"], is_train=False, num_worker=num_worker)
    config["t_test_loader_eval"] = load_images(config["t_data_list_test_path"], batch_size=config["batch_size_nat"],
                                               prefix=config["data_prefix"], is_train=False, num_worker=num_worker)

    config["t_train_loader_eval_ae"] = load_images(config["t_data_list_train_path"], batch_size=config["batch_size_ae"],
                                                   prefix=config["data_prefix"], is_train=False, num_worker=num_worker)
    config["t_test_loader_eval_ae"] = load_images(config["t_data_list_test_path"], batch_size=config["batch_size_ae"],
                                                  prefix=config["data_prefix"], is_train=False, num_worker=num_worker)
    return config


def get_classification_model_and_optimizer_twins(config):
    basic_net_sd = robust_network.RobustBasicNet(base_net='ResNet50Fc_Robust',
                                                 resume_path=config["resume_path"],
                                                 use_bottleneck=True,
                                                 bottleneck_dim=config["bottleneck_dim"], width=config["width"],
                                                 class_num=config["class_num"])
    basic_net_sd.classifier_layer = config["classifier"]

    basic_net_td = robust_network.RobustBasicNet(base_net='ResNet50Fc_Robust',
                                                 resume_path=config["resume_path"],
                                                 use_bottleneck=True,
                                                 bottleneck_dim=config["bottleneck_dim"], width=config["width"],
                                                 class_num=config["class_num"])

    basic_net_td.classifier_layer = config["classifier"]
    param_group = []

    if config["paradigm"] == 'uda':
        param_group.extend(basic_net_sd.get_parameters(with_c=False))
        param_group.extend(basic_net_td.get_parameters(with_c=False))

        config["optimizer"] = {"type": torch.optim.SGD,
                               "optim_params": {'lr': 0.001, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
        optimizer_config = config["optimizer"]
        optimizer = optimizer_config["type"](param_group, **(optimizer_config["optim_params"]))
        lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

        config["lr_scheduler"] = lr_scheduler
        config["schedule_param"] = optimizer_config["lr_param"]

    elif config["paradigm"] == 'suda':
        for k, v in basic_net_sd.base_network.named_parameters():
            param_group += [{'params': v, 'lr': 1e-5}]
        for k, v in basic_net_sd.bottleneck_layer.named_parameters():
            param_group += [{'params': v, 'lr': 1e-3}]
        for k, v in basic_net_td.base_network.named_parameters():
            param_group += [{'params': v, 'lr': 1e-5}]
        for k, v in basic_net_td.bottleneck_layer.named_parameters():
            param_group += [{'params': v, 'lr': 1e-3}]
        optimizer = torch.optim.Adam(param_group, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        optimizer = op_copy(optimizer)
    ##########################################################

    basic_net_sd = torch.nn.DataParallel(basic_net_sd).cuda()
    basic_net_td = torch.nn.DataParallel(basic_net_td).cuda()
    config["basic_net_sd"] = basic_net_sd
    config["basic_net_td"] = basic_net_td
    config["optimizer"] = optimizer

    return config


def eval_reference_model(config):
    reference_model = config["reference_model"]
    reference_model.eval()
    acc_tgt_tr, _, all_pseudo_labels, all_labels, all_output = eval_with_nat(config["t_train_loader_eval"],
                                                                             reference_model,
                                                                             output_index=config["output_index"],
                                                                             flag=False)
    acc_tgt_te, _, _, _, _ = eval_with_nat(config["t_test_loader_eval"], reference_model,
                                           output_index=config["output_index"],
                                           flag=False)
    log_str = "| Train Acc: %.2f | Test Acc: %.2f |" % (acc_tgt_tr, acc_tgt_te)
    config["out_file"].write(str(log_str))
    config["out_file"].flush()
    print(log_str)

    config["all_pseudo_labels"] = all_pseudo_labels.cuda()
    config["all_output"] = all_output.cuda()
    config["all_labels"] = all_labels

    return config


def eval_with_nat(loader, basic_net, output_index=None, flag=False, random_type=None, random_ratio=0.0):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            forwards = basic_net(inputs)

            if output_index is not None:
                outputs = forwards[output_index]
            else:
                outputs = forwards

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    all_pl = torch.squeeze(predict)
    mean_ent = 0
    # mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    accuracy = torch.sum(all_pl.float() == all_label).item() / float(all_label.size()[0])

    if random_type is not None:
        print("Original acc: ", accuracy * 100)

        if random_type == 'incorrect':
            incorrect_index = (all_pl.float() != all_label).numpy()
            index_sele = np.where(incorrect_index == 1)[0]
        elif random_type == 'random':
            index_sele = np.random.choice(range(len(all_pl)), int(len(all_pl) * random_ratio))

        ran_label = torch.from_numpy(np.random.randint(0, 65, (len(index_sele),), dtype=np.int))
        all_pl[index_sele] = ran_label
        accuracy = torch.sum(all_pl.float() == all_label).item() / float(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, all_pl
    else:
        return accuracy * 100, mean_ent, all_pl, all_label, all_output


def train_twin(config):
    basic_net_sd = config["basic_net_sd"]
    basic_net_td = config["basic_net_td"]
    optimizer = config["optimizer"]

    if config["paradigm"] == 'uda':
        lr_scheduler = config["lr_scheduler"]
        schedule_param = config["schedule_param"]

    all_pseudo_labels = config["all_pseudo_labels"]
    all_output = config["all_output"]
    all_labels = config["all_labels"]

    best_acc = config["best_acc"]
    epoch = config["epoch"]

    train_acc = 0
    test_acc = 0
    fgsm_acc = 0
    pgd20_acc = 0
    cw_acc = 0

    # AFSR
    in_features = config["bottleneck_dim"]
    num_classes = config["class_num"]
    criterion_cent_shift = center.CenterLoss(num_classes=num_classes, feat_dim=num_classes,
                                             use_gpu=True)  # logits space
    optimizer_centloss = torch.optim.SGD(criterion_cent_shift.parameters(), lr=0.5)

    for iter_num in range(config["num_iterations"]):

        try:
            inputs_source, labels_source, _ = iter_source.next()
        except:
            iter_source = iter(config["s_train_loader"])
            inputs_source, labels_source, _ = iter_source.next()

        try:
            inputs_target, labels_target, tar_idx = iter_target.next()
        except:
            iter_target = iter(config["t_train_loader"])
            inputs_target, labels_target, tar_idx = iter_target.next()

            basic_net_td.eval()

            if epoch > 4:
                train_acc, _, _, _, _ = eval_with_nat(config["t_train_loader_eval"], basic_net_td, flag=False)
                test_acc, _, _, _, _ = eval_with_nat(config["t_test_loader_eval"], basic_net_td, flag=False)
                # _, fgsm_acc = attack.eval_robust(basic_net, config["t_test_loader_eval_ae"], perturb_steps=1, epsilon=0.031,
                #                                  step_size=0.031, loss_fn="cent", category="Madry", rand_init=True)
                _, pgd20_acc = attack.eval_robust(basic_net_td, config["t_test_loader_eval_ae"], perturb_steps=20,
                                                  epsilon=0.031,
                                                  step_size=0.031 / 4, loss_fn="cent", category="Madry", rand_init=True)
                # _, cw_acc = attack.eval_robust(basic_net, config["t_test_loader_eval_ae"], perturb_steps=30, epsilon=0.031,
                #                                step_size=0.031 / 4, loss_fn="cw", category="Madry", rand_init=True)

            basic_net_td.train()

            # save model
            if pgd20_acc > best_acc:
                best_acc = pgd20_acc
                state_dict = basic_net_td.module.state_dict()
                torch.save(state_dict, os.path.join(config["output_path"], "best_acc_model.pth.tar"))

            # log
            pl_acc = torch.sum(all_pseudo_labels.cpu().float() == all_labels).item() / float(
                all_labels.size()[0]) * 100
            log_str = 'Task: %s | Epoch: [%d | %d] | PL %.2f | Train %.2f | Test %.2f | FGSM %.2f | PGD20 %.2f | CW %.2f | best %.2f |\n' % (
                config["task_name"],
                epoch,
                int(config["num_iterations"] / len(config["t_train_loader"])) - 1,
                pl_acc,
                train_acc,
                test_acc,
                fgsm_acc * 100,
                pgd20_acc * 100,
                cw_acc * 100,
                best_acc * 100
            )
            config["out_file"].write(str(log_str))
            config["out_file"].flush()
            print(log_str)

            epoch += 1

        if inputs_target.size()[0] == 1:
            print('Input size is ', inputs_target.size()[0])
            continue
        if inputs_source.size()[0] == 1:
            print('Input size is ', inputs_source.size()[0])
            continue

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        pseudo_labels = all_pseudo_labels[tar_idx]

        aux_net_sd = pickle.loads(pickle.dumps(basic_net_sd))
        aux_net_td = pickle.loads(pickle.dumps(basic_net_td))
        aux_net_sd.eval()
        aux_net_td.eval()
        basic_net_sd.train()
        basic_net_td.train()

        if config["paradigm"] == 'uda':
            optimizer = lr_scheduler(optimizer, iter_num, **schedule_param)
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()
        bs_tgt = len(inputs_target)
        bs_src = len(inputs_source)

        ### target  ###
        loss_cent = 0
        if 'AFSR' in config["at"]:
            x_2, x_org, _ = \
                get_ae_AFSR(aux_net_td, criterion_cent_shift, inputs_target, pseudo_labels)
            loss_cent, fea_2, logits_pred_out_2, fea_org, logits_pred_out_org = \
                AFSR_forward(basic_net_td, criterion_cent_shift, pseudo_labels, x_2, x_org)

        # ###################### PGD
        x_adv = get_ae_pgd7(aux_net_td, inputs_target, pseudo_labels)
        fea, logits_pred, _ = basic_net_td(x_adv, multi_outputs=True)
        adv_loss = nn.CrossEntropyLoss(reduction='sum')(logits_pred,
                                                        pseudo_labels)

        # ### source  ###
        loss_cent_src = 0
        if 'AFSR' in config["at"]:
            x_2_src, x_org_src, _ = \
                get_ae_AFSR(aux_net_sd, criterion_cent_shift, inputs_source, labels_source)
            loss_cent_src, fea_2_src, logits_pred_out_2_src, fea_org_src, logits_pred_out_org_src = \
                AFSR_forward(basic_net_sd, criterion_cent_shift, labels_source, x_2_src, x_org_src)

            x_adv_src = get_ae_pgd7(aux_net_sd, inputs_source, labels_source)
            fea_src, logits_pred_src, _ = basic_net_sd(x_adv_src, multi_outputs=True)
            adv_loss_src = nn.CrossEntropyLoss(reduction='sum')(logits_pred_src, labels_source)

        ### loss ###
        adv_loss_sum = 0
        optimizer_centloss.zero_grad()
        optimizer.zero_grad()

        if 'AFSR' in config["at"]:
            loss_center = ((loss_cent / bs_tgt) * config["AFSR_coefficient"]) / 2
            loss_center.backward(retain_graph=True)
            # loss_center.backward()
            optimizer_centloss.zero_grad()
            criterion_cent_shift.zero_grad()

        if 'AFSR' in config["at"]:
            at_loss = (adv_loss_src + adv_loss) / (bs_src + bs_tgt)
        else:
            at_loss = adv_loss / bs_tgt

        loss_center = ((loss_cent_src / bs_src) * config["AFSR_coefficient"]) / 2  ###################
        adv_loss_sum += (at_loss + loss_center)

        adv_loss_sum.backward()
        optimizer.step()
        optimizer_centloss.step()

        iter_num += 1
