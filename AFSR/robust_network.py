import numpy as np
import torch

from torchvision import models

from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
from robustness.tools.helpers import InputNormalize

from torch import nn


class ResNet50Fc_Robust(nn.Module):
    def __init__(self, resume_path):
        super(ResNet50Fc_Robust, self).__init__()

        if resume_path is None:
            model_resnet50 = models.resnet50(pretrained=True)
        elif 'imagenet' in resume_path:
            # else:
            attacker_model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet(data_path=''),
                                                       resume_path=resume_path, parallel=False)
            model_resnet50 = attacker_model.model

        self.normalizer = InputNormalize(new_mean=torch.tensor([0.485, 0.456, 0.406]),
                                         new_std=torch.tensor([0.229, 0.224, 0.225]))
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1

        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.normalizer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class NewResNetFc(nn.Module):
    def __init__(self):
        super(NewResNetFc, self).__init__()

        self.normalizer = InputNormalize(new_mean=torch.tensor([0.485, 0.456, 0.406]),
                                         new_std=torch.tensor([0.229, 0.224, 0.225]))
        self.feature_layers = None
        self.bottleneck = None
        self.fc = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, multi_outputs=False):
        x = self.normalizer(inputs)
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        y = self.fc(x)

        features = x
        outputs = y

        if multi_outputs:
            softmax_outputs = self.softmax(outputs)
            return features, outputs, softmax_outputs
        else:
            return outputs

    def get_parameters(self, with_c=True):
        if with_c:
            parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                              {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2}, \
                              {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                              {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list


class RobustBasicNet(nn.Module):

    def __init__(self, base_net='ResNet50Fc_Robust', resume_path=None, use_bottleneck=True, bottleneck_dim=1024,
                 width=1024, class_num=31):
        super(RobustBasicNet, self).__init__()

        ## set base network
        self.base_network = ResNet50Fc_Robust(resume_path=resume_path)
        self.use_bottleneck = use_bottleneck
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim),
                                      nn.BatchNorm1d(bottleneck_dim), nn.ReLU(),
                                      nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

    def forward(self, inputs, multi_outputs=False):
        features = self.base_network(inputs)
        features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        if multi_outputs:
            softmax_outputs = self.softmax(outputs)
            return features, outputs, softmax_outputs
        else:
            return outputs

    # def forward(self, x):
    #     x = self.normalizer(x)
    #     x = self.feature_layers(x)
    #     x = x.view(x.size(0), -1)
    #     if self.use_bottleneck and self.new_cls:
    #         x = self.bottleneck(x)
    #     y = self.fc(x)
    #     return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self, with_c=True):
        if with_c:
            parameter_list = [{"params": self.base_network.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                              {"params": self.bottleneck_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}, \
                              {"params": self.classifier_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.base_network.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                              {"params": self.bottleneck_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork_gradient(nn.Module):
    def __init__(self, in_feature, hidden_size, coeff_times=1):
        super(AdversarialNetwork_gradient, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)
        self.coeff_times = coeff_times

    def forward(self, x, iter_time=1):
        if self.training:
            self.iter_num += (1 * iter_time)
        self.coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter) * self.coeff_times
        x = x * 1.0
        x.register_hook(grl_hook(self.coeff))
        x = self.ad_layer1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def DANN(features, ad_net, src_size, tgt_size):
    ad_out = ad_net(features)
    dc_target = torch.from_numpy(np.array([[1]] * src_size + [[0]] * tgt_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, coeff_times=1):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size)
        self.coeff_times = coeff_times

    def forward(self, x, iter_time=1):
        if self.training:
            self.iter_num += (1 * iter_time)
        self.coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter) * self.coeff_times
        x = x * 1.0
        x.register_hook(grl_hook(self.coeff))
        x = self.ad_layer1(x)
        # x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        # x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
