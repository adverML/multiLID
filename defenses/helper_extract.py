#!/usr/bin/env python3

import os, sys, pdb
import torch 
import numpy as np



def registrate_whitebox_features(args, model):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if args.model == 'wrn28-10':
        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list

        model.init_conv.register_forward_hook( get_activation('init_conv') )
        model.conv2.register_forward_hook( get_activation('seq_conv2') )
        model.conv3.register_forward_hook( get_activation('seq_conv3') )
        model.conv4.register_forward_hook( get_activation('seq_conv4') )
        
        model.conv2[0].residual[1].register_forward_hook( get_activation('conv2_0_relu_1') )
        model.conv2[0].residual[4].register_forward_hook( get_activation('conv2_0_relu_4') )

        model.conv2[1].residual[1].register_forward_hook( get_activation('conv2_1_relu_1') )
        model.conv2[1].residual[4].register_forward_hook( get_activation('conv2_1_relu_4') )

        model.conv2[2].residual[1].register_forward_hook( get_activation('conv2_2_relu_1') )
        model.conv2[2].residual[4].register_forward_hook( get_activation('conv2_2_relu_4') )

        model.conv2[3].residual[1].register_forward_hook( get_activation('conv2_3_relu_1') )
        model.conv2[3].residual[4].register_forward_hook( get_activation('conv2_3_relu_4') )


        model.conv3[0].residual[1].register_forward_hook( get_activation('conv3_0_relu_1') )
        model.conv3[0].residual[4].register_forward_hook( get_activation('conv3_0_relu_4') )

        # 5
        model.conv3[1].residual[1].register_forward_hook( get_activation('conv3_1_relu_1') )
        model.conv3[1].residual[4].register_forward_hook( get_activation('conv3_1_relu_4') )

        model.conv3[2].residual[1].register_forward_hook( get_activation('conv3_2_relu_1') )
        model.conv3[2].residual[4].register_forward_hook( get_activation('conv3_2_relu_4') )

        # 7
        model.conv3[3].residual[1].register_forward_hook(get_activation('conv3_3_relu_1') )
        model.conv3[3].residual[4].register_forward_hook(get_activation('conv3_3_relu_4') )

        # 8 
        model.conv4[0].residual[1].register_forward_hook(get_activation('conv4_0_relu_1') )
        model.conv4[0].residual[4].register_forward_hook(get_activation('conv4_0_relu_4') )

        # 9
        model.conv4[1].residual[1].register_forward_hook(get_activation('conv4_1_relu_1') )
        model.conv4[1].residual[4].register_forward_hook(get_activation('conv4_1_relu_4') )

        # 10
        model.conv4[2].residual[1].register_forward_hook(get_activation('conv4_2_relu_1') )
        model.conv4[2].residual[4].register_forward_hook(get_activation('conv4_2_relu_4') )

        # 11
        model.conv4[3].residual[1].register_forward_hook(get_activation('conv4_3_relu_1') )
        model.conv4[3].residual[4].register_forward_hook(get_activation('conv4_3_relu_4') )

        # 12
        model.relu.register_forward_hook(get_activation('relu'))

    elif args.model == 'vgg16':
        model_features = model.features
        def get_layer_feature_maps(X, layers):
            X_l = []
            for i in range(len(model_features)):
                X = model_features[i](X)
                if i in layers:
                    Xc = torch.Tensor(X.cpu())
                    X_l.append(Xc)
            return X_l

    else:
        NotImplementedError("Feature extractor for {args.model} is not implemented!")

    return get_layer_feature_maps, activation

