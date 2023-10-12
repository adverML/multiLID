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
                    X_l.append(Xc.cuda())
            return X_l

    else:
        NotImplementedError("Feature extractor for {args.model} is not implemented!")

    return get_layer_feature_maps, activation


def get_whitebox_features_old(args, logger, model):

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    layer_nr = int(args.nr)
    logger.log("INFO: layer_nr " + str(layer_nr) ) 

    if args.net in ['cif10vgg', 'cif10vggnew' , 'cif100vgg']:
        # indices of activation layers
        # act_layers= [2,5,9,12,16,19,22,26,29,32,36,39,42]
        
        #get a list of all feature maps of all layers
        model_features = model.features
        def get_layer_feature_maps(X, layers):
            X_l = []
            for i in range(len(model_features)):
                X = model_features[i](X)
                if i in layers:
                    Xc = torch.Tensor(X.cpu())
                    X_l.append(Xc.cuda())
            return X_l

        # default layer
        layers = [9, 16, 22, 29, 36, 42] # fourier_act_layers
        
        if args.net == 'cif100vgg' and (args.attack == 'cw' or args.attack == 'df'):
            layers = [42]
            
        if args.detector in ['LID', 'LIDNOISE', 'multiLID', 'FFTmultiLIDMFS', 'FFTmultiLIDPFS', 'Mahalanobis']:
            layers = [2,5,9,12,16,19,22,26,29,32,36,39,42]

        if layer_nr == 0:
            layers = [2]
        elif layer_nr == 1:
            layers = [5]
        elif layer_nr == 2:
            layers = [9]
        elif layer_nr == 3:
            layers = [12]
        elif layer_nr == 4:
            layers = [16]
        elif layer_nr == 5:
            layers = [19]
        elif layer_nr == 6:
            layers = [22]
        elif layer_nr == 7:
            layers = [26]
        elif layer_nr == 8:
            layers = [29]
        elif layer_nr == 9:
            layers = [32]
        elif layer_nr == 10:
            layers = [36]
        elif layer_nr == 11:
            layers = [39]
        elif layer_nr == 12:
            layers = [42]
        else:
            logger.log( "INFO: layer nr > 12" + ", args.nr " + str(args.nr) )
            assert True

    elif args.net == 'cif10rn34sota':
        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list

        if not args.nr == -1:
            model.layer1[0].conv2.register_forward_hook( get_activation('1_conv2_0') )
            model.layer1[1].conv2.register_forward_hook( get_activation('1_conv2_1') )
            model.layer1[2].conv2.register_forward_hook( get_activation('1_conv2_2') )

            model.layer2[0].conv2.register_forward_hook( get_activation('2_conv2_0') )
            model.layer2[1].conv2.register_forward_hook( get_activation('2_conv2_1') )
            model.layer2[2].conv2.register_forward_hook( get_activation('2_conv2_2') )
            model.layer2[3].conv2.register_forward_hook( get_activation('2_conv2_3') )

            model.layer3[0].conv2.register_forward_hook( get_activation('3_conv2_0') )
            model.layer3[1].conv2.register_forward_hook( get_activation('3_conv2_1') )
            model.layer3[2].conv2.register_forward_hook( get_activation('3_conv2_2') )
            model.layer3[3].conv2.register_forward_hook( get_activation('3_conv2_3') )
            model.layer3[4].conv2.register_forward_hook( get_activation('3_conv2_4') )

            model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
            model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
            model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') ) 

        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):
                
                if not args.attack == 'fgsm':
                    # last block
                    model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
                    model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
                    model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') ) 
                    layers = [
                        '4_conv2_0', '4_conv2_1', '4_conv2_2'
                    ]
                else:
                    model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
                    model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
                    layers = [
                        '4_conv2_0', '4_conv2_1'
                    ]
            else:
                model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
                model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
                model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') ) 
                layers = [
                    '4_conv2_2'
                    # 'conv5_0_relu', 'conv5_1_relu'
                ]

        if layer_nr == 0:
            layers = ['1_conv2_0']
        elif layer_nr == 1:
            layers = ['1_conv2_1']
        elif layer_nr == 2:
            layers = ['1_conv2_2']
        elif layer_nr == 3:
            layers = ['2_conv2_0']
        elif layer_nr == 4:
            layers = ['2_conv2_1']
        elif layer_nr == 5:
            layers = ['2_conv2_2']
        elif layer_nr == 6:
            layers = ['2_conv2_3']
        elif layer_nr == 7:
            layers = ['3_conv2_0']
        elif layer_nr == 8:
            layers = ['3_conv2_1']
        elif layer_nr == 9:
            layers = ['3_conv2_2']
        elif layer_nr == 10:
            layers = ['3_conv2_3']
        elif layer_nr == 11:
            layers = ['3_conv2_4']
        elif layer_nr == 12:
            layers = ['4_conv2_0']
        elif layer_nr == 13:
            layers = ['4_conv2_1']
        elif layer_nr == 14:
            layers = ['4_conv2_2']
        else:
            logger.log( "INFO: layer nr > 14" + ", args.nr " + str(args.nr) )
            assert True


    elif args.net in ['cif10rn34', 'cif100rn34']:

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list

        #layer_name = layer_name_cif10

        if not args.nr == -1:

            model.conv2_x[0].residual_function[2].register_forward_hook( get_activation('conv2_0_relu') )
            model.conv2_x[1].residual_function[2].register_forward_hook( get_activation('conv2_1_relu') )
            model.conv2_x[2].residual_function[2].register_forward_hook( get_activation('conv2_2_relu') )

            model.conv3_x[0].residual_function[2].register_forward_hook( get_activation('conv3_0_relu') )
            model.conv3_x[1].residual_function[2].register_forward_hook( get_activation('conv3_1_relu') )
            model.conv3_x[2].residual_function[2].register_forward_hook( get_activation('conv3_2_relu') )
            model.conv3_x[3].residual_function[2].register_forward_hook( get_activation('conv3_3_relu') )

            model.conv4_x[0].residual_function[2].register_forward_hook( get_activation('conv4_0_relu') )
            model.conv4_x[1].residual_function[2].register_forward_hook( get_activation('conv4_1_relu') )
            model.conv4_x[2].residual_function[2].register_forward_hook( get_activation('conv4_2_relu') )
            model.conv4_x[3].residual_function[2].register_forward_hook( get_activation('conv4_3_relu') )
            model.conv4_x[4].residual_function[2].register_forward_hook( get_activation('conv4_4_relu') )

            model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
            model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
            model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )      

        else:
            if not ( args.attack  in ['df', 'cw', 'fab','fab-t'] ):
                
                if not args.attack == 'fgsm':
                    # last block
                    model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
                    model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
                    model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )    
                    layers = [
                        'conv5_0_relu', 'conv5_1_relu', 'conv5_2_relu'
                    ]
                else:
                    model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
                    model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
                    # model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )    
                    layers = [
                        'conv5_0_relu', 'conv5_1_relu'
                    ]
            else:
                model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
                model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
                model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )   
                layers = [
                    'conv5_2_relu'
                    # 'conv5_0_relu', 'conv5_1_relu'
                ]

        if layer_nr == 0:
            layers = ['conv2_0_relu']
        elif layer_nr == 1:
            layers = ['conv2_1_relu']
        elif layer_nr == 2:
            layers = ['conv2_2_relu']
        elif layer_nr == 3:
            layers = ['conv3_0_relu']
        elif layer_nr == 4:
            layers = ['conv3_1_relu']
        elif layer_nr == 5:
            layers = ['conv3_2_relu']
        elif layer_nr == 6:
            layers = ['conv3_3_relu']
        elif layer_nr == 7:
            layers = ['conv4_0_relu']
        elif layer_nr == 8:
            layers = ['conv4_1_relu']
        elif layer_nr == 9:
            layers = ['conv4_2_relu']
        elif layer_nr == 10:
            layers = ['conv4_3_relu']
        elif layer_nr == 11:
            layers = ['conv4_4_relu']
        elif layer_nr == 12:
            layers = ['conv5_0_relu']
        elif layer_nr == 13:
            layers = ['conv5_1_relu']
        elif layer_nr == 14:
            layers = ['conv5_2_relu']
        else:
            logger.log( "INFO: layer nr > 14" + ", args.nr " + str(args.nr) )
            assert True


    elif (args.net in [ 'mnist', 'cif10', 'cif100', 'celebaHQ32', 'imagenet32' 
       , 'celebaHQ64', 'celebaHQ128', 'celebaHQ256'
       , 'imagenet64', 'imagenet128']):

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list

        if not args.nr == -1 or args.detector in ['LIDLessFeatures', 'multiLIDLessFeatures']:
            model.init_conv.register_forward_hook( get_activation('init_conv') )
            model.conv2.register_forward_hook( get_activation('seq_conv2') )
            model.conv3.register_forward_hook( get_activation('seq_conv3') )
            model.conv4.register_forward_hook( get_activation('seq_conv4') )
            model.relu.register_forward_hook(get_activation('relu'))
        
        if not args.nr == -1 or args.detector in ['LID', 'LIDNOISE', 'multiLID', 'FFTmultiLIDMFS', 'FFTmultiLIDPFS', 'Mahalanobis']:
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
            
            
        else:

            if not (args.attack == 'df' or  args.attack == 'cw'):
                
                if args.detector == 'LayerPFS':
                    # 7
                    model.conv3[3].residual[1].register_forward_hook(get_activation('conv3_3_relu_1'))
                    model.conv3[3].residual[4].register_forward_hook(get_activation('conv3_3_relu_4'))
                    
                    # 12
                    model.relu.register_forward_hook(get_activation('relu'))
                    
                    layers = [
                        'conv3_3_relu_1', 'conv3_3_relu_4', 'relu', 
                    ]
                else:
                    # 3
                    model.conv2[3].residual[1].register_forward_hook( get_activation('conv2_3_relu_1') )
                    model.conv2[3].residual[4].register_forward_hook( get_activation('conv2_3_relu_4') )
                    # 4
                    model.conv3[0].residual[1].register_forward_hook( get_activation('conv3_0_relu_1') )
                    model.conv3[0].residual[4].register_forward_hook( get_activation('conv3_0_relu_4') )
                    
                    layers = [
                        'conv2_3_relu_1', 'conv2_3_relu_4', 'conv3_0_relu_1', 'conv3_0_relu_4'
                    ]
                    

            else: # df cw
                if args.detector == 'LayerPFS':                   
                    # 12
                    model.relu.register_forward_hook(get_activation('relu'))
                    
                    layers = [
                        'relu', 
                    ]   
                else:
                
                    model.conv4[3].residual[1].register_forward_hook(get_activation('conv4_3_relu_1'))
                    model.conv4[3].residual[4].register_forward_hook(get_activation('conv4_3_relu_4'))
                    model.relu.register_forward_hook(get_activation('relu'))

                    if args.net in ['celebaHQ32', 'celebaHQ64', 'celebaHQ128', 'celebaHQ256']:
                        layers = [
                            'relu'
                        ]
                    else: 
                        layers = [

                            'relu'
                        ]

        if args.detector in ['LIDLessFeatures', 'multiLIDLessFeatures']:
            layers = [
                'init_conv',
                'seq_conv2',
                'seq_conv3',
                'seq_conv4',

            ]
        

        if args.detector in ['LID', 'LIDNOISE', 'multiLID', 'FFTmultiLIDMFS', 'FFTmultiLIDPFS']:
            
            layers = [
                'conv2_0_relu_4',  'conv2_1_relu_4',  'conv2_2_relu_4', 'conv2_3_relu_4', 'conv3_0_relu_4', 'conv3_1_relu_4', 
                'conv3_2_relu_4', 'conv3_3_relu_4', 'conv4_0_relu_4', 'conv4_1_relu_4','conv4_2_relu_4', 'conv4_3_relu_4', 'relu' 
            ]
            
        if args.detector in [ 'Mahalanobis' ]:
            layers = [
                'conv2_0_relu_4',  'conv2_1_relu_4',  'conv2_2_relu_4', 'conv2_3_relu_4', 'conv3_0_relu_4', 'conv3_1_relu_4', 
                'conv3_2_relu_4', 'conv3_3_relu_4', 'conv4_0_relu_4', 'conv4_1_relu_4','conv4_2_relu_4', 'conv4_3_relu_4', 'relu' 
            ]

        if layer_nr == 0:
            layers = ['conv2_0_relu_4']
        elif layer_nr == 1:
            layers = ['conv2_1_relu_4']
        elif layer_nr == 2:
            layers = ['conv2_2_relu_4']
        elif layer_nr == 3:
            layers = ['conv2_3_relu_4']
        elif layer_nr == 4:
            layers = ['conv3_0_relu_4']
        elif layer_nr == 5:
            layers = ['conv3_1_relu_4']
        elif layer_nr == 6:
            layers = ['conv3_2_relu_4']
        elif layer_nr == 7:
            layers = ['conv3_3_relu_4']
        elif layer_nr == 8:
            layers = ['conv4_0_relu_4']
        elif layer_nr == 9:
            layers = ['conv4_1_relu_4']
        elif layer_nr == 10:
            layers = ['conv4_2_relu_4']
        elif layer_nr == 11:
            layers = ['conv4_3_relu_4']
        elif layer_nr == 12:
            layers = ['relu']
        else:
            logger.log( "INFO: layer nr > 12" + ", args.nr " + str(args.nr) )
            assert True


    elif args.net in ['imagenet', 'imagenet_hierarchy']:

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list

        if not args.nr == -1 or args.detector in ['LID', 'LIDNOISE', 'multiLID',  'FFTmultiLIDMFS', 'FFTmultiLIDPFS', 'Mahalanobis']:
            model.relu.register_forward_hook( get_activation('0_relu') )

            model.layer1[0].relu.register_forward_hook( get_activation('layer_1_0_relu') )
            model.layer1[1].relu.register_forward_hook( get_activation('layer_1_1_relu') )
            model.layer1[2].relu.register_forward_hook( get_activation('layer_1_2_relu') )

            model.layer2[0].relu.register_forward_hook( get_activation('layer_2_0_relu') )
            model.layer2[1].relu.register_forward_hook( get_activation('layer_2_1_relu') )
            model.layer2[2].relu.register_forward_hook( get_activation('layer_2_2_relu') )
            model.layer2[3].relu.register_forward_hook( get_activation('layer_2_3_relu') )

            model.layer3[0].relu.register_forward_hook( get_activation('layer_3_0_relu') )
            model.layer3[1].relu.register_forward_hook( get_activation('layer_3_1_relu') )
            model.layer3[2].relu.register_forward_hook( get_activation('layer_3_2_relu') )
            model.layer3[3].relu.register_forward_hook( get_activation('layer_3_3_relu') )
            model.layer3[4].relu.register_forward_hook( get_activation('layer_3_4_relu') )
            model.layer3[5].relu.register_forward_hook( get_activation('layer_3_5_relu') )

            model.layer4[0].relu.register_forward_hook( get_activation('layer_4_0_relu') )
            model.layer4[1].relu.register_forward_hook( get_activation('layer_4_1_relu') )
            model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )

        else:
            model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )

            if not (args.attack == 'df' or  args.attack == 'cw'):

                model.layer3[0].relu.register_forward_hook( get_activation('layer_3_0_relu') )
                model.layer3[1].relu.register_forward_hook( get_activation('layer_3_1_relu') )
                
                model.layer3[4].relu.register_forward_hook( get_activation('layer_3_4_relu') )
                
                layers = [
                    'layer_3_0_relu', 'layer_3_1_relu', 'layer_3_4_relu'
                ]
            else:
                layers = [

                    'layer_4_2_relu'
                ]


        if args.detector in ['LID']:
            layers = [
                '0_relu', 'layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu', 'layer_2_1_relu', 'layer_2_2_relu',  'layer_2_3_relu', 
                'layer_3_0_relu', 'layer_3_1_relu',  'layer_3_2_relu',  'layer_3_3_relu', 'layer_3_4_relu',  'layer_3_5_relu',  'layer_4_0_relu',  'layer_4_1_relu',   'layer_4_2_relu'
            ]
        
        if args.detector in ['LIDNOISE', 'multiLID', 'FFTmultiLIDMFS', 'FFTmultiLIDPFS']:
            layers = [
             '0_relu', 'layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu', 'layer_2_1_relu', 'layer_2_2_relu',  'layer_2_3_relu', 
                'layer_3_0_relu', 'layer_3_1_relu',  'layer_3_2_relu',  'layer_3_3_relu', 'layer_3_4_relu',  'layer_3_5_relu',  'layer_4_0_relu',  'layer_4_1_relu',   'layer_4_2_relu'
            ]
        
        if args.detector in ['Mahalanobis']:
            layers = [
                'layer_3_0_relu', 'layer_3_1_relu',  'layer_3_2_relu', 'layer_3_3_relu', 'layer_3_4_relu',  'layer_3_5_relu',  'layer_4_0_relu',  'layer_4_1_relu',   'layer_4_2_relu'
            ]
        
        
        if layer_nr == 0:
            layers = ['0_relu']
        elif layer_nr == 1:
            layers = ['layer_1_0_relu']
        elif layer_nr == 2:
            layers = ['layer_1_1_relu']
        elif layer_nr == 3:
            layers = ['layer_1_2_relu']
        elif layer_nr == 4:
            layers = ['layer_2_0_relu']
        elif layer_nr == 5:
            layers = ['layer_2_1_relu']
        elif layer_nr == 6:
            layers = ['layer_2_2_relu']
        elif layer_nr == 7:
            layers = ['layer_2_3_relu']
        elif layer_nr == 8:
            layers = ['layer_3_0_relu']
        elif layer_nr == 9:
            layers = ['layer_3_1_relu']
        elif layer_nr == 10:
            layers = ['layer_3_2_relu']
        elif layer_nr == 11:
            layers = ['layer_3_3_relu']
        elif layer_nr == 12:
            layers = ['layer_3_4_relu']   
        elif layer_nr == 13:
            layers = ['layer_3_5_relu']
        elif layer_nr == 14:
            layers = ['layer_4_0_relu']
        elif layer_nr == 15:
            layers = ['layer_4_1_relu']
        elif layer_nr == 16:
            layers = ['layer_4_2_relu']
        else:
            logger.log( "INFO: layer nr > 16" + ", args.nr " + str(args.nr) )
            assert True


    elif args.net == 'restricted_imagenet':

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list
        
        if not args.nr == -1:
            model.relu.register_forward_hook( get_activation('0_relu') )

            model.layer1[0].relu.register_forward_hook( get_activation('layer_1_0_relu') )
            model.layer1[1].relu.register_forward_hook( get_activation('layer_1_1_relu') )
            model.layer1[2].relu.register_forward_hook( get_activation('layer_1_2_relu') )

            model.layer2[0].relu.register_forward_hook( get_activation('layer_2_0_relu') )
            model.layer2[1].relu.register_forward_hook( get_activation('layer_2_1_relu') )
            model.layer2[2].relu.register_forward_hook( get_activation('layer_2_2_relu') )
            model.layer2[3].relu.register_forward_hook( get_activation('layer_2_3_relu') )

            model.layer3[0].relu.register_forward_hook( get_activation('layer_3_0_relu') )
            model.layer3[1].relu.register_forward_hook( get_activation('layer_3_1_relu') )
            model.layer3[2].relu.register_forward_hook( get_activation('layer_3_2_relu') )
            model.layer3[3].relu.register_forward_hook( get_activation('layer_3_3_relu') )
            model.layer3[4].relu.register_forward_hook( get_activation('layer_3_4_relu') )
            model.layer3[5].relu.register_forward_hook( get_activation('layer_3_5_relu') )

            model.layer4[0].relu.register_forward_hook( get_activation('layer_4_0_relu') )
            model.layer4[1].relu.register_forward_hook( get_activation('layer_4_1_relu') )
            model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )

        else:
            model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )

            if not (args.attack == 'df' or  args.attack == 'cw'):
                
                layers = [
                    'layer_4_2_relu'
                ]
            else:
                layers = [
                    'layer_4_2_relu'
                ]

        if layer_nr == 0:
            layers = ['0_relu']
        elif layer_nr == 1:
            layers = ['layer_1_0_relu']
        elif layer_nr == 2:
            layers = ['layer_1_1_relu']
        elif layer_nr == 3:
            layers = ['layer_1_2_relu']
        elif layer_nr == 4:
            layers = ['layer_2_0_relu']
        elif layer_nr == 5:
            layers = ['layer_2_1_relu']
        elif layer_nr == 6:
            layers = ['layer_2_2_relu']
        elif layer_nr == 7:
            layers = ['layer_2_3_relu']
        elif layer_nr == 8:
            layers = ['layer_3_0_relu']
        elif layer_nr == 9:
            layers = ['layer_3_1_relu']
        elif layer_nr == 10:
            layers = ['layer_3_2_relu']
        elif layer_nr == 11:
            layers = ['layer_3_3_relu']
        elif layer_nr == 12:
            layers = ['layer_3_4_relu']
        elif layer_nr == 13:
            layers = ['layer_3_5_relu']
        elif layer_nr == 14:
            layers = ['layer_4_0_relu']
        elif layer_nr == 15:
            layers = ['layer_4_1_relu']
        elif layer_nr == 16:
            layers = ['layer_4_2_relu']
        else:
            logger.log( "INFO: layer nr > 16" + ", args.nr " + str(args.nr) )
            assert True


    elif args.net == 'cif10_rb':

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list

        if not args.nr == -1:
            # 0
            model.layer[0].block[0].relu_0.register_forward_hook( get_activation('layer_0_0_relu_0') )
            model.layer[0].block[0].relu_1.register_forward_hook( get_activation('layer_0_0_relu_1') )
            # 1
            model.layer[0].block[1].relu_0.register_forward_hook( get_activation('layer_0_1_relu_0') )
            model.layer[0].block[1].relu_1.register_forward_hook( get_activation('layer_0_1_relu_1') )
            # 2
            model.layer[0].block[2].relu_0.register_forward_hook( get_activation('layer_0_2_relu_0') )
            model.layer[0].block[2].relu_1.register_forward_hook( get_activation('layer_0_2_relu_1') )
            # 3
            model.layer[0].block[3].relu_0.register_forward_hook( get_activation('layer_0_3_relu_0') )
            model.layer[0].block[3].relu_1.register_forward_hook( get_activation('layer_0_3_relu_1') )
            # 4
            model.layer[1].block[0].relu_0.register_forward_hook( get_activation('layer_1_0_relu_0') )
            model.layer[1].block[0].relu_1.register_forward_hook( get_activation('layer_1_0_relu_1') )
            # 5
            model.layer[1].block[1].relu_0.register_forward_hook( get_activation('layer_1_1_relu_0') )
            model.layer[1].block[1].relu_1.register_forward_hook( get_activation('layer_1_1_relu_1') )
            # 6
            model.layer[1].block[2].relu_0.register_forward_hook( get_activation('layer_1_2_relu_0') )
            model.layer[1].block[2].relu_1.register_forward_hook( get_activation('layer_1_2_relu_1') )
            # 7
            model.layer[1].block[3].relu_0.register_forward_hook( get_activation('layer_1_3_relu_0') )
            model.layer[1].block[3].relu_1.register_forward_hook( get_activation('layer_1_3_relu_1') )
            # 8
            model.layer[2].block[0].relu_0.register_forward_hook( get_activation('layer_2_0_relu_0') )
            model.layer[2].block[0].relu_1.register_forward_hook( get_activation('layer_2_0_relu_1') )
            # 9
            model.layer[2].block[1].relu_0.register_forward_hook( get_activation('layer_2_1_relu_0') )
            model.layer[2].block[1].relu_1.register_forward_hook( get_activation('layer_2_1_relu_1') )
            # 10
            model.layer[2].block[2].relu_0.register_forward_hook( get_activation('layer_2_2_relu_0') )
            model.layer[2].block[2].relu_1.register_forward_hook( get_activation('layer_2_2_relu_1') )
            # 11
            model.layer[2].block[3].relu_0.register_forward_hook( get_activation('layer_2_3_relu_0') )
            model.layer[2].block[3].relu_1.register_forward_hook( get_activation('layer_2_3_relu_1') )
            # 12
            model.relu.register_forward_hook( get_activation('relu') )

        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):
                # 5
                model.layer[1].block[1].relu_0.register_forward_hook( get_activation('layer_1_1_relu_0') )
                model.layer[1].block[1].relu_1.register_forward_hook( get_activation('layer_1_1_relu_1') )

                # 7
                model.layer[1].block[3].relu_0.register_forward_hook( get_activation('layer_1_3_relu_0') )
                model.layer[1].block[3].relu_1.register_forward_hook( get_activation('layer_1_3_relu_1') )
                layers = [
                    'layer_1_1_relu_0', 'layer_1_1_relu_1', 'layer_1_3_relu_0', 'layer_1_3_relu_1'
                ]
            else:
                # 12
                model.relu.register_forward_hook( get_activation('relu') )
                layers = [  'relu'  ]
        
        if layer_nr == 0:
            layers = ['layer_0_0_relu_0', 'layer_0_0_relu_1']
        elif layer_nr == 1:
            layers = ['layer_0_1_relu_0', 'layer_0_1_relu_1']
        elif layer_nr == 2:
            layers = ['layer_0_2_relu_0', 'layer_0_2_relu_1']
        elif layer_nr == 3:
            layers = ['layer_0_3_relu_0', 'layer_0_3_relu_1']
        elif layer_nr == 4:
            layers = ['layer_1_0_relu_0', 'layer_1_0_relu_1']
        elif layer_nr == 5:
            layers = ['layer_1_1_relu_0', 'layer_1_1_relu_1']
        elif layer_nr == 6:
            layers = ['layer_1_2_relu_0', 'layer_1_2_relu_1']
        elif layer_nr == 7:
            layers = ['layer_1_3_relu_0', 'layer_1_3_relu_0']
        elif layer_nr == 8:
            layers = ['layer_2_0_relu_0', 'layer_2_0_relu_1']
        elif layer_nr == 9:
            layers = ['layer_2_1_relu_0', 'layer_2_1_relu_1']
        elif layer_nr == 10:
            layers = ['layer_2_2_relu_0', 'layer_2_2_relu_1']
        elif layer_nr == 11:
            layers = ['layer_2_3_relu_0', 'layer_2_3_relu_1']
        elif layer_nr == 12:
            layers = ['relu']
        else:
            logger.log( "INFO: layer nr > 12" + ", args.nr " + str(args.nr) )
            assert True

    logger.log('INFO: ' + str(layers))

    return get_layer_feature_maps, layers, model, activation
