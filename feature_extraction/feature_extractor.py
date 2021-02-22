from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers
from candidate_models.model_commitments import cornet_brain_pool

from torchvision import transforms
import torchvision.models as models
from torch.utils import model_zoo
from torch.utils.data import Dataset
import torch.nn as nn
import torch

from PIL import Image

import cornet
import h5py
import numpy as np
import h5py
import os

class FeatureExtractor():
    
    def __init__(self):
        
        self.pytorch_models = self.define_pytorch_models()
        self.base_models = base_model_pool
        self.cornet_models = cornet_brain_pool
    
    def get_image_list(self, images_path):
        image_list = []
        for root,dirs,filelist in os.walk(images_path):
            for f in filelist:
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
                    image_list.append(os.path.join(root, f))
        return image_list

    def extract_features(self, images_path, cnn_model, save_dir, use_pytorch=False):
        """ given an images path and CNN model name, extract activations from model and save in save_dir
            if use_brain_score true, all activations extracted and docs for layers are found 
                here:https://github.com/brain-score/candidate_models/blob/master/candidate_models/model_commitments/model_layer_def.py
            else if not true, pytorch used
        """
        print('cnn model name', cnn_model)
        if not os.path.isdir(save_dir):
            os.mkdirs(save_dir)
        save_path = os.path.join(save_dir, cnn_model)
        image_list = self.get_image_list(images_path)
        print('Found {} images'.format(len(image_list)))
        
        if use_pytorch:
            if cnn_model in self.pytorch_models.keys():
                print('Found ' + cnn_model + ' in torchvision models!')
                loader = self.get_dataloader(image_list)
                model = self.pytorch_models[cnn_model]
                model.eval()
                activations = {}
                self.register_hooks(model,activations)

                full_activations = self.get_activations(model, activations, loader) 
                self.save_h5(full_activations, save_path)
                print('Saved ' + cnn_model + 'activations to ' + save_path + '.h5')
        
        else:
            if cnn_model in base_model_pool:
                print('Found ' + cnn_model +  ' in brain_score base_model_pool!')
                model = base_model_pool[cnn_model]  

                # get layers
                layers = model_layers[cnn_model]

                # save activations in dictionary format
                act_dict = {}
                for layer in layers:
                    activations = model(stimuli=image_list, layers=[layer])  # (3)
                    act_dict[layer] = activations

                # write h5 file
                self.save_h5(act_dict,save_path)

                print('Saved ' + cnn_model + 'activations to ' + save_path  + '.h5')


            elif cnn_model in cornet_brain_pool:
                print('Found ' + cnn_model + ' in brain_score cornet_model_pool!')
                loader = self.get_dataloader(image_list)
                one_letter_code = cnn_model.split('-')[1]
                model = cornet.get_model(one_letter_code,pretrained=True,map_location=torch.device('cpu')).module
                model.eval()
                activations = {}
                self.register_hooks_COR(model,activations,one_letter_code == "RT" or one_letter_code == "R")

                full_activations = self.get_activations(model, activations, loader)
                self.save_h5(full_activations, save_path)
                print('Saved ' + cnn_model + 'activations to ' + save_path + '.h5')

            elif cnn_model in self.pytorch_models.keys():
                print('Found ' + cnn_model + ' in torchvision models!')
                loader = self.get_dataloader(image_list)
                model = self.pytorch_models[cnn_model]
                model.eval()
                activations = {}
                self.register_hooks(model,activations)

                full_activations = self.get_activations(model, activations, loader) 
                self.save_h5(full_activations, save_path)
                print('Saved ' + cnn_model + 'activations to ' + save_path + '.h5')

            else:
                raise ValueError("Model not supported. Call show_supported_models() to see what models are currently available.") 

    def get_dataloader(self, image_list):
        dataset = ImageData(image_list)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64)
        return loader

    def get_activations(self, model, activations, loader):
        full_activations = {}
           
        with torch.no_grad():
            for img in loader:
                model(img)
                for key in activations:
                    if key in full_activations:
                        full_activations[key] = np.concatenate((full_activations[key],activations[key]),axis=0)
                    else:
                        full_activations[key] = activations[key]
        return full_activations
    
    def show_supported_models(self, ):
        print("Brain-Score\n")
        print('Base model names:', base_model_pool.keys())
        print('\n')
        print('CORnets:', cornet_brain_pool.keys())
        print("Pytorch\n")
        print(self.pytorch_models.keys())

    def register_hooks(self, model,activation_dict,name=""):
        '''
        GENERALIZED PYTORCH HOOK ADDER
        Parses through _module attributes
        only add convolutional layers now
        changed naming scheme to separate out variable names with '-'
        '''
        for entity in vars(model)["_modules"]:
            if len(vars(vars(model)["_modules"][entity])["_modules"]) == 0 and isinstance(vars(model)["_modules"][entity],nn.Conv2d):
                if len(name) > 0:
                    vars(model)["_modules"][entity].register_forward_hook(self.get_activation(activation_dict,name + "-" + entity))
                else:
                    vars(model)["_modules"][entity].register_forward_hook(self.get_activation(activation_dict,entity))
            else:
                if len(name) > 0:
                    self.register_hooks(vars(model)["_modules"][entity],activation_dict,name + "-" + entity)
                else:
                    self.register_hooks(vars(model)["_modules"][entity],activation_dict,entity)

    def save_h5(self, data_activity,name):
        '''
        data should be in dictionary format
        '''
        with h5py.File(name + ".h5",'w') as f:
            for key in data_activity:
                f.create_dataset(key,data = data_activity[key])

    def get_activation(self, activation_dict,name):
        ''' 
        Helper method to register activity hooks in pytorch model 
            activation_dict: dictionary mapping layer name to activation array
            name: name of the layer
        '''
        def hook(model,input, output):
            activation_dict[name] = output.detach()
        return hook
    
    def register_hooks_COR(self, model,activation_dict,flag = False):
        '''
        takes advantage of the self.output for the 4 classes of layers in CORnet models
        '''
        layers = ['V1','V2','V4','IT']
        if flag:
            model.V4.output.register_forward_hook(self.get_activation(activation_dict,'V4'))
            model.IT.output.register_forward_hook(self.get_activation(activation_dict,'IT'))
        else:
            for idx,layer in enumerate(layers):
                model[idx].output.register_forward_hook(self.get_activation(activation_dict,layer))

    def define_pytorch_models(self):
        pytorch_model_dict = {
            'resnet18': models.resnet18(pretrained=True),
            'alexnet': models.alexnet(pretrained=True),
            'squeezenet' : models.squeezenet1_0(pretrained=True),
            'vgg16' : models.vgg16(pretrained=True), 
            'densenet' : models.densenet161(pretrained=True),
            'inception' : models.inception_v3(pretrained=True), 
            'googlenet' : models.googlenet(pretrained=True),
            'shufflenet' : models.shufflenet_v2_x1_0(pretrained=True),
            'mobilenet' : models.mobilenet_v2(pretrained=True), 
            'resnext50' : models.resnext50_32x4d(pretrained=True),
            'wide_resnet50_2' : models.wide_resnet50_2(pretrained=True),
            'mnasnet': models.mnasnet1_0(pretrained=True)
        }
        return pytorch_model_dict
    
    def view_saved_features(self, feature_path):
        with h5py.File(feature_path,'r') as f:
            print(f.keys())
            print(f[f.keys()[0]].shape)

        
class ImageData(Dataset):
    def __init__(self, filelist):
        self.data = []
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for file in filelist:            
            im = Image.open(file).convert('RGB')
            self.data.append(self.preprocess(im))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
