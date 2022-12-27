# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_custom import resnet50_baseline
# from resnet_custom import resnet50_baseline

class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_tasks: number of tasks
    """
    def __init__(self, L, D, dropout=False, n_tasks=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_tasks
        return A, x


class TOAD(nn.Module):
    def __init__(self, task_classes=[3,], dropout=False):
        super().__init__()
        size = [1024, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.LeakyReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L=size[1], D=size[1], dropout=False, n_tasks=len(task_classes))
        fc.append(attention_net)

        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.ModuleList()
        for n_class in task_classes:
            self.classifier.append(nn.Linear(size[1], n_class))

    def forward(self, h, attention_only=False):
        # in
        ## h: N, 1024
        A, h = self.attention_net(h)
        # out
        ## A: N, n_tasks
        ## h: N, 512
        A = torch.transpose(A, 1, 0)
        ## A: n_tasks, N
        if attention_only:
            return A
        # apply attention
        A = F.softmax(A, dim=1)
        # in
        ## A: n_tasks, N
        ## h: N, 512
        M = torch.mm(A, h)
        ## M: n_tasks, 512
        output_list = []
        for i in range(len(self.classifier)):
            output_list.append(self.classifier[i](M[i].unsqueeze(0)))
        return output_list


class ResTOAD(nn.Module):
    def __init__(self, task_classes=3, dropout=False, requires_grad=True, loss_fn=None, input_channel=3):
        """
        task_classes: 这里的task可以是分类,也可以是回归,如是回归,则取值1. 
            例如第一个任务是3分类,第二个任务是回归,则task_classed=[3,1],当然,相应的损失函数也要主要根据任务而变化.
        loss_fn: tuple，损失函数。有多少个任务就多少个损失函数。
        """
        super().__init__()
        
        self.task_classes = task_classes
        self.loss_fn = loss_fn
        
        self.resnet_baseline = resnet50_baseline(pretrained=True,requires_grad=requires_grad)
        # 将平均池化改为最大池化
        self.resnet_baseline.avgpool = nn.AdaptiveMaxPool2d(output_size=1)
        
        if input_channel != 3:
            assert requires_grad==True, 'One should not change the first convolution layer when requires_grad is False.'
            self.resnet_baseline.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        size = [1024, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.LeakyReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if isinstance(task_classes,list):
            n_tasks=len(task_classes)
        elif isinstance(task_classes,int):
            n_tasks=1
        attention_net = Attn_Net_Gated(L=size[1], D=size[1], dropout=False, n_tasks=n_tasks)
        fc.append(attention_net)

        self.attention_net = nn.Sequential(*fc)
        if type(task_classes)==list and len(task_classes)>1:
            self.classifier = nn.ModuleList()
            for n_class in task_classes:
                self.classifier.append(nn.Linear(size[1], n_class))
        elif type(task_classes)==list and len(task_classes)==1:
            self.classifier = nn.Linear(size[1], task_classes[0])
        elif type(task_classes)==int:
            self.classifier = nn.Linear(in_features=size[1], out_features=task_classes, bias=True)

    def forward(self, images, attention_only=False):
        """
        return list of classification logit
        """
        # images.shape: (N, 3, 256, 256)
        # out: (N, 1024)
        h = self.resnet_baseline(images)
        # in
        ## h: N, 1024
        A, h = self.attention_net(h)
        # out
        ## A: N, n_tasks
        ## h: N, 512
        A = torch.transpose(A, 1, 0)
        ## A: n_tasks, N
        if attention_only:
            return A
        # apply attention
        A = F.softmax(A, dim=1)
        # in
        ## A: n_tasks, N
        ## h: N, 512
        M = torch.mm(A, h)# MIL的核心就是通过这个矩阵乘法实现N个实例的特征向量压缩到一个特征向量.
        ## M: n_tasks, 512
        if isinstance(self.classifier, torch.nn.modules.container.ModuleList):
            output_list = []
            for i in range(len(self.classifier)):
                output_list.append(self.classifier[i](M[i].unsqueeze(0)))
            return output_list
        else:
            output = self.classifier(M[0].unsqueeze(0))
            return output


# +
class Attn_Net(nn.Module):
    """
    Attention Network
    """
    def __init__(self, resnet_baseline, attn_net_gated):
        super(Attn_Net, self).__init__()
        self.resnet_baseline = resnet_baseline
        self.attn_net_gated = attn_net_gated

    def forward(self, images, attention_only=False):
        """
        return list of classification logit
        """
        # images.shape: (N, 3, 256, 256)
        # out: (N, 1024)
        h = self.resnet_baseline(images)
        # in
        ## h: N, 1024
        A, h = self.attn_net_gated(h)
        # out
        ## A: N, n_tasks
        ## h: N, 512
        A = torch.transpose(A, 1, 0)
        ## A: n_tasks, N
        if attention_only:
            return A
        # apply attention
        A = F.softmax(A, dim=1)
        # in
        ## A: n_tasks, N
        ## h: N, 512
        M = torch.mm(A, h)# MIL的核心就是通过这个矩阵乘法实现N个实例的特征向量压缩到一个特征向量.
        ## M: n_tasks, 512
        return M
    

class MultimodalResTOAD(nn.Module):
    """
    多(3)模态的ResTOAD。输入有3个分支（SWAN,T1,T2），每个分支的MRI影像切面数量不同，在最后的全连接层前一层再融合（拼接）。
    """
    def __init__(self, modal_names=3, task_classes=3, loss_fn=None, dropout=False, requires_grad=True):
        """
        loss_fn: 损失函数,若是双任务,则是tuple, tuple内的每个元素都是一种损失函数.
        """
        super().__init__()
        
        self.modal_names = modal_names
        self.task_classes = task_classes
        self.loss_fn = loss_fn
        
        self.resnet_baseline = nn.ModuleDict()
        self.attention_net = nn.ModuleDict()
        for modal_name in self.modal_names:
            self.resnet_baseline[modal_name] = resnet50_baseline(pretrained=True,requires_grad=requires_grad)
        
            size = [1024, 512, 256]
            fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
            if dropout:
                fc.append(nn.Dropout(0.25))
            fc.extend([nn.Linear(size[1], size[1]), nn.LeakyReLU()])
            if dropout:
                fc.append(nn.Dropout(0.25))
            if isinstance(task_classes,list):
                n_tasks=len(task_classes)
            elif isinstance(task_classes,int):
                n_tasks=1
            attn_net_gated = Attn_Net_Gated(L=size[1], D=size[1], dropout=False, n_tasks=n_tasks)
            fc.append(attn_net_gated)
            self.attention_net[modal_name] = nn.Sequential(*fc)
            
#             self.attention_net[modal_name] = Attn_Net(resnet_baseline, attn_net_gated)
            
        if type(task_classes)==list and len(task_classes)>1:
            self.classifier = nn.ModuleList()
            for n_class in task_classes:
                self.classifier.append(nn.Linear(size[1]*len(modal_names), n_class))
        elif type(task_classes)==list and len(task_classes)==1:
            self.classifier = nn.Linear(size[1]*len(modal_names), task_classes[0])
            #self.classifier = nn.Linear(size[1], task_classes[0])
        elif type(task_classes)==int:
            self.classifier = nn.Linear(size[1]*len(modal_names), task_classes)
            #self.classifier = nn.Linear(size[1], task_classes)

    def singlemodal_forward(self, modal_name, images, attention_only=False):
        """
        return list of classification logit
        """
        # images.shape: (N, 3, 256, 256)
        # out: (N, 1024)
        h = self.resnet_baseline[modal_name](images)
        # in
        ## h: N, 1024
        A, h = self.attention_net[modal_name](h)
        # out
        ## A: N, n_tasks
        ## h: N, 512
        A = torch.transpose(A, 1, 0)
        ## A: n_tasks, N
        if attention_only:
            return A
        # apply attention
        A = F.softmax(A, dim=1)
        # in
        ## A: n_tasks, N
        ## h: N, 512
        M = torch.mm(A, h)# MIL的核心就是通过这个矩阵乘法实现N个实例的特征向量压缩到一个特征向量.
        ## M: n_tasks, 512
        return M
        
     
    def forward(self, dict_images, attention_only=False):
        """
        dict_images: dict of images. images.shape: (N, 3, 256, 256)
        """
        Ms = []
        for modal_name in self.modal_names:
            if modal_name in dict_images:
                M = self.singlemodal_forward(modal_name, dict_images[modal_name], attention_only=False)
            else:
                M = torch.zeros(len(self.task_classes),512,requires_grad=False).to(self.attention_net['SWAN'][0].weight.device)
            Ms.append( M )
#             import pdb
#             pdb.set_trace()
#             Ms.append( self.attention_net[modal_name](images, attention_only=False) )
#             modals_loc[modal_name] = i
        
        # 多任务
        # 多任务情况下，M[i].shape: (n_tasks,512)
        if isinstance(self.classifier, torch.nn.modules.container.ModuleList):
            outputs = ()
            for i in range(len(self.classifier)):
                outputs += ( self.classifier[i]( torch.cat(Ms,dim=1)[i,:].unsqueeze(0) ), ) #
            return outputs
        # 单任务
        # 单任务情况下，M[i].shape: (1,512)
        else:
            output = self.classifier( torch.cat(Ms,dim=1) )
            # output = self.classifier( torch.cat([M for modal_name,M in Ms.items()],dim=1) )
#             cat_features = torch.cat([self.attention_net[modal_name](images, attention_only=False) for modal_name, images in dict_images.items()],dim=1)
#             output = self.classifier( cat_features )
#             ###################################################
#             # 调试用,仅取第一个模态
#             output = self.classifier( Ms[0] )
#             ###################################################
            return output


# +

class ResTOADMix(nn.Module):
    """
    MRI影像和临床特征混合模型.
    """
    def __init__(self, task_classes=3, dropout=False, requires_grad=True, loss_fn=None, input_channel=3):
        """
        task_classes: 这里的task可以是分类,也可以是回归,如是回归,则取值1. 
            例如第一个任务是3分类,第二个任务是回归,则task_classed=[3,1],当然,相应的损失函数也要主要根据任务而变化.
        loss_fn: tuple，损失函数。有多少个任务就多少个损失函数。
        """
        super().__init__()
        
        self.task_classes = task_classes
        self.loss_fn = loss_fn
        
        self.resnet_baseline = resnet50_baseline(pretrained=True,requires_grad=requires_grad)
        # 将平均池化改为最大池化
        self.resnet_baseline.avgpool = nn.AdaptiveMaxPool2d(output_size=1)
        
        if input_channel != 3:
            assert requires_grad==True, 'One should not change the first convolution layer when requires_grad is False.'
            self.resnet_baseline.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        size = [1024, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.LeakyReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if isinstance(task_classes,list):
            n_tasks=len(task_classes)
        elif isinstance(task_classes,int):
            n_tasks=1
        attention_net = Attn_Net_Gated(L=size[1], D=size[1], dropout=False, n_tasks=n_tasks)
        fc.append(attention_net)

        self.attention_net = nn.Sequential(*fc)
        if type(task_classes)==list and len(task_classes)>1:
            self.classifier = nn.ModuleList()
            for n_class in task_classes:
                self.classifier.append(nn.Linear(size[1]+2, n_class))
        elif type(task_classes)==list and len(task_classes)==1:
            self.classifier = nn.Linear(size[1]+2, task_classes[0])
        elif type(task_classes)==int:
            self.classifier = nn.Linear(in_features=size[1]+2, out_features=task_classes, bias=True)

    #def forward(self, images: torch.Tensor, clins: torch.Tensor, attention_only: bool=False):
    def forward(self, input_tensor: tuple, attention_only: bool=False):
        """
        return list of classification logit
        """
        images, clins = input_tensor[0], input_tensor[1]
#         import pdb
#         pdb.set_trace()
        # age normalization
        clins[:,1] = (clins[:,1]-60.0)/20.0
        
        
        # images.shape: (N, 3, 256, 256)
        # out: (N, 1024)
        h = self.resnet_baseline(images)
        # in
        ## h: N, 1024
        A, h = self.attention_net(h)
        # out
        ## A: N, n_tasks
        ## h: N, 512
        A = torch.transpose(A, 1, 0)
        ## A: n_tasks, N
        if attention_only:
            return A
        # apply attention
        A = F.softmax(A, dim=1)
        # in
        ## A: n_tasks, N
        ## h: N, 512
        M = torch.mm(A, h)# MIL的核心就是通过这个矩阵乘法实现N个实例的特征向量压缩到一个特征向量.
        ## M: n_tasks, 512
        if isinstance(self.classifier, torch.nn.modules.container.ModuleList):
            output_list = []
            for i in range(len(self.classifier)):
                output_list.append(self.classifier[i]( torch.cat([M[i].unsqueeze(0),clins],dim=1)))
            return output_list
        else:
            output = self.classifier(torch.cat([M,clins],dim=1))
            return output

# +
# if __name__ == "__main__":
#     # 单模态
#     model = ResTOAD(task_classes=[3,],dropout=True,requires_grad=True,input_channel=1)
#     images = torch.tensor(np.zeros((10,1,256,256))).to(torch.float32)
#     output = model(images)
#     print(output)

# +
# if __name__ == "__main__":
#     import torch.optim as optim
#     # 多模态
#     model = MultimodalResTOAD(modal_names=['SWAN','T1','T2'], task_classes=[3,], dropout=True, requires_grad=False)
#     list_images = {
#         'SWS': torch.tensor(np.zeros((10,3,256,256))).to(torch.float32), 
#         'T1': torch.tensor(np.zeros((10,3,256,256))).to(torch.float32),
#         'T2': torch.tensor(np.zeros((10,3,256,256))).to(torch.float32),
#     }
#     myloss = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001 )
#     label = torch.tensor([0]).float()
#     output = model(list_images)
#     loss = myloss(output,label)
#     loss.backward()
#     print(output)
#     print(model.attention_net['SWAN'][0].weight.grad)
    
#     for name, params in  model.attention_net['SWAN'][0].named_parameters():
#         print(name, params.requires_grad, params.grad)
# -

if __name__ == "__main__":
    # 单模态影像和临床特征混合模型
    model = ResTOADMix(task_classes=[3,],dropout=True,requires_grad=True,input_channel=1)
    images = torch.tensor(np.zeros((10,1,256,256))).to(torch.float32)
    clins = torch.tensor([1,57]).to(torch.float32)
    output = model(images,clins)
    print(output)
