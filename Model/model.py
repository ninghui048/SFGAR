import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from .model_utils import evaluate
from .blocks import Extractor, Classifier, CrossEntropy
global rate
rate = 0
def estimate_gaussian(h):
    mu = h.mean(dim=0)       
    var = h.var(dim=0) + 1e-8  
    return mu, var
def kl_divergence(mu, var):
    return 0.5 * torch.sum(-torch.log(var) - 1 + var + mu ** 2)
def distillation_loss(student_logits, teacher_logits, T=1.0):
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction='batchmean') * (T * T)
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.layer_unit_count_list = args.layer_unit_count_list
        self.layer_count = len(self.layer_unit_count_list)
        self.Extractor = Extractor(self.layer_unit_count_list[:-1], args)
        self.Classifier = Classifier(self.layer_unit_count_list[-2:])
        self.models = nn.ModuleList([self.Extractor, self.Classifier])
        self.CrossEntropy = CrossEntropy(args)
        self.CrossEntropyLoss = nn.CrossEntropyLoss().to(args.device)
        self.MSELoss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    def AdversarialLoss(self,pred_1,pred_2,pred_3=None):
        cls_loss_1 = self.CrossEntropyLoss(
            pred_1,
            torch.zeros(pred_1.size(0)).type(torch.LongTensor).to(self.args.device)
        )
        cls_loss_2 = self.CrossEntropyLoss(
            pred_2,
            torch.ones(pred_2.size(0)).type(torch.LongTensor).to(self.args.device)
        )
        if pred_3 is not None:
            cls_loss_3 = self.CrossEntropyLoss(
                pred_3,
                torch.ones(pred_3.size(0)).type(torch.LongTensor).to(self.args.device)
            )
            return cls_loss_1 + cls_loss_2 + cls_loss_3
        return cls_loss_1 + cls_loss_2
    def train_source(self, source_data, optimizer, criterion, epoch):
        self.enable_source()
        outputs = self.forward(source_data)[source_data.train_mask]
        labels = source_data.y[source_data.train_mask]
        loss = self.CrossEntropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), evaluate(outputs, labels, self.args.metric)
    def train_target(self, target_data,source_model, GraphG,KLmodel,DAmodel,target_optimizer):
        self.enable_target()
        global rate
        rate = self.args.rate
        target_optimizer.zero_grad()
        X,A = GraphG()
        freeze_embed = source_model.Extractor(X,A)
        adapt_embed = self.Extractor(X,A)
        target_embed = self.Extractor(target_data.x,target_data.edge_index)
        mu, var = estimate_gaussian(freeze_embed)
        kloss = kl_divergence(mu,var)
        freeze_embed_mu = freeze_embed.mean(dim=1)
        adapt_embed_mu = adapt_embed.mean(dim=1)
        freeze_embed_var = freeze_embed.var(dim=1)
        adapt_embed_var = adapt_embed.var(dim=1)
        l = self.MSELoss(freeze_embed_var,adapt_embed_var) + self.MSELoss(freeze_embed_mu,adapt_embed_mu)
        source_domain_freeze_preds = KLmodel(freeze_embed)
        generator_domain_preds = KLmodel(adapt_embed)
        advloss_1 = self.AdversarialLoss(source_domain_freeze_preds,generator_domain_preds)
        source_domain_adap_preds = DAmodel(adapt_embed)
        target_domain_preds = DAmodel(target_embed)
        advloss_2 = self.AdversarialLoss(source_domain_adap_preds,target_domain_preds)
        outputs = source_model.Classifier(target_embed)
        loss = - self.args.kl_w * kloss         + \
                 self.args.mse_w * l +  \
                 self.args.adv1_w * advloss_1  +\
                 self.args.adv2_w * advloss_2   
        loss.backward()
        target_optimizer.step()
        return loss.item(),evaluate(outputs, target_data.y, self.args.metric)
    def forward(self, data):
        x = self.Extractor(data.x, data.edge_index)
        x = self.Classifier(x)
        return x
    def enable_source(self):
        for model in self.models:
            model.train()
    def enable_target(self):
        self.Extractor.train()
        self.Classifier.train()
class Generator(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(Generator, self).__init__()
        self.args = args
        self.dropout = dropout
        self.A_linear = nn.Linear(self.args.num_features,self.args.num_features)
        self.X_linear = nn.Linear(self.args.num_features,self.args.num_features)
        self.init_weight(self.A_linear)
        self.init_weight(self.X_linear)
        self.sample =  torch.randn(self.args.num_N,self.args.num_features ).to(self.args.device)
    def init_weight(self,linear):
        nn.init.kaiming_normal_(linear.weight)
        if linear.bias is not None:
            nn.init.constant_(linear.bias, 0)
    def forward(self, ):
        X = self.X_linear(self.sample).to(self.args.device)
        A = self.A_linear(self.sample).to(self.args.device)
        A = torch.mm(A, A.T).to(self.args.device)
        A = torch.sigmoid(A)
        K = self.args.num_E
        num_nodes = A.size(0)
        A[torch.arange(num_nodes), torch.arange(num_nodes)] = float('-inf')
        A_flat = A.flatten()
        topk_indices = torch.topk(A_flat, K).indices
        row = topk_indices // A.size(1)
        col = topk_indices % A.size(1)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = to_undirected(edge_index)
        return X,edge_index
class Domain_model(nn.Module):
    def __init__(self,args):
        super(Domain_model,self).__init__()
        self.args = args
        self.domain_model = nn.Sequential(
                    GRL(),
                    nn.Linear(self.args.layer_unit_count_list[-2], 40),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(40, 2),
                    ).to(self.args.device)
    def forward(self,embed):
        return self.domain_model(embed)
class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        global rate
        grad_output = grad_output.neg() * rate
        return grad_output, None