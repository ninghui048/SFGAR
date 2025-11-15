import os
import sys
import torch
import argparse
import itertools
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from Model import (Model,
                   Domain_model,
                   Generator,
                   test,
                   TwitchDataset,
                   EllipticDataset,
                   init_layer,
                   save_model, 
                   load_model,
                   set_random_seed
                   )
def train_procedure(args, source_model, target_adapt_model,source_optimizer, target_optimizer, criterion, source_data, target_data,KLmodel,DAmodel,GraphG):
    best_val_acc = 0.0
    best_val_loss = 1e8
    if args.is_source_train:
        source_model.train()
        for epoch in range(args.source_epochs):
            train_loss, train_accuracy = source_model.train_source(source_data, source_optimizer, criterion, epoch)
            val_loss, val_accuracy = test(source_model, args, source_data, criterion, 'valid')
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_loss = val_loss
                save_model(args, "source", source_model)
        print('Best source valid acc\t{:.6f}\t Best valid loss\t{:.6f}'.format(best_val_acc, best_val_loss))
    else:
        source_model = load_model(args, "source", source_model)
    if args.is_baseline:
        loss,acc = test(source_model, args, target_data, criterion, "test")
    else:
        source_model = load_model(args, "source", source_model)
        target_adapt_model = load_model(args, "source", target_adapt_model)
        KLmodel.train()
        DAmodel.train()
        GraphG.train()
        best_test_acc = 0.0
        for epoch in range(args.target_epochs):
            loss,test_accuracy = target_adapt_model.train_target(target_data,source_model, GraphG,KLmodel,DAmodel,target_optimizer)
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_epoch = epoch
                save_model(args, "target", target_adapt_model)
            print(f"epoch:{epoch} test acc: {test_accuracy},best_acc: {best_test_acc},loss:{loss}")
        return test_accuracy
def main(args):
    num_nodes_dict = {"E10":46647,"M10":34333,"S10":58097,"DE":9498,"EN":7126,"FR":6549}
    num_label_dict = {"S10":3,"M10":3,"E10":3,"DE":2,"EN":2,"FR":2}
    num_feature_dict = {"S10":165,"M10":165,"E10":165,"DE":3170,"EN":3170,"FR":3170}
    vars(args)['num_label'] = num_label_dict[args.source_dataset]
    vars(args)['num_features'] = num_feature_dict[args.source_dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.source_dataset in ["DE","EN","FR"]:
        vars(args)['num_source_nodes'] = num_nodes_dict[args.source_dataset]
        source_data = TwitchDataset("./data/Twitch/{}".format(args.source_dataset), name=args.source_dataset, )
        source_data = source_data[0]
        source_data = source_data.to(device)
    elif args.source_dataset in ["S10","M10","E10"]:
        source_data = EllipticDataset("./data/Elliptic/{}".format(args.source_dataset), name=args.source_dataset, )
        source_data = source_data[0]
        vars(args)['num_source_nodes'] = source_data.x.size(0)
        source_data = source_data.to(device)
    else:
        print(r'Dataset need to be defined!')
        sys.exit()
    if args.target_dataset in ["DE","EN","FR"]:
        target_data = TwitchDataset("./data/Twitch/{}".format(args.target_dataset), name=args.target_dataset, )
        target_data = target_data[0]
        vars(args)['num_target_nodes'] = num_nodes_dict[args.target_dataset]
        target_data = target_data.to(device)
    elif args.target_dataset in ["S10","M10","E10"]:
        vars(args)['num_target_nodes'] = num_nodes_dict[args.target_dataset]
        target_data = EllipticDataset("./data/Elliptic/{}".format(args.target_dataset), name=args.target_dataset, )
        target_data = target_data[0]
        target_data = target_data.to(device)
    else:
        print(r'Dataset need to be defined!')
        sys.exit()
    last_acc = []
    for i in range(10,60,10):
        set_random_seed(i)
        init_layer(args)
        source_model = Model(args)
        target_adapt_model = Model(args)
        KLmodel = Domain_model(args)
        DAmodel = Domain_model(args)
        GraphG = Generator(args)
        source_model.to(device)
        target_adapt_model.to(device)
        KLmodel.to(device)
        DAmodel.to(device)
        GraphG.to(device)
        source_optimizer = Adam(source_model.parameters(), lr=args.source_lr, weight_decay=args.source_wd)
        target_adapt_models = [target_adapt_model, KLmodel,DAmodel, GraphG]
        params = itertools.chain(*[model.parameters() for model in target_adapt_models])
        target_optimizer = Adam(params, lr=args.target_lr, weight_decay=args.target_wd)
        source_criterion = nn.CrossEntropyLoss()
        test_accuracy = train_procedure(args, source_model, target_adapt_model,source_optimizer, target_optimizer, source_criterion, source_data, target_data,KLmodel,DAmodel,GraphG)
        last_acc.append(test_accuracy)
    acc_mean = np.mean(last_acc,axis=0)
    acc_std = np.std(last_acc,axis=0)
    print(f"source dataset:{args.source_dataset}, target dataset:{args.target_dataset}, acc_mean:{acc_mean:.4f}, acc_std:{acc_std:.4f}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', type=str, default="EN")
    parser.add_argument('--target_dataset', type=str, default="DE")
    parser.add_argument('--source_epochs', type=int, default=100)
    parser.add_argument('--target_epochs', type=int, default=100) 
    parser.add_argument('--source_lr', type=float, default=0.001)
    parser.add_argument('--target_lr', type=float, default=0.00001)
    parser.add_argument('--source_wd', type=float, default=5e-4)
    parser.add_argument('--target_wd', type=float, default=1e-5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--gnn_model', type=str, default="GCN",)
    parser.add_argument('--head', type=int, default=4) 
    parser.add_argument('--metric', type=str, default="micro") 
    parser.add_argument('--model_name', type=str, default="ENDE") 
    parser.add_argument('--rate', type=float, default=0.05)
    parser.add_argument('--num_E', type=int, default=10000)
    parser.add_argument('--num_N', type=int, default=10000)
    parser.add_argument('--kl_w', type=float, default=0.01)  
    parser.add_argument('--mse_w', type=float, default=0.1)
    parser.add_argument('--adv1_w', type=float, default=1)
    parser.add_argument('--adv2_w', type=float, default=1)
    parser.add_argument('--activation', type=str, default="prelu")  
    parser.add_argument('--is_source_train', type=bool, default=True)
    parser.add_argument('--is_baseline', type=bool, default=False)
    parser.add_argument('--source_model_path', type=str, default=" ")
    parser.add_argument('--random_seed', type=int, default=10)
    args = parser.parse_args()
    main(args)