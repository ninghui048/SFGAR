import torch
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score,accuracy_score
def init_layer(args):
    layer_unit_count_list = [args.num_features] 
    if args.num_layers == 2:
        layer_unit_count_list.extend([128])
    elif args.num_layers == 3:
        layer_unit_count_list.extend([128, 128])
    elif args.num_layers == 4:
        layer_unit_count_list.extend([128, 128, 128])
    elif args.num_layers == 5:
        layer_unit_count_list.extend([128, 128, 128, 128])
    elif args.num_layers == 6:
        layer_unit_count_list.extend([128,128,128,128,128])
    elif args.num_layers == 7:
        layer_unit_count_list.extend([128,128,128,128,128,128])
    layer_unit_count_list.append(args.num_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vars(args)["device"] = device
    vars(args)["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    vars(args)["layer_unit_count_list"] = layer_unit_count_list
def save_model(args, prefix, model):
    torch.save({'model_state_dict': model.state_dict()}, f"./checkpoint/{args.model_name}/{prefix}_{args.random_seed}.pkl")
def load_model(args, prefix, model):
    state_dict = torch.load(f"./checkpoint/{args.model_name}/{prefix}_{args.random_seed}.pkl", map_location=args.device)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model
def predict(output):
    labels = output.argmax(dim=1)    
    return labels
def evaluate(output, labels, metric):
    preds = predict(output)
    corrects = preds.eq(labels)
    labels = labels.cpu().numpy()
    num_labels = np.max(labels) + 1
    preds = torch.argmax(output, dim = 1).cpu().numpy()
    acc = accuracy_score(labels,preds)
    return acc
def test(model, args, data, criterion, mode = 'valid'):
    model.eval()
    outputs = model(data)
    if mode == 'valid':
        outputs = outputs[data.val_mask]
        labels = data.y[data.val_mask]
    else:
        labels = data.y
    loss = criterion(outputs, labels)
    acc = evaluate(outputs, labels, args.metric)
    return loss, acc
def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)