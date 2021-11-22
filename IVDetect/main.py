import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from torch import nn
import gc
import vul_model
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse

tqdm.pandas()


class MyDatset(Dataset):
    def __init__(self, _datapoint_files, file_dir):
        self.datapoint_files = _datapoint_files
        self.file_dir = file_dir

    def __getitem__(self, index):
        graph_file = os.getcwd() + "/{}".format(self.file_dir) + "{}".format(self.datapoint_files[index])
        graph = torch.load(graph_file)
        return graph

    def __len__(self):
        return len(self.datapoint_files)


# only batch = 1
def collate_batch(batch):
    _data = batch[0]
    return _data


def start_training(starting_epochs, params):
    train_path = args.processed_dir + 'train_graph/'
    train_files = [f for f in os.listdir(train_path) if
                   os.path.isfile(os.path.join(train_path, f))]
    train_dataset = MyDatset(train_files, train_path)

    test_path = args.processed_dir + 'test_graph/'
    test_files = [f for f in os.listdir(test_path) if
                  os.path.isfile(os.path.join(test_path, f))]
    test_dataset = MyDatset(test_files, test_path)

    valid_path = args.processed_dir + 'valid_graph/'
    valid_files = [f for f in os.listdir(valid_path) if
                   os.path.isfile(os.path.join(valid_path, f))]
    valid_dataset = MyDatset(valid_files, valid_path)

    # shuffle is done with generating dataset, in order to continue training pre-trained model
    # shuffle is turned off in data loader
    train_loader = DataLoader(train_dataset, collate_fn=collate_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate_batch, shuffle=False)
    print("train", len(train_dataset), "valid", len(valid_dataset), "test", len(test_dataset))
    max_epochs = params['epochs']
    trainer(max_epochs=max_epochs, starting_epochs=starting_epochs, _trainLoader=train_loader,
            _validLoader=valid_loader, _testLoader=test_loader, _params=params)


def evaluate_metrics(model, _loader, device):
    print('evaluate >')
    model.eval()
    with torch.no_grad():
        all_predictions, all_targets, all_probs = [], [], []
        for graph in _loader:
            graph = graph.cuda()
            target = graph.y
            out = model(graph.my_data, graph.edge_index)
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            prob_1 = out.cpu().detach().numpy()[0][1]
            all_probs.append(prob_1)
            all_predictions.append(pred)
            all_targets.append(target)
            del graph.my_data, graph.edge_index, graph.y, graph, out
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc_score = round(auc(fpr, tpr) * 100, 2)
        acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        print(acc)
        precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        matrix = confusion_matrix(all_targets, all_predictions)
        target_names = ['non-vul', 'vul']
        report = classification_report(all_targets, all_predictions, target_names=target_names)
        result = "auc: {}".format(auc_score) + " acc: {}".format(acc) + " precision: {}".format(precision) + " recall: {}".format(recall) + " f1: {}".format(f1) + " \nreport:\n{}".format(report) + "\nmatrix:\n{}".format(matrix)
        print(result)
    model.train()
    return auc_score


def trainer(max_epochs, starting_epochs, _trainLoader, _validLoader, _testLoader, _params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vul_model.Vulnerability(h_size=_params['hidden_size'], num_node_feature=5, num_classes=2,feature_representation_size=128,drop_out_rate=_params['dropout_rate'], num_conv_layers=_params['num_conv_layers'])
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08)
    print("learning rate : ", optimizer.param_groups[0]['lr'])
    criterion = nn.CrossEntropyLoss()
    starting_epochs += 1
    valid_auc = 0
    for e in range(starting_epochs, max_epochs):
        train_loss = train(e, _trainLoader, model, criterion, optimizer, device)
        torch.save(model, os.getcwd() + "/model/trained_model_{}.pt".format(e))
        valid_auc = evaluate_metrics(model=model, _loader=_validLoader, device=device)
        # nni.report_intermediate_result(valid_auc)
        if train_loss < 0.3:
            break
        gc.collect()
    # nni.report_final_result(valid_auc)


def train(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    model.train()
    for index, graph in enumerate(tqdm(_trainLoader)):
        if index % 500 == 0:
            print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        if device != 'cpu':
            graph = graph.cuda()
        target = graph.y
        optimizer.zero_grad()
        out = model(graph.my_data, graph.edge_index)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        correct += predicted.eq(target).sum().item()
        del graph.my_data, graph.edge_index, graph.y, graph, predicted, out
    avg_train_loss = train_loss / len(_trainLoader)
    acc = correct / len(_trainLoader)
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    gc.collect()
    return avg_train_loss


if __name__ == '__main__':
    torch.manual_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, help='dir of processed split datapoints',
                        default='data/')
    parser.add_argument('--out_dir', type=str, help='output of trained model state',
                        default='result/')
    parser.add_argument('-v', '--evaluate', action='store_true')
    args = parser.parse_args()
    params = {'hidden_size': 128, 'lr': 0.0001, 'dropout_rate': 0.3, 'epochs': 100, 'num_conv_layers': 3}
    print("reading processed datas from {}".format(args.processed_dir))
    start_training(0, params)
