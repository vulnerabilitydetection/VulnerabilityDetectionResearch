import os
import random

import torch
from tqdm import tqdm

graph_path = 'data/pyg_graph/'
graph_files = [f for f in os.listdir(graph_path) if
               os.path.isfile(os.path.join(graph_path, f))]

index = 0
vul_list = []
nonvul_list = []
for graph_file in tqdm(graph_files):
    graph = torch.load(os.getcwd() + "/" + graph_path + "/" + graph_file)
    if graph.y == 0:
        nonvul_list.append(graph_file)
    else:
        vul_list.append(graph_file)


# 1/5 sampler
def sampler(_vul_list, _nonvul_list):
    random.shuffle(_vul_list)
    random.shuffle(_nonvul_list)
    _vul_list = _vul_list[:(int(len(_vul_list) / 5))]
    _nonvul_list = _nonvul_list[:(int(len(_nonvul_list) / 5))]
    return _vul_list, _nonvul_list

# do a 1/10 sample
# vul_list, nonvul_list = sampler(vul_list, nonvul_list)

len_vul = len(vul_list)
len_nonvul = len(nonvul_list)
ratio = len_nonvul / len_vul
print("vul: " + str(len_vul) + " non_vul: " + str(len_nonvul) + " ratio: " + str(ratio))
vul_partition = int(0.8 * len_vul)
train_list = vul_list[:vul_partition] + nonvul_list[:vul_partition]
random.shuffle(train_list)
non_vul_valid = int(int(0.1 * len_vul) * ratio)
valid_list = vul_list[vul_partition:vul_partition + int(0.1 * len_vul)] + nonvul_list[
                                                                          vul_partition:vul_partition + non_vul_valid]
random.shuffle(valid_list)
test_list = vul_list[vul_partition + int(0.1 * len_vul):] + nonvul_list[
                                                            vul_partition + non_vul_valid:vul_partition + non_vul_valid + non_vul_valid]

print("train: " + str(len(train_list)) + "\tvalid:" + str(len(valid_list)) + "\ttest: " + str(len(test_list)) + "\t")

for index, train in enumerate(tqdm(train_list)):
    train_graph = torch.load(os.getcwd() + "/" + graph_path + "/"  + train)
    torch.save(train_graph, os.getcwd() + "/data/train_graph/data_{}.pt".format(index))

for index, valid in enumerate(tqdm(valid_list)):
    valid_graph = torch.load(os.getcwd() + "/" + graph_path + "/"  + valid)
    torch.save(valid_graph, os.getcwd() + "/data/valid_graph/data_{}.pt".format(index))

for index, test in enumerate(tqdm(test_list)):
    test_graph = torch.load(os.getcwd() + "/" + graph_path + "/"  + test)
    torch.save(test_graph, os.getcwd() + "/data/test_graph/data_{}.pt".format(index))
