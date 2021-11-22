import pandas as pd
import re
import numpy as np
import torch
from tqdm import tqdm


class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


def read_data(file_name, lines):
    data = pd.read_csv(file_name, nrows=lines)
    return data


def remove_comment(code):
    in_comment = 0
    output = []
    for line in code:
        if in_comment == 0:
            if line.find("/*") != -1:
                if line.find("*/") == -1:
                    in_comment = 1
            else:
                if line.find("//") != -1:
                    if line.find("//") > 0:
                        line = line[:line.find("//")]
                        output.append(line)
                else:
                    output.append(line)
        else:
            if line.find("*/") != -1:
                in_comment = 0
    return output


def merge_code(code):
    output = []
    combine_ = 0
    temp = ""
    for unit in code:
        if combine_ == 0:
            if len(unit) == 1 and unit.istitle():
                temp = temp + unit
                combine_ = 1
            else:
                output.append(unit)
        else:
            if len(unit) == 1 and unit.istitle():
                temp = temp + unit
            else:
                combine_ = 0
                output.append(temp)
                output.append(unit)
                temp = ""
    return output


def collect_code_data(data):
    big_code = []
    data_length = data.shape[0]
    for i in range(data_length):
        code_ = data.at[i, "code"]
        code_ = code_.splitlines()
        code_ = remove_comment(code_)
        code_ = " ".join(code_)
        code_ = re.sub('[^a-zA-Z0-9]', ' ', code_)
        code_ = re.sub(' +', ' ', code_)
        code_ = re.sub(r"([A-Z])", r" \1", code_).split()
        code_ = merge_code(code_)
        big_code.append(code_)
    return big_code


def collect_pdg(data):
    output = []
    data_length = data.shape[0]
    for i in tqdm(range(data_length), desc='collect_pdg'):
        pdg = data.at[i, "trees"].split("#")[12].splitlines()
        start_node = []
        end_node = []
        pair = {}
        for line in pdg:
            if line.find("-->>") != -1:
                a = line.split("-->>")[0]
                b = line.split("-->>")[1]
                belong_a = a.split("_")[-1].replace("(", "").replace(")\" ", "")
                belong_b = b.split("_")[-1].replace("(", "").replace(")\" ", "")
                if belong_a not in pair.keys():
                    pair[belong_a] = [belong_b]
                else:
                    pair[belong_a].append(belong_b)
        for key in pair.keys():
            for target in pair[key]:
                try:
                    key_int = int(key)
                    target_int = int(target)
                    start_node.append(key_int)
                    end_node.append(target_int)
                except ValueError:
                    pass
        output.append(torch.tensor([start_node, end_node]))
    return output


def generate_feature_1(data, dic, embed_dim):
    feature_1_set = []
    data_length = data.shape[0]
    for i in tqdm(range(data_length), desc='fea_1'):
        code_ = data.at[i, "code"]
        # print(code_)
        code_ = code_.splitlines()
        feature_set = []
        for line in code_:
            line = re.sub('[^a-zA-Z0-9]', ' ', line)
            line = re.sub(' +', ' ', line)
            line = re.sub(r"([A-Z])", r" \1", line).split()
            line = merge_code(line)
            sub_token_list = []
            for unit in line:
                if unit in dic.key_to_index:
                    sub_token_list.append(np.array(dic[unit]))
                else:
                    sub_token_list.append(np.zeros(embed_dim))
            feature_set.append(sub_token_list)
        feature_1_set.append(feature_set)
    return feature_1_set


def collect_ast(data):
    output = []
    data_length = data.shape[0]
    for i in tqdm(range(data_length), desc='collect_ast'):
        ast = data.at[i, "trees"].split("#")[1].splitlines()
        edge = {}
        value = {}
        belong = {}
        idfier = {}
        variable_type = {}
        temp_str = ""
        for line in ast:
            if line.find("-->>") != -1:
                if line.strip()[-1] != "\"":
                    temp_str = temp_str + line
                    continue
                else:
                    if temp_str != "":
                        line = temp_str
                        temp_str = ""
                a = line.split("-->> ")[0]
                b = line.split("-->> ")[1]
                id_a = a.split("_")[2].replace("(", "").replace(")", "")
                id_b = b.split("_")[2].replace("(", "").replace(")", "")
                if id_a not in edge.keys():
                    edge[id_a] = [id_b]
                else:
                    edge[id_a].append(id_b)
                t_a = None
                n_a = None
                temp_a = a.split("_joern_type_")
                if temp_a[-1].find("_joern_name_") != -1:
                    list_ta = temp_a[-1].split("_joern_name_")
                    t_a = list_ta[0].replace("(", "").replace(")", "")
                else:
                    list_ta = temp_a[-1].split("joern_name_")
                if list_ta[-1].find("_joern_line_") != -1:
                    list_na = list_ta[-1].split("_joern_line_")
                    n_a = list_na[0].replace("(", "").replace(")", "")
                if n_a:
                    v_a = n_a
                else:
                    v_a = t_a
                t_b = None
                n_b = None
                temp_b = b.split("_joern_type_")
                if temp_b[-1].find("_joern_name_") != -1:
                    list_tb = temp_b[-1].split("_joern_name_")
                    t_b = list_tb[0].replace("(", "").replace(")", "")
                else:
                    list_tb = temp_b[-1].split("joern_name_")
                if list_tb[-1].find("_joern_line_") != -1:
                    list_nb = list_tb[-1].split("_joern_line_")
                    n_b = list_nb[0].replace("(", "").replace(")", "")
                if n_b:
                    v_b = n_b
                else:
                    v_b = t_b
                if id_a not in value.keys():
                    value[id_a] = v_a
                if id_b not in value.keys():
                    value[id_b] = v_b
                belong_a = a.split("_")[-1].replace("(", "").replace(")", "")
                belong_b = b.split("_")[-1].replace("(", "").replace(")", "")
                if belong_a not in belong.keys():
                    belong[belong_a] = [id_a]
                else:
                    if id_a not in belong[belong_a]:
                        belong[belong_a].append(id_a)
                if belong_b not in belong.keys():
                    belong[belong_b] = [id_b]
                else:
                    if id_b not in belong[belong_b]:
                        belong[belong_b].append(id_b)
                if t_a == "IDENTIFIER":
                    if belong_a not in idfier.keys():
                        idfier[belong_a] = {n_a: t_a}
                    else:
                        if n_a not in idfier[belong_a].keys():
                            idfier[belong_a].append({n_a: t_a})
                if t_b == "IDENTIFIER":
                    if belong_b not in idfier.keys():
                        idfier[belong_b] = {n_b: t_b}
                    else:
                        if n_b not in idfier[belong_b].keys():
                            idfier[belong_b][n_b] = t_b
                if a.find("(METHOD)") != -1 and b.find("(METHOD_PARAMETER_IN)") != -1:
                    temp_b2 = b.split("_joern_code_")
                    if temp_b2[-1].find("_joern_type_") != -1:
                        list_tb2 = temp_b2[-1].split("_joern_type_")
                        node_code_b = list_tb2[0].replace("(", "").replace(")", "").replace("*", "")
                        node_code_b_new = [k for k in node_code_b.split(" ") if k != '']
                        if len(node_code_b_new) > 1:
                            code_type = node_code_b_new[-2]
                            code_name = node_code_b_new[-1]
                            if code_name not in variable_type.keys():
                                variable_type[code_name] = {belong_b.replace("\"", ""): code_type}
                            else:
                                variable_type[code_name][belong_b.replace("\"", "")] = code_type
                if a.find("(BLOCK)") != -1 and b.find("(<operator>.assignment)") != -1:
                    temp_b3 = b.split("_joern_code_")
                    if temp_b3[-1].find("_joern_type_") != -1:
                        list_tb3 = temp_b3[-1].split("_joern_type_")
                        node_code_b2 = list_tb3[0].replace("(", "").replace(")", "")
                        code_name = ""
                        for j in range(len(ast)):
                            if ast[j].find("-->>") != -1:
                                a_ = ast[j].split("-->>")[0]
                                b_ = ast[j].split("-->>")[1]
                                if a_.find(node_code_b2) != -1 and b_.find("(IDENTIFIER)") != -1:
                                    temp_b4 = b_.split("_joern_code_")
                                    if temp_b4[-1].find("_joern_type_") != -1:
                                        list_tb4 = temp_b4[-1].split("_joern_type_")
                                        code_name = list_tb4[0].replace("(", "").replace(")", "").replace("*", "")
                            if code_name != "":
                                break
                        if code_name != "":
                            source_code = data.at[i, "code"]
                            code_line = source_code.split("\n")[int(belong_b.replace("\"", ""))-1]
                            pos = code_line.find(node_code_b2)
                            if pos != -1:
                                code_type = code_line[0: pos].strip().split(" ")[-1]
                                if code_type != "":
                                    if code_name not in variable_type.keys():
                                        variable_type[code_name] = {
                                            belong_b.replace("\"", ""): code_type.replace("\t", "")}
                                    else:
                                        variable_type[code_name][belong_b.replace("\"", "")] = code_type.replace("\t", "")
        for key in idfier.keys():
            line_num = int(key.replace("\"", ""))
            for var in idfier[key].keys():
                if var in variable_type.keys():
                    marker = -1
                    for line_number in variable_type[var].keys():
                        line_num_checker = int(line_number)
                        if (line_num >= line_num_checker) and (line_num_checker > marker):
                            marker = line_num_checker
                            idfier[key][var] = variable_type[var][line_number]
        output.append([edge, value, belong, idfier])
    return output


def find_root(edges):
    appeared = []
    for edge in edges.keys():
        for node in edges[edge]:
            if node not in appeared:
                appeared.append(node)
    for key in edges.keys():
        if key not in appeared:
            return key


def collect_nodes(root, edges, order):
    order.append(root)
    if root in edges.keys():
        children = edges[root]
        for child in children:
            order = collect_nodes(child, edges, order)
    return order


def collect_tree_info(data):
    big_ast = []
    trees = collect_ast(data)
    for tree in trees:
        order = []
        root = find_root(tree[0])
        order = collect_nodes(root, tree[0], order)
        ast_ = []
        for node in order:
            try:
                ast_.append(tree[1][node])
            except KeyError:
                pass
        big_ast.append(ast_)
    return big_ast


def build_tree(edge_list, value_list, id, store_value):
    root = Tree()
    root.id = id
    store_value[id] = torch.tensor(value_list.get(id), dtype=torch.float)
    if edge_list.get(id):
        for child_id in edge_list.get(id):
            new_child, store_value = build_tree(edge_list, value_list, child_id, store_value)
            root.add_child(new_child)
    return root, store_value


def generate_feature_2(data, dic, embed_dim):
    trees_output = []
    trees = collect_ast(data)
    for tree in tqdm(trees, desc='fea_2'):
        subtree_set_edges = {}
        edges = tree[0]
        values = tree[1]
        for stmt in tree[2].keys():
            covered_nodes = tree[2][stmt]
            sub_edges = {}
            sub_values = {}
            for key in edges.keys():
                for node in edges[key]:
                    if key in covered_nodes and node in covered_nodes:
                        if key not in sub_edges.keys():
                            sub_edges[key] = [node]
                        else:
                            if node not in sub_edges[key]:
                                sub_edges[key].append(node)
            for key in sub_edges.keys():
                if key not in sub_values.keys():
                    if values[key] in dic.key_to_index:
                        sub_values[key] = np.array(dic[values[key]])
                    else:
                        sub_values[key] = np.zeros(embed_dim)
                for node in sub_edges[key]:
                    if node not in sub_values.keys():
                        if values[node] in dic.key_to_index:
                            sub_values[node] = np.array(dic[values[node]])
                        else:
                            sub_values[node] = np.zeros(embed_dim)
            root = find_root(sub_edges)
            sub_val = {}
            if root is None:
                continue
            subtree_t, subtree_v = build_tree(sub_edges, sub_values, root, sub_val)
            stmt = int(stmt.replace("\"", ""))
            subtree_set_edges[stmt] = [subtree_t, subtree_v]
        trees_output.append(subtree_set_edges)
    return trees_output


def generate_feature_3(data, dic, embed_dim):
    idf_output = []
    trees = collect_ast(data)
    for tree in tqdm(trees, desc='fea_3'):
        subtree_set_idf = {}
        idf = tree[3]
        for stmt in idf.keys():
            seq = []
            for key in idf[stmt].keys():
                if key in dic.key_to_index:
                    seq.append(dic[key])
                else:
                    seq.append(np.zeros(embed_dim))
                if idf[stmt][key] in dic.key_to_index:
                    seq.append(np.array(dic[idf[stmt][key]]))
                else:
                    seq.append(np.zeros(embed_dim))
            subtree_set_idf[stmt] = seq
        idf_output.append(subtree_set_idf)
    return idf_output


def find_control(control_list, stmt_num, stmt_list, seq, depth, limit):
    record = []
    if stmt_num in control_list.keys():
        control_stmt = control_list[stmt_num]
        seq = seq + stmt_list[int(control_stmt.replace("\"", "")) - 1]
        record.append(control_stmt)
    if depth < limit:
        for stmt in record:
            find_control(control_list, stmt, stmt_list, seq, depth + 1, limit)
    return seq


def generate_feature_4(data, dic, embed_dim, depth=2):
    control_output = []
    trees = collect_ast(data)
    for i in tqdm(range(len(trees)), desc='fea_4'):
        edges = trees[i][0]
        belong = trees[i][2]
        control = {}
        for key in edges.keys():
            for node in edges[key]:
                a = key
                b = node
                a_ = None
                b_ = None
                for line_num in belong.keys():
                    covered_set = belong[line_num]
                    if a in covered_set:
                        a_ = line_num
                    if b in covered_set:
                        b_ = line_num
                if a_ != b_ and a_ is not None and b_ is not None:
                    control[b_] = a_
        code_ = data.at[i, "code"]
        code_ = code_.splitlines()
        feature_set = []
        for line in code_:
            line = re.sub('[^a-zA-Z0-9]', ' ', line)
            line = re.sub(' +', ' ', line)
            line = re.sub(r"([A-Z])", r" \1", line).split()
            line = merge_code(line)
            sub_token_list = []
            for unit in line:
                if unit in dic.key_to_index:
                    sub_token_list.append(np.array(dic[unit]))
                else:
                    sub_token_list.append(np.zeros(embed_dim))
            feature_set.append(sub_token_list)
        embedding = {}
        for line_num in belong.keys():
            output_seq = find_control(control, line_num, feature_set, [], 1, depth)
            embedding[line_num] = output_seq
        control_output.append(embedding)
    return control_output


def collect_data_dependency(data):
    output = []
    data_length = data.shape[0]
    for i in range(data_length):
        dfg = data.at[i, "trees"].split("#")[3].splitlines()
        out_ = {}
        in_ = {}
        all_ = []
        for line in dfg:
            if line.find("-->>") != -1:
                a = line.split("-->>")[0]
                b = line.split("-->>")[1]
                id_a = a.split("_")[-1].replace("(", "").replace(")", "")
                id_b = b.split("_")[-1].replace("(", "").replace(")", "")
                if id_a not in out_.keys():
                    out_[id_a] = [id_b]
                else:
                    out_[id_a].append(id_b)

                if id_b not in in_.keys():
                    in_[id_b] = [id_a]
                else:
                    in_[id_b].append(id_a)
                if id_a not in all_:
                    all_.append(id_a)
                if id_b not in all_:
                    all_.append(id_b)
        output.append([out_, in_, all_])
    return output


def find_df(out_list, in_list, stmt_num, stmt_list, seq, depth, limit, label):
    record_out = []
    record_in = []
    if label != 2:
        if stmt_num in out_list.keys():
            for stmt in out_list[stmt_num]:
                try:
                    seq = seq + stmt_list[int(stmt.replace("\"", '')) - 1]
                    record_out.append(stmt)
                except:
                    pass
    if label != 1:
        if stmt_num in in_list.keys():
            for stmt in in_list[stmt_num]:
                try:
                    seq = seq + stmt_list[int(stmt.replace("\"", "")) - 1]
                    record_in.append(stmt)
                except:
                    pass
    if depth < limit:
        for stmt in record_out:
            seq = find_df(out_list, in_list, stmt, stmt_list, seq, depth + 1, limit, 1)
        for stmt in record_in:
            seq = find_df(out_list, in_list, stmt, stmt_list, seq, depth + 1, limit, 2)
    return seq


def generate_feature_5(data, dic, embed_dim, depth=2):
    data_output = []
    dfgs = collect_data_dependency(data)
    for i in tqdm(range(len(dfgs)), desc='fea_5'):
        out_ = dfgs[i][0]
        in_ = dfgs[i][1]
        all_ = dfgs[i][2]
        feature_set = []
        code_ = data.at[i, "code"]
        code_ = code_.splitlines()
        for line in code_:
            line = re.sub('[^a-zA-Z0-9]', ' ', line)
            line = re.sub(' +', ' ', line)
            line = re.sub(r"([A-Z])", r" \1", line).split()
            line = merge_code(line)
            sub_token_list = []
            for unit in line:
                if unit in dic.key_to_index:
                    sub_token_list.append(np.array(dic[unit]))
                else:
                    sub_token_list.append(np.zeros(embed_dim))
            feature_set.append(sub_token_list)
        embedding = {}
        for stmt in all_:
            output_seq = find_df(out_, in_, stmt, feature_set, [], 1, depth, 0)
            embedding[stmt] = output_seq
        data_output.append(embedding)
    return data_output
