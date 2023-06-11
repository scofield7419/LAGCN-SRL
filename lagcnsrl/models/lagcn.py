# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    """
    Vanilla GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)  # hidden: seq_len * hidden
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class LabeledGraphConvolution(nn.Module):
    """
    Label-aware GCN layer
    """
    def __init__(self, type_dim, in_features, out_features, type, bias=True):
        super(LabeledGraphConvolution, self).__init__()
        self.type = type
        self.fc =  nn.Linear(type_dim, in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj, dep_embed):
        batch_size, max_len, feat_dim = text.shape
        if 'bert' not in self.type:
            dep_embed = self.fc(dep_embed)
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)
        val_sum = val_us + dep_embed
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)
        hidden = torch.matmul(val_sum, self.weight)  # hidden: seq_len * hidden
        output = hidden.transpose(1,2) * adj_us
        output = torch.sum(output, dim=2)

        if self.bias is not None:
            return output + self.bias
        else:
            return output



class LAGCN(nn.Module):
    def __init__(self, bert, opt):
        super(LAGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.lgc1 = LabeledGraphConvolution(opt.type_dim, opt.bert_dim, opt.bert_dim, opt.default_type_dim)
        self.lgc1 = LabeledGraphConvolution(opt.type_dim, opt.bert_dim, opt.bert_dim, opt.default_type_dim)
        self.lgc1 = LabeledGraphConvolution(opt.type_dim, opt.bert_dim, opt.bert_dim, opt.default_type_dim)
        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.fc_single = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.bert_dropout)
        self.ensemble_linear = nn.Linear(1,3)
        self.ensemble = nn.Parameter(torch.FloatTensor(3, 1))
        self.dep_embedding = nn.Embedding(opt.type_num, opt.type_dim, padding_idx=0)


    def get_attention(self, val_out, dep_embed,adj):
        batch_size, max_len, feat_dim = val_out.shape
        # attention_score = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32, device='cuda')
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, dep_embed), -1)
        atten_expand = (val_cat * val_cat.transpose(1,2))
        attention_score = torch.sum(atten_expand, dim=-1)
        #scale
        attention_score = attention_score / np.power(feat_dim, 0.5)
        # softmax
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score, adj) # mask
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def get_avarage(self, aspect_indices, x):
        aspect_indices_us = torch.unsqueeze(aspect_indices, 2)  # aspect indices has no CLS
        x_mask = x * aspect_indices_us
        aspect_len = (aspect_indices_us != 0).sum(dim=1)
        x_sum = x_mask.sum(dim=1)
        x_av = torch.div(x_sum, aspect_len)
        return x_av


    def forward(self, inputs, output_attention=False):
        text_bert_indices, bert_segments_ids, valid_ids, adj = inputs[0],inputs[1],inputs[2],inputs[3]
        context_indices, aspect_indices, text_bert_single_indices = inputs[4], inputs[5], inputs[6]
        graph = inputs[7]
        dep_embed = self.dep_embedding(graph)

        # bert input: sentence pair or single sentence
        if 'pair' in self.opt.data_input :
            sequence_output, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        else:
            sequence_output, pooled_output = self.bert(text_bert_single_indices)

        # remove word split e.g. ##ing
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')

        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        valid_output = self.dropout(valid_output)

        # only use sentence, remove aspect hidden states and CLS
        context_indices = torch.unsqueeze(context_indices, 2)
        valid_output = valid_output * context_indices
        valid_output = torch.cat((valid_output[:,1:,:],valid_output[:,0:1,:]), 1)

        attention_score_for_output = []
        if self.opt.layer == "gcn" :
            # vanilla GCN layers
            adj = adj.float()
            x_1 = F.relu(self.gc1(valid_output, adj))
            x_2 = F.relu(self.gc2(x_1, adj))
            x_3 = F.relu(self.gc3(x_2, adj))

        if self.opt.layer == "agcn" :
            # vanilla GAT layers with neighbor attention
            attention_score = self.get_attention(valid_output, dep_embed, adj)
            x_1 = F.relu(self.gc1(valid_output, attention_score))
            attention_score = self.get_attention(x_1, dep_embed, adj)
            x_2 = F.relu(self.gc2(x_1, attention_score))
            attention_score = self.get_attention(x_2, dep_embed, adj)
            x_3 = F.relu(self.gc3(x_2, attention_score))

        if self.opt.layer == "lgcn":
            #LAGCN layers without neighbor attention
            x_1 = F.relu(self.lgc1(valid_output, adj,dep_embed))
            x_2 = F.relu(self.lgc1(x_1, adj,dep_embed))
            x_3 = F.relu(self.lgc1(x_2, adj,dep_embed))

        if self.opt.layer == "lagcn":
            #LAGCN layers with attention
            attention_score = self.get_attention(valid_output, dep_embed, adj)
            attention_score_for_output.append(attention_score)
            x_1 = F.relu(self.lgc1(valid_output, attention_score,dep_embed))
            attention_score = self.get_attention(x_1, dep_embed, adj)
            attention_score_for_output.append(attention_score)
            x_2 = F.relu(self.lgc1(x_1, attention_score,dep_embed))
            attention_score = self.get_attention(x_2, dep_embed, adj)
            attention_score_for_output.append(attention_score)
            x_3 = F.relu(self.lgc1(x_2, attention_score,dep_embed))

        # average pool
        x_1_pool = self.get_avarage(aspect_indices,x_1)
        x_2_pool = self.get_avarage(aspect_indices, x_2)
        x_3_pool = self.get_avarage(aspect_indices, x_3)

        # no ensemble
        if self.opt.ensemble == "none":
            x = x_3_pool
            x = self.dropout(x)

        if self.opt.ensemble == "average":
            x_pool = torch.stack((x_1_pool, x_2_pool, x_3_pool), -1)
            x = torch.matmul(x_pool, F.softmax(self.ensemble_linear.weight, dim=0))
            x = x.squeeze(dim=-1)
            x = self.dropout(x)

        if self.opt.ensemble == "attention":
            x_pool = torch.stack((x_1_pool, x_2_pool, x_3_pool), 1)
            pooled_output = torch.unsqueeze(pooled_output, -1)
            alpha_mat = torch.matmul(x_pool, pooled_output)
            alpha = F.softmax(alpha_mat, dim=1)
            x = torch.matmul(alpha.transpose(1,2), x_pool).squeeze(1)

        output = self.fc_single(x)
        if output_attention is True:
            return output, attention_score_for_output
        else:
            return output