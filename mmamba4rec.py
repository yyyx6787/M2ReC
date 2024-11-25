import torch
from torch import nn
# from mamba_ssm import Mamba as Mambao
from mamba.mamba_ssm.modules.mamba_simple import Mamba
# from mamba.mamba_ssm.modules.mamba_simple_v2 import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import numpy as np
import pickle
import pandas as pd
def load_data(file):
	data_load_file = []
	file_1 = open(file, "rb")
	# file_1.seek(0)
	data_load_file = pickle.load(file_1)
	return data_load_file



class Mamba4Rec(SequentialRecommender):
	def __init__(self, config, dataset):
		super(Mamba4Rec, self).__init__(config, dataset)

		self.hidden_size = config["hidden_size"]
		self.loss_type = config["loss_type"]
		self.num_layers = config["num_layers"]
		self.dropout_prob = config["dropout_prob"]
		
		# Hyperparameters for Mamba block
		self.d_state = config["d_state"]
		self.d_conv = config["d_conv"]
		self.expand = config["expand"]
		self.data = config["dataset"]
		
		self.llm_vec = load_data('./dataset/{}/item_meta_emb.pkl'.format(self.data))
		self.llm_vec[0] = np.array([0.0]*1024)
		
		
		self.llm_matrix = pd.DataFrame([self.llm_vec[i] for i in range(len(self.llm_vec))])
		# tmp = torch.tensor(self.llm_matrix.to_numpy())
		# idx = torch.nonzero(tmp).T  
		# data = tmp[idx[0],idx[1]]
		# coo_a = torch.sparse_coo_tensor(idx, data, tmp.shape)
		# self.score = torch.spmm(coo_a,coo_a.T)
		# print("check:", self.score.size())
		# println()
		self.item_embedding = nn.Embedding(
			self.n_items, self.hidden_size, padding_idx=0
		)
			
		self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(self.dropout_prob)
		self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
		# self.position_embedding = nn.Embedding(
  #           self.max_seq_length + 1, self.hidden_size)
		
		self.mamba_layers = nn.ModuleList([
			MambaLayer(
				d_model=self.hidden_size,
				d_state=self.d_state,
				d_conv=self.d_conv,
				expand=self.expand,
				dropout=self.dropout_prob,
				num_layers=self.num_layers,
			) for _ in range(self.num_layers)
		])
		# self.mamba_layers = nn.ModuleList([
		# 	Bimamba(config) for _ in range(self.num_layers)
		# 	])
		if self.loss_type == "BPR":
			self.loss_fct = BPRLoss()
		elif self.loss_type == "CE":
			self.loss_fct = nn.CrossEntropyLoss()
		else:
			raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

		self.apply(self._init_weights)
		self.l2 = nn.Linear(20375, 64)
		self.l1 = model = nn.Sequential(
          nn.Conv1d(200, 200, 4, stride=3),
          nn.GELU(),
          nn.Linear(341, 64),
          nn.GELU()
        )
		# self.alpha = nn.Parameter(torch.ones(150, self.hidden_size) * 0.98, requires_grad=True)
		self.alpha = nn.Parameter(torch.FloatTensor(1)*0.5, requires_grad=True)
		self.beta = nn.Parameter(torch.FloatTensor(1)*0.5, requires_grad=True)


	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	


	def forward(self, item_seq, item_seq_len):
		

		item_index = item_seq.clone()[0].cpu().tolist()
		# item_index = item_seq.clone()[0].cuda()
		item_llm_vec = np.array(self.llm_matrix.iloc[item_index, :])
		item_llm_vec = torch.tensor(item_llm_vec).float().cuda()
		# llm_output = self.l1(item_llm_vec).unsqueeze(0)
		llm_output = self.l1(item_llm_vec)
		# mean = 0  
		# stddev = 0.1  
		# noise = torch.tensor(np.random.normal(mean, stddev, size=item_seq.shape)).cuda()
		# item_seq = (item_seq + noise).long()
		item_emb = self.item_embedding(item_seq)
		
		input_emb = self.alpha*item_emb + self.beta*llm_output


		input_emb = item_emb
		input_emb = self.dropout(input_emb)
		input_emb = self.LayerNorm(input_emb)

		for i in range(self.num_layers):
			# item_emb = self.mamba_layers[i](input_emb, output_all_encoded_layers=False, item_seq_len=item_seq_len)
			item_emb = self.mamba_layers[i](input_emb)
			# import pdb
			# pdb.set_trace()
			# item_emb = self.mamba_layers[i](item_emb)
			# print("check item:", len(item_emb), item_emb.size(),item_emb[0][:10])
			# println()
		
		seq_output = self.gather_indexes(item_emb, item_seq_len - 1)  #torch.Size([1, 64])

		# print("check mamba output:", seq_output.size())
		# println(0)
		return seq_output

	def calculate_loss(self, interaction):
		item_seq = interaction[self.ITEM_SEQ]
		item_seq_len = interaction[self.ITEM_SEQ_LEN]
		seq_output = self.forward(item_seq, item_seq_len)
		
		pos_items = interaction[self.POS_ITEM_ID]
		# pos_items = interaction["item_id"]
		# indices = torch.LongTensor(np.random.choice(pos_items.size()[0], 10, replace=False)).cuda()
		# neg_items = torch.index_select(pos_items, 0, indices).cuda()
		# neg_items = pos_items.clone()
		# neg_items[-10:] = pos_items[:10]
		
		if self.loss_type == "BPR":
			# neg_items = interaction[self.NEG_ITEM_ID]
			pos_items_emb = self.item_embedding(pos_items)
			neg_items_emb = self.item_embedding(neg_items)
			pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
			neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
			# cos_s = self.cos(pos_items_emb,neg_items_emb)
			# print("cos_s:", cos_s)
			# println()
			loss = self.loss_fct(pos_score, neg_score)
			return loss
		else:  # self.loss_type = 'CE'
			test_item_emb = self.item_embedding.weight
			logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
			loss = self.loss_fct(logits, pos_items)
			return loss

	def predict(self, interaction):
		item_seq = interaction[self.ITEM_SEQ]
		item_seq_len = interaction[self.ITEM_SEQ_LEN]
		test_item = interaction[self.ITEM_ID]
		seq_output = self.forward(item_seq, item_seq_len)
		test_item_emb = self.item_embedding(test_item)
		scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
		return scores

	def full_sort_predict(self, interaction):
		item_seq = interaction[self.ITEM_SEQ]
		item_seq_len = interaction[self.ITEM_SEQ_LEN]
		seq_output = self.forward(item_seq, item_seq_len)
		test_items_emb = self.item_embedding.weight
		scores = torch.matmul(
			seq_output, test_items_emb.transpose(0, 1)
		)  # [B, n_items]
		return scores
	
class MambaLayer(nn.Module):
	def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
		super().__init__()
		self.num_layers = num_layers
		self.mamba = Mamba(
				# This module uses roughly 3 * expand * d_model^2 parameters
				d_model=d_model,
				d_state=d_state,
				d_conv=d_conv,
				expand=expand,
			)
		self.dropout = nn.Dropout(dropout)
		self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
		self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
	
	def forward(self, input_tensor):
		hidden_states = self.mamba(input_tensor)
		if self.num_layers == 1:        # one Mamba layer without residual connection
			hidden_states = self.LayerNorm(self.dropout(hidden_states))
		else:                           # stacked Mamba layers with residual connections
			hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
		hidden_states = self.ffn(hidden_states)
		return hidden_states

class FeedForward(nn.Module):
	def __init__(self, d_model, inner_size, dropout=0.2):
		super().__init__()
		self.w_1 = nn.Linear(d_model, inner_size)
		self.w_2 = nn.Linear(inner_size, d_model)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

	def forward(self, input_tensor):
		hidden_states = self.w_1(input_tensor)
		hidden_states = self.activation(hidden_states)
		hidden_states = self.dropout(hidden_states)

		hidden_states = self.w_2(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)

		return hidden_states



class BiMamba_Layer(nn.Module):
	def __init__(self, config) -> None:
		super().__init__()
		self.config = config
		mamba = {'mamba2':Mamba, # Mamba2
				 'mamba1':Mamba,
				 # 'mamba3':Mamba,
				 }
		model1 = mamba['mamba1']
		model2 = mamba['mamba2']
		# model3 = mamba['mamba3']
		self.token_forward = model1(
			d_model = self.config['hidden_size'],
			d_state = 32,
			d_conv = 4,
			expand = 2
		)

		self.token_backward = model2(
			d_model = self.config['hidden_size'],
			d_state = 32,
			d_conv = 4,
			expand = 2
		)
		# self.combine = model3(
		# 	d_model = self.config['hidden_size'],
		# 	d_state = 32,
		# 	d_conv = 4,
		# 	expand = 2
		# )
		self.dropout1 = self.config["dropout_prob"]
		self.d_model = self.config["hidden_size"]
		self.activation = nn.GELU()
		# self.project = nn.Linear(config.hidden_size*2,config.hidden_size)
		self.project = nn.Linear(2,1)
		self.dropout = nn.Dropout(self.dropout1)
		self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-12)
		self.ffn = FeedForward(d_model=self.d_model, inner_size=self.d_model*4, dropout=self.dropout1)
		self.num_layers = self.config["num_layers"]

	# def flip(self,hidden_states, lengths):

	#     lengths = lengths[0]
	#     batch_data = []
	#     for i in range(hidden_states.shape[0]):
	#         # import pdb
	#         # pdb.set_trace()
	#         data = hidden_states[i][:lengths[i]].flip(dims=[0])
	#         padding = hidden_states[i][lengths[i]:]
	#         batch_data.append(torch.cat([data,padding],dim=0).unsqueeze(0))
	#     hidden_states = torch.cat(batch_data,dim=0)
	#     return hidden_states
	def flip(self,hidden_states, lengths):
		return hidden_states.flip(dims=[1])

	
	def forward(self,
				hidden_states: torch.Tensor,
				**kwargs):
		lengths = [kwargs['lengths'].data]
		# lengths = kwargs['lengths']
		bi_layer_name = ['mamba1_bi','mamba2_bi']
		hidden_forward = self.token_forward(hidden_states)
		# hidden_forward = self.activation(hidden_forward)
		if self.config['layers_name'] in bi_layer_name:
			# hidden_backward = self.token_backward(torch.flip(hidden_states,[1]))
			# hidden_backward = torch.flip(hidden_backward,[1])
			hidden_backward = self.flip(hidden_states,lengths)
			hidden_backward = self.token_backward(hidden_backward)
			# hidden_backward = self.activation(hidden_backward)
			hidden_backward = hidden_backward.flip(dims=[1])
			# output = torch.cat([hidden_forward,hidden_backward],dim=-1)
			# output = hidden_forward + hidden_backward
								   # stacked Mamba layers with residual connections
			
			# print("hidden_states:", hidden_states.size())
			# output = self.project(hidden_states)
		else:
			output = (hidden_forward + hidden_backward)/2
			
		# output = (hidden_forward + hidden_backward)/2  # stacked Mamba layers with residual connections
		# output = hidden_forward

		output = torch.cat([hidden_forward.unsqueeze(-1),hidden_backward.unsqueeze(-1)],dim=-1)
		output = self.project(output).squeeze(-1)
		
		# output = self.combine(output)
		output1 = self.LayerNorm(self.dropout(output)+ hidden_states)
		output2 = self.ffn(output1)
		# output1 = self.LayerNorm(self.dropout(hidden_forward) + hidden_states)
		# output2 = self.ffn(output1)

		# output3 = self.LayerNorm(self.dropout(hidden_backward) + hidden_states)
		# output4 = self.ffn(output3)
		# output = (output2 + output4)/2
		
		# return (output,)
		return output2


class Bimamba(nn.Module):
	def __init__(self,config):
		super().__init__()
		config['layers_name'] = 'mamba1_bi'
		self.layer = nn.ModuleList([BiMamba_Layer(
				config = config,
			) for _ in range(config['num_layers'])])
	def get_length(self,x):
		return (x==0.).sum(dim=-1)

	
	def forward(self,x,**kwargs):
		item_seq_len = kwargs["item_seq_len"]
		# print("mask:", mask, mask.size())
		# lengths = self.get_length(mask).squeeze()
		# print("first check lengths:", lengths)
		lengths = item_seq_len
		# print("first check lengths:", lengths)
		# pritnln()
		# import pdb
		# pdb.set_trace()
		hidden_states = x
		if kwargs['output_all_encoded_layers']:
			all_hidden_states = ()
			all_hidden_states = all_hidden_states + (x,)
		for i, layer_module in enumerate(self.layer):
			
			layer_outputs = layer_module(hidden_states,lengths=lengths)
			hidden_states = layer_outputs
			if kwargs['output_all_encoded_layers']:
				all_hidden_states = all_hidden_states + (hidden_states,)
		if kwargs['output_all_encoded_layers']:
			return all_hidden_states
		else:
			return hidden_states