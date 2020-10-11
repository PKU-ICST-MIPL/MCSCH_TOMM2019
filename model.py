import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as md
	
def make_fc_layer(input,output):
	fc = nn.Linear(input, output)
	fc.weight.data.normal_(0, 0.01)
	fc.bias.data.zero_()
	return fc

def _initialize_weights(model,rag=0.01):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()	
	
class imgNet(torch.nn.Module):

	def __init__(self,bit_len,batch_size,fea_len_low,fea_len_mid,fea_len_hig,cgc,gap=16):
		super(imgNet, self).__init__()
		
		self.gap = gap
		self.lstm_hidden = 1024
		self.hidden = 1024
		self.bit_len = bit_len
		self.batch_size = batch_size
		self.cgc = cgc
		self.lstm = nn.RNNCell(self.hidden, self.lstm_hidden)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		
		self.actor_linear = make_fc_layer(self.lstm_hidden, 1)		
		
		self.low_fc = nn.Sequential( nn.Linear(fea_len_low,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(self.hidden,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		_initialize_weights(self.low_fc)

		self.mid_fc = nn.Sequential( nn.Linear(fea_len_mid,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(self.hidden,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		_initialize_weights(self.mid_fc)

		self.hig_fc = nn.Sequential( nn.Linear(fea_len_hig,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(self.hidden,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		_initialize_weights(self.hig_fc)
	
		
#		self.opt1 = optim.SGD(self.model.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt2 = optim.SGD(self.lstm.parameters(), lr=0.001,momentum=0.9, weight_decay=0.005)
		self.opt3 = optim.SGD(self.actor_linear.parameters(), lr=0.001,momentum=0.9, weight_decay=0.005)
		self.opt4 = optim.SGD(self.low_fc.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt5 = optim.SGD(self.mid_fc.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt6 = optim.SGD(self.hig_fc.parameters(), lr=0.001,momentum=0.9, weight_decay=0.005)

	def forward(self,x_low, x_mid, x_hig):		
	
		x_low = self.low_fc(x_low)
		x_mid = self.mid_fc(x_mid)
		x_hig = self.hig_fc(x_hig)
		

		
		hx = x_hig #Variable(torch.zeros(self.batch_size, self.lstm_hidden).cuda())
#		hx = x_low
		ipt = x_hig
		probs = []
		tip = 1
		for step in range(self.bit_len):
			if step==self.cgc[0]:
				ipt = x_mid
				hx = self.lstm(ipt,hx)
				tip = 1
			elif step==self.cgc[1]:
				ipt = x_low
				hx = self.lstm(ipt,hx)
				tip = 1
			elif tip%self.gap==0:
				hx = self.lstm(ipt,hx)
				tip = 1
			else:
				hx = self.lstm(hx,hx)
				tip = tip + 1
#			hx = self.lstm(hx,hx)			
			prob = torch.sigmoid(self.actor_linear(hx))
			probs.append(prob)
			
        
		return probs,hx
	
	def zero_grad(self):
#		self.opt1.zero_grad()
		self.opt2.zero_grad()
		self.opt3.zero_grad()
		self.opt4.zero_grad()
		self.opt5.zero_grad()
		self.opt6.zero_grad()
						
	def step(self):
#		self.opt1.step()
		self.opt2.step()
		self.opt3.step()
		self.opt4.step()
		self.opt5.step()
		self.opt6.step()


	def low_lr(self,rate):
#		for pg in self.opt1.param_groups:
#			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt2.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt3.param_groups:
			pg['lr'] = pg['lr'] * rate
					
		for pg in self.opt4.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt5.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt6.param_groups:
			pg['lr'] = pg['lr'] * rate


class txtNet(torch.nn.Module):

	def __init__(self,bit_len,batch_size,fea_len_low,fea_len_mid,cgc,gap=16):
		super(imgNet, self).__init__()
		
		self.gap = gap
		self.lstm_hidden = 1024
		self.hidden = 1024
		self.bit_len = bit_len
		self.batch_size = batch_size
		self.cgc = cgc
		self.lstm = nn.RNNCell(self.hidden, self.lstm_hidden)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		
		self.actor_linear = make_fc_layer(self.lstm_hidden, 1)		
		
		self.low_fc = nn.Sequential( nn.Linear(fea_len_low,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(self.hidden,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		_initialize_weights(self.low_fc)

		self.mid_fc = nn.Sequential( nn.Linear(fea_len_mid,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(self.hidden,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		_initialize_weights(self.mid_fc)

		self.opt2 = optim.SGD(self.lstm.parameters(), lr=0.001,momentum=0.9, weight_decay=0.005)
		self.opt3 = optim.SGD(self.actor_linear.parameters(), lr=0.001,momentum=0.9, weight_decay=0.005)
		self.opt4 = optim.SGD(self.low_fc.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt5 = optim.SGD(self.mid_fc.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)

	def forward(self, x_wd, x_sen):		
	
		x_wd = self.low_fc(x_wd)
		x_sen = self.mid_fc(x_sen)

		hx = x_wd #Variable(torch.zeros(self.batch_size, self.lstm_hidden).cuda())

		ipt = x_wd
		probs = []
		tip = 1
		for step in range(self.bit_len):
			if step==self.cgc[0]:
				ipt = x_sen
				hx = self.lstm(ipt,hx)
				tip = 1
			elif step==self.cgc[1]:
				ipt = x_wd
				hx = self.lstm(ipt,hx)
				tip = 1
			elif tip%self.gap==0:
				hx = self.lstm(ipt,hx)
				tip = 1
			else:
				hx = self.lstm(hx,hx)
				tip = tip + 1
		
			prob = torch.sigmoid(self.actor_linear(hx))
			probs.append(prob)

		return probs,hx
	
	def zero_grad(self):

		self.opt2.zero_grad()
		self.opt3.zero_grad()
		self.opt4.zero_grad()
		self.opt5.zero_grad()
		self.opt6.zero_grad()
						
	def step(self):

		self.opt2.step()
		self.opt3.step()
		self.opt4.step()
		self.opt5.step()
		self.opt6.step()

	def low_lr(self,rate):
		
		for pg in self.opt2.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt3.param_groups:
			pg['lr'] = pg['lr'] * rate
					
		for pg in self.opt4.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt5.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt6.param_groups:
			pg['lr'] = pg['lr'] * rate

	
			
class single(torch.nn.Module):

	def __init__(self,bit_len,batch_size,fea_len,gap=16):
		super(single, self).__init__()
		self.gap = gap
		self.lstm_hidden = 1024
		self.hidden = 1024
		self.bit_len = bit_len
		self.batch_size = batch_size
		
		self.lstm = nn.RNNCell(self.hidden, self.lstm_hidden)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		
		self.actor_linear = make_fc_layer(self.lstm_hidden, 1)		
		self.low_fc = nn.Sequential( nn.Linear(fea_len,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(self.hidden,self.hidden),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		_initialize_weights(self.low_fc,1)

		self.opt2 = optim.SGD(self.lstm.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt3 = optim.SGD(self.actor_linear.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt4 = optim.SGD(self.low_fc.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)


	def forward(self, inputs):
	
	
		x_low = self.low_fc(inputs)

		
		hx = x_low #Variable(torch.zeros(self.batch_size, self.lstm_hidden).cuda())

		probs = []
		for step in range(self.bit_len):
			if step%self.gap==0:
				hx = self.lstm(x_low,hx)
			else:
				hx = self.lstm(hx,hx)
				
			prob = torch.sigmoid(self.actor_linear(hx))
			probs.append(prob)
			
        
		return probs,hx
	
	def zero_grad(self):

		self.opt2.zero_grad()
		self.opt3.zero_grad()
		self.opt4.zero_grad()
				
	def step(self):

		self.opt2.step()
		self.opt3.step()
		self.opt4.step()

	def low_lr(self,rate):
			
		for pg in self.opt2.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt3.param_groups:
			pg['lr'] = pg['lr'] * rate
					
		for pg in self.opt4.param_groups:
			pg['lr'] = pg['lr'] * rate
			

class atten(torch.nn.Module):

	def __init__(self):
		super(atten, self).__init__()
		
		self.lstm_hidden = 2048
		self.atten = make_fc_layer(self.lstm_hidden, 1)
		self.opt1 = optim.SGD(self.atten.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)

	def forward(self, v1, v2):
		vx = torch.cat([v1,v2],1)
		probs = torch.sigmoid( self.atten(vx) )
		return probs
	
	def zero_grad(self):
		self.opt1.zero_grad()
			
	def step(self):
		self.opt1.step()

	def low_lr(self,rate):
		for pg in self.opt1.param_groups:
			pg['lr'] = pg['lr'] * rate
