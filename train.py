#coding:utf-8
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import model
from torch.autograd import Variable
from torchvision import datasets, transforms
from dataloader import Dataloader
from sys import argv
import function

scrp,gpu,bit,batch,dataset = argv
g_z = 12
steps = 20000
rate = 0.1
observe = True

##setting
checkpoint_path = 'checkpoint-%s-%sbit' % (dataset,bit)
logpath = 'log-%s-%sbit.txt' % (dataset,bit)
if not os.path.exists(checkpoint_path):
	os.mkdir(checkpoint_path)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
batch_size = int(batch)
bit_len = int(bit)

change_feature_bit = [20,28]
gap = 16


####dataset
flag = True

data_path = "/home/yezhaoda/tomm/dataset/mir/feature/"
list_path = "/home/yezhaoda/tomm/dataset/mir/list/"


if dataset=='mir':
	traintest=Dataloader(list_path, data_path,"mir",True)
	flag = False

	
###model
model_img = model.imgNet(bit_len,batch_size,25088,4096,4096,change_feature_bit,gap)
model_img.cuda()
model_txt = model.single(bit_len,batch_size,1000)  ## MIR dataset do not have sentence feature, else use txtNet
model_txt.cuda()
model_atten = model.atten()
model_atten.cuda()
md_loss = torch.nn.CosineEmbeddingLoss(reduce=False)
print('model over')

###train
tag_img = True
tag_txt = True

episode_length = 0

target = torch.Tensor(batch_size)
for i in range(len(target)):
	target[i]= 1
target = Variable(target).cuda()
	
while True:
	episode_length += 1
	
	
	if episode_length%steps==0:
		if tag_img:
			model_img.low_lr(rate)
		if tag_txt:
			model_txt.low_lr(rate)
		model_atten.low_lr(rate)
		
	
	if episode_length%1000==0:
		if tag_txt:
			path=checkpoint_path+'/'+str(episode_length)+'_txt.model';
			torch.save(model_txt.state_dict(),path)
		if tag_img:
			path=checkpoint_path+'/'+str(episode_length)+'_img.model';
			torch.save(model_img.state_dict(),path)		
			
	if tag_txt:
		model_txt.train()	
	if tag_img:
		model_img.train()	
	

	if dataset=='mir':
		ori_img_fc7,pos_img_fc7,neg_img_fc7,ori_img_fc6,pos_img_fc6,neg_img_fc6,ori_img_p5 ,pos_img_p5,neg_img_p5,ori_txt,pos_txt,neg_txt,lab=traintest.get_batch_wiki(batch_size)
		
	if tag_img:
		ori_img_fc7=Variable(ori_img_fc7).cuda()
		pos_img_fc7=Variable(pos_img_fc7).cuda()
		neg_img_fc7=Variable(neg_img_fc7).cuda()
		
		ori_img_fc6=Variable(ori_img_fc6).cuda()
		pos_img_fc6=Variable(pos_img_fc6).cuda()
		neg_img_fc6=Variable(neg_img_fc6).cuda()
		
		ori_img_p5=Variable(ori_img_p5).cuda()
		pos_img_p5=Variable(pos_img_p5).cuda()
		neg_img_p5=Variable(neg_img_p5).cuda()
		
		hash_o_img = Variable(torch.zeros(batch_size,1).cuda())
		hash_p_img = Variable(torch.zeros(batch_size,1).cuda())
		hash_n_img = Variable(torch.zeros(batch_size,1).cuda())
		
		probs_o_img, hx_o_img = model_img(ori_img_p5,ori_img_fc6,ori_img_fc7)
		probs_p_img, hx_p_img = model_img(pos_img_p5,pos_img_fc6,pos_img_fc7)
		probs_n_img, hx_n_img = model_img(neg_img_p5,neg_img_fc6,neg_img_fc7)
				
	if tag_txt:
		ori_txt=Variable(ori_txt).cuda()
		pos_txt=Variable(pos_txt).cuda()
		neg_txt=Variable(neg_txt).cuda()
		hash_o_txt = Variable(torch.zeros(batch_size,1).cuda())
		hash_p_txt = Variable(torch.zeros(batch_size,1).cuda())
		hash_n_txt = Variable(torch.zeros(batch_size,1).cuda())
		probs_o_txt, hx_o_txt = model_txt(ori_txt)
		probs_p_txt, hx_p_txt = model_txt(pos_txt)
		probs_n_txt, hx_n_txt = model_txt(neg_txt)
	

	for i in range(bit_len):
		if tag_img:
			hash_o_img = torch.cat([hash_o_img,probs_o_img[i]],1)
			hash_p_img = torch.cat([hash_p_img,probs_p_img[i]],1)
			hash_n_img = torch.cat([hash_n_img,probs_n_img[i]],1)
		if tag_txt:
			hash_o_txt = torch.cat([hash_o_txt,probs_o_txt[i]],1)
			hash_p_txt = torch.cat([hash_p_txt,probs_p_txt[i]],1)
			hash_n_txt = torch.cat([hash_n_txt,probs_n_txt[i]],1)	
	
	model_atten.zero_grad()
####
	output_loss_img = 0
	if tag_img:
		hash_o_img = hash_o_img[:,1:]  ### the first bit is empty
		hash_p_img = hash_p_img[:,1:]
		hash_n_img = hash_n_img[:,1:]
		tri_loss_img = torch.mean(function.triplet_margin_loss(hash_o_img,hash_p_img,hash_n_img))
		model_img.zero_grad()
		if not tag_txt:
			tri_loss_img.backward()
		output_loss_img = tri_loss_img.item()
		
	output_loss_txt = 0	
	if tag_txt:
		hash_o_txt = hash_o_txt[:,1:]
		hash_p_txt = hash_p_txt[:,1:]
		hash_n_txt = hash_n_txt[:,1:]
		tri_loss_txt = torch.mean(function.triplet_margin_loss(hash_o_txt,hash_p_txt,hash_n_txt))
		model_txt.zero_grad()
		if not tag_img:
			tri_loss_txt.backward()
		output_loss_txt = tri_loss_txt.item()
		
	output_cross_loss_txt = 0
	output_cross_loss_img = 0
	if tag_img and tag_txt:
		
		cross_pair_txt = torch.trace( md_loss(hash_o_txt,hash_o_img,target) * model_atten(hx_o_txt,hx_o_img) )
##		cross_pair_img = torch.trace( md_loss(hash_o_img,hash_o_txt,target) * model_atten(hx_o_img,hx_o_txt) )  #if model more than 2
		
		cross_pair = cross_pair_txt #+ cross_pair_img
		
		cross_loss_txt = torch.mean(function.triplet_margin_loss(hash_o_txt,hash_p_img,hash_n_img))
		cross_loss_img = torch.mean(function.triplet_margin_loss(hash_o_img,hash_p_txt,hash_n_txt))
		
		final_loss_txt =  cross_loss_txt + tri_loss_txt
		final_loss_img =  cross_loss_img + tri_loss_img
		
		final_loss = final_loss_txt + final_loss_img + cross_pair
		final_loss.backward()
		
#### 更新参数

	if tag_img:
		model_img.step()
		
	if tag_txt:
		model_txt.step()
		
	model_atten.step()
	
	if episode_length%100==0:	
		print(str(episode_length)+' '+str(output_loss_img) + ' ' + str(output_loss_txt)  + "\n")
		file=open(logpath,"a")
		file.write(str(episode_length)+' '+str(output_loss_img) + ' ' + str(output_loss_txt) + "\n")
		file.close()
	


        
        