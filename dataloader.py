import torch
import random
from torchvision import transforms as trans
from PIL import Image


class Dataloader(object):
	def __init__(self,list_path,data_path,dataset='cifar',cache_flag=True):
		self.path=data_path
		self.cache_flag = cache_flag

		if dataset=='mir':
			fp=open(list_path + "train_img.txt","r")
			path_img = fp.readlines()
			for i in range(len(path_img)):
				path_img[i] = path_img[i].replace("\r\n","")
			self.img = path_img
						
			self.len = len(path_img)
			print ("dataset : mir\ntrain_size : " + str(self.len))
			self.list = list(range(self.len))
			
			random.shuffle(self.list)
			
			
			self.lab =          torch.load(data_path + "train_lab.t7")
			self.cache_img_fc7= torch.load(data_path + "train_fc7.t7")
			self.cache_img_fc6= torch.load(data_path + "train_fc6.t7")
			self.cache_img_p5=  torch.load(data_path + "train_p5.t7")
			self.cache_txt=     torch.load(data_path + "train_txt.t7")
					
			self.idx = 0
			
		
		self.tri_idx = 0
		
	def get_image_path(self,image_path):
#		print('get img',image_path)
		ori_img=Image.open(image_path)
		img=self.trans4(ori_img)
		if len(img)==1:
			ori_img = ori_img.convert("RGB")
			img=self.trans4(ori_img)
#			print len(img)
		return img
		
	def get_pos_neg(self,ori):
		listx = list(range(self.len))
		random.shuffle(listx)
	#	print self.len

		pos = listx[0]
		tx = 1
		
		while (torch.sum(self.lab[pos]*self.lab[ori])==0) or (pos==ori):
			pos = listx[tx]
			tx = (tx + 1)%self.len
		
		neg = listx[tx]
		
		while torch.sum(self.lab[neg]*self.lab[ori])>0:
			neg = listx[tx]
			tx = (tx + 1)%self.len
			
		return pos,neg
		

	def get_batch_wiki(self,batch_size=5):
		ori_img_fc7=torch.zeros(batch_size,4096)		
		pos_img_fc7=torch.zeros(batch_size,4096)		
		neg_img_fc7=torch.zeros(batch_size,4096)

		ori_img_fc6=torch.zeros(batch_size,4096)		
		pos_img_fc6=torch.zeros(batch_size,4096)		
		neg_img_fc6=torch.zeros(batch_size,4096)

		ori_img_p5 =torch.zeros(batch_size,25088)		
		pos_img_p5 =torch.zeros(batch_size,25088)		
		neg_img_p5 =torch.zeros(batch_size,25088)
		
		ori_txt=torch.zeros(batch_size,1000)
		pos_txt=torch.zeros(batch_size,1000)
		neg_txt=torch.zeros(batch_size,1000)
		
		
		lab=torch.zeros(batch_size,24)
		
		for ix in range(0,batch_size):
			ori_idx = self.list[self.idx]
			pos_idx,neg_idx = self.get_pos_neg(ori_idx)
			
			
			ori_img_fc7[ix]=self.cache_img_fc7[ori_idx]			
			pos_img_fc7[ix]=self.cache_img_fc7[pos_idx]
			neg_img_fc7[ix]=self.cache_img_fc7[neg_idx]
			                              
			ori_img_fc6[ix]=self.cache_img_fc6[ori_idx]			
			pos_img_fc6[ix]=self.cache_img_fc6[pos_idx]
			neg_img_fc6[ix]=self.cache_img_fc6[neg_idx]
			                              
			ori_img_p5[ix] = self.cache_img_p5[ori_idx]			
			pos_img_p5[ix] = self.cache_img_p5[pos_idx]
			neg_img_p5[ix] = self.cache_img_p5[neg_idx]

			ori_txt[ix]=self.cache_txt[ori_idx]
			pos_txt[ix]=self.cache_txt[pos_idx]
			neg_txt[ix]=self.cache_txt[neg_idx]
			
			lab[ix]=self.lab[ori_idx]
			self.idx += 1
			if self.idx==self.len:
				self.idx = 0
				random.shuffle(self.list)
				
		return ori_img_fc7,pos_img_fc7,neg_img_fc7,ori_img_fc6,pos_img_fc6,neg_img_fc6,ori_img_p5 ,pos_img_p5 ,neg_img_p5,ori_txt,pos_txt,neg_txt,lab
#		return ori_img_fc7,pos_img_fc7,neg_img_fc7,ori_txt,pos_txt,neg_txt,lab
			   
			   
		
	def get_train(self,batch_size=16):
		pic=torch.zeros(batch_size,3,224,224)
		label=torch.zeros(batch_size,1)
		for ix in range(0,batch_size):
			pic[ix]=self.get_image5(self.classidx,self.ix)
			label[ix]=self.classidx
			self.ix+=1
			if self.ix==self.list[self.classidx]:
				self.ix=self.start
				self.classidx+=1
			if self.classidx==self.classed:
				self.classidx=0
		return pic,label		
		
		
	def calc_m_s(self,batch_size=5000):
		mean = torch.zeros(3)
		std = torch.zeros(3)
		for ix in range(0,batch_size):
			x = self.get_image2(self.classidx,self.ix)
			for i in range(3):
				mean[i] += x[i].mean()
				std[i] += x[i].std()
			self.ix+=1
			if self.ix==self.list[self.classidx]:
				self.ix=self.start
				self.classidx+=1
			if self.classidx==self.classed:
				self.classidx=self.classst
		mean /= batch_size
		std /= batch_size
		return mean,std
		
	def get_valid(self,classidx,batch_size,bias):
		pic=torch.zeros(batch_size,3,224,224)
		for ix in range(bias,bias+batch_size):
			pic[ix-bias]=self.get_image3(classidx,ix)
		return pic
			