from torch.nn import Sequential
from torch.nn import Module         # Rather than using the Sequential PyTorch class to implement LeNet, we’ll instead subclass the Module object so you can see how PyTorch implements neural networks using classes
from torch.nn import Conv2d         # PyTorch’s implementation of convolutional layers
from torch.nn import MultiheadAttention        # PyTorch’s implementation of convolutional layers
from torch.nn import Linear         # Fully connected layers
from torch.nn import LayerNorm
from torch.nn import Dropout
from torch.nn import MaxPool2d      # Applies 2D max-pooling to reduce the spatial dimensions of the input volume
from torch.nn import ReLU, LeakyReLU, Sigmoid           # ReLU activation function
from torch.nn import GELU
from torch.nn import Tanh
from torch.nn.modules.activation import Tanh
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.flatten import Flatten


from Utils import imageToPatches
import Config as cfg


class CAE_new(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(CAE_new, self).__init__()

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 4 * 1 * 1
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 256 * 8 * 8
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded

class CAE_AHZ(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(CAE_AHZ, self).__init__()

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded

class PSGN_Vanilla(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(PSGN_Vanilla, self).__init__()

		self.c1 = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)
		
		self.c2 = Sequential(
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c3 = Sequential(
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c4 = Sequential(
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c5 = Sequential(
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c6 = Sequential(
			Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c7 = Sequential(
			Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2), # 8 * 4 * 4
			ReLU(),	
		)

		self.fc1 = Sequential(
			Linear(512*2*2, 2048),
			ReLU(),
			Linear(2048, 1024*3),
			ReLU(),
			Linear(1024*3, 1024*3),
			ReLU(),
		)
		
		self.d1 = Sequential(
			ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2) # 8 * 5 * 5	
		)
		
		self.c8 = Sequential(
			Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d2 = Sequential(
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1) # 8 * 5 * 5	
		)

		self.c9 = Sequential(
			Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d3 = Sequential(
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1), # 8 * 5 * 5	
			ReLU()
		)

		self.c10 = Sequential(
			Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.c11 = Sequential(
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)


	def forward(self, x):

		x0 = self.c1(x)
		x1 = self.c2(x0)
		x2 = self.c3(x1)
		x3 = self.c4(x2)
		x4 = self.c5(x3)
		x5 = self.c6(x4)
		x = self.c7(x5)
		x = self.fc1(torch.flatten(x, 1))
		x = x.reshape(-1, 1024, 3)
		# print(x.shape)

		return x

class PSGN(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(PSGN, self).__init__()

		self.c1 = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)
		
		self.c2 = Sequential(
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c3 = Sequential(
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c4 = Sequential(
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c5 = Sequential(
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c6 = Sequential(
			Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
		)

		self.c7 = Sequential(
			Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2), # 8 * 4 * 4
			ReLU(),	
		)

		self.fc1 = Sequential(
			Linear(512*2*2, 2048),
			ReLU(),
			Linear(2048, 1024),
			ReLU(),
			Linear(1024, 63*3),
			ReLU()
		)
		
		self.d1 = Sequential(
			ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2) # 8 * 5 * 5	
		)
		
		self.c8 = Sequential(
			Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d2 = Sequential(
			Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1) # 8 * 5 * 5	
		)

		self.c9 = Sequential(
			Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.d3 = Sequential(
			Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 8 * 5 * 5	
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1), # 8 * 5 * 5	
			ReLU()
		)

		self.c10 = Sequential(
			Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)

		self.c11 = Sequential(
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1) # 8 * 4 * 4			
		)


	def forward(self, x):

		x0 = self.c1(x)
		x1 = self.c2(x0)
		x2 = self.c3(x1)
		x3 = self.c4(x2)
		x4 = self.c5(x3)
		x5 = self.c6(x4)
		x = self.c7(x5)
		x_additional = self.fc1(torch.flatten(x, 1))
		x_additional = x_additional.reshape(-1, 63, 3)
		x = self.d1(x)
		x5 = self.c8(x5)
		x = F.relu(torch.add(x, x5))
		x = self.d2(x)
		x4 = self.c9(x4)
		x = F.relu(torch.add(x, x4))
		x = self.d3(x)
		x3 = self.c10(x3)
		x = F.relu(torch.add(x, x3))
		x = self.c11(x)
		x = x.reshape(-1, 31*31, 3)
		x = torch.cat([x_additional, x], dim=1)
		# print(x.shape)

		return x

class CAE_AHZ_Attention(Module):
	def __init__(self):
		super(CAE_AHZ_Attention, self).__init__()
		
		self.sequence_length = 256
		self.input_size = 256
		self.embed_size = 128
		self.positional_encoding = torch.nn.Parameter(torch.rand(self.embed_size*2, self.embed_size*2))

		self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) 
		self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

		self.attention1 = MultiheadAttention(128, 32)

		self.deconv1 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1)
		self.deconv2 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
		self.deconv3 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.deconv4 = ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
		
		self.linear1 = Linear(32*18*18, 2700)
		self.linear2 = Linear(2700, cfg.SAMPLE_SIZE*3)

		self.maxpool = MaxPool2d(kernel_size=2, stride=2)



	def forward(self, x):
		batch_size, channels, sequence_length, input_size = x.shape 
		# Positional encoding
		x = x.reshape(batch_size*channels, sequence_length, -1)
		for i in range(batch_size*channels):
			x[i] = torch.add(x[i], self.positional_encoding)
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(batch_size, channels, sequence_length, input_size)
		batch_size, channels, sequence_length, input_size = x.shape
		# Conv 1
		x = torch.relu(self.conv1(x))
		x = self.maxpool(x)
		batch_size, channels, sequence_length, input_size = x.shape
		# Attn 1
		x = x.reshape(batch_size*channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention1(x, x, x)
		x = torch.relu(attn_output)
		# Conv 2		
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(batch_size, channels, sequence_length, input_size)
		x = torch.relu(self.conv2(x))
		x = self.maxpool(x)
		# Conv3
		x = torch.relu(self.conv3(x))
		x = self.maxpool(x)
		# Conv4
		x = torch.relu(self.conv4(x))
		x = self.maxpool(x)
		
		# Deconv 1-2
		x = torch.relu(self.deconv2(x))
		x = torch.relu(self.deconv3(x))
		# Linear 1-2
		batch_size, channels, height, width = x.shape
		x = x.reshape(-1, channels*height*width)
		x = torch.relu(self.linear1(x))
		x = self.linear2(x)
		x = torch.tanh(x)

		return x


class PureAttention(Module):
	
	def __init__(self):
		super(PureAttention, self).__init__()
		# TODO: Initialize myModel
		
		self.sequence_length = 256
		self.input_size = 256
		self.embed_size = 256		
		self.positional_encoding = torch.nn.Parameter(torch.rand(self.embed_size, self.embed_size))

		self.attention1 = MultiheadAttention(256, 32)
		self.attention2 = MultiheadAttention(128, 32)
		self.attention3 = MultiheadAttention(64, 16)
		self.attention4 = MultiheadAttention(32, 8)
		self.attention5 = MultiheadAttention(16, 4)
		
		self.linear1 = Linear(3*8*8, 3*12*12)
		self.linear2 = Linear(3*12*12, 768)
		self.linear3 = Linear(768, cfg.SAMPLE_SIZE*3)
		self.linear4 = Linear(cfg.SAMPLE_SIZE*3, cfg.SAMPLE_SIZE*3)

		self.deconv1 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1) # 8 * 5 * 5
		self.deconv2 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1) # 8 * 5 * 5
		self.deconv3 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1) # 4 * 15 * 15


		self.maxpool = MaxPool2d(kernel_size=2, stride=2)



	def forward(self, x):
		batch_size, channels, sequence_length, input_size = x.shape
        
		# Positional encoding
		x = x.reshape(batch_size*channels, sequence_length, -1)
		for i in range(batch_size*channels):
			x[i] = torch.add(x[i], self.positional_encoding)
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(batch_size, channels, sequence_length, input_size)

		# Attention layer 1
		batch_size, channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size*channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention1(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)

		# Attention layer 2
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention2(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)

		# Attention layer 3
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention3(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)

		# Attention layer 4
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention4(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)


		# Attention layer 5
		batch_size_channels, sequence_length, input_size = x.shape
		x = x.reshape(batch_size_channels, sequence_length, input_size)		
		attn_output, attn_output_weights = self.attention5(x, x, x)
		x = torch.relu(attn_output)
		x = self.maxpool(x)



		# Deconv 1-2
		batch_size_channels, sequence_length, input_size = x.shape
		x = torch.unsqueeze(x, dim=0)
		x = x.reshape(int(batch_size_channels/3), 3, sequence_length, input_size)
		# x = self.deconv1(x)
		# x = self.deconv2(x)
		# x = self.deconv3(x)


		# Linear 1-2
		# batch_size_channels, sequence_length, input_size = x.shape
		# x = torch.unsqueeze(x, dim=0)
		# x = x.reshape(int(batch_size_channels/3), 3, sequence_length, input_size)
		batch_size, channels, height, width = x.shape
		x = x.reshape(-1, channels*height*width)
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		x = torch.relu(self.linear3(x))
		x = self.linear4(x)
		x = torch.tanh(x)

		return x


class PreLayerNormAttention(Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = Sequential(
            Linear(embed_dim, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, embed_dim),
            Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_points, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_points - Number of points in the output point cloud
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Layers/Networks
        self.input_layer = Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.layerNorm = LayerNorm(embed_dim)
        
		# Reconstruction head (FC)
        self.fc1 = Linear(num_patches*embed_dim, num_patches*embed_dim*2)
        self.fc2 = Linear(num_patches*embed_dim*2, 2700)
        self.fc3 = Linear(2700, num_points)

		# Reconstruction head (Deconv)
        self.deconv1 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1) # 8 * 5 * 5
        self.deconv2 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1) # 8 * 5 * 5
        self.deconv3 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # 4 * 15 * 15
        self.deconv4 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # 4 * 15 * 15			
        self.linear1 = Linear(16*12*12, 2700)
        self.linear2 = Linear(2700, cfg.SAMPLE_SIZE*3)

		
        self.dropout = Dropout(dropout)

        # Parameters/Embeddings
        self.pos_embedding = torch.nn.Parameter(torch.randn(1,num_patches,embed_dim))

    def forward(self, x):
		# Preprocess input -> Convert the input image to patches
        x = imageToPatches(x, self.patch_size, True)        
        B, T, _ = x.shape 

		# Linear projection of flattened patches       
        x = self.input_layer(x)
		
        # Add positional encoding
        x = x + self.pos_embedding

        # Apply Transforrmer
        x = self.dropout(x)  
        x = x.transpose(0, 1) # Shape: (patch_size, batch_size, embed_dim)
        x = self.transformer(x)
                

        # Reconstruction head (FC)
        out = self.layerNorm(x)
        out = out.transpose(0,1)
        out = out.reshape(-1, self.num_patches*self.embed_dim)
        

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))


        return out


class ConViT(Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_points, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_points - Number of points in the output point cloud
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Layers/Networks
        self.conv1 = Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(2, stride=2)

        self.input_layer = Linear((patch_size**2), embed_dim)
        self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.layerNorm = LayerNorm(embed_dim)
        
		# Reconstruction head (FC)
        self.fc1 = Linear(num_patches*embed_dim, num_patches*embed_dim*2)
        self.fc2 = Linear(num_patches*embed_dim*2, 2700)
        self.fc3 = Linear(2700, num_points)

		# Reconstruction head (Deconv)
        self.deconv1 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1) # 8 * 5 * 5
        self.deconv2 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1) # 8 * 5 * 5
        self.deconv3 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # 4 * 15 * 15
        self.deconv4 = ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1) # 4 * 15 * 15			
        self.linear1 = Linear(16*12*12, 2700)
        self.linear2 = Linear(2700, cfg.SAMPLE_SIZE*3)

		
        self.dropout = Dropout(dropout)

        # Parameters/Embeddings
        self.pos_embedding = torch.nn.Parameter(torch.randn(1,num_patches,embed_dim))

    def forward(self, x):
		# Preprocess input -> Convert the input image to patches
        x = imageToPatches(x, self.patch_size, False)
        x = x.reshape(-1, self.num_patches*3, self.patch_size, self.patch_size)

		# Apply convolution operation on image patches
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.conv2(x)
        # x = self.maxpool(x)
        x = x.reshape(-1, self.num_patches, self.patch_size**2)

        B, T, _ = x.shape

		# Linear projection of flattened patches       
        x = self.input_layer(x)
		
        # Add positional encoding
        # x = x + self.pos_embedding

        # Apply Transforrmer
        x = self.dropout(x)  
        x = x.transpose(0, 1) # Shape: (patch_size, batch_size, embed_dim)
        x = self.transformer(x)
        

        # Reconstruction head (FC)
        out = self.layerNorm(x)
        out = out.transpose(0,1)
        out = out.reshape(-1, self.num_patches*self.embed_dim)

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))


		# Reconstruction head (Deconv)
        # out = self.layerNorm(x)
        # out = out.transpose(0,1)
        # out = out.reshape(-1, self.num_patches, int(np.sqrt(self.embed_dim)), int(np.sqrt(self.embed_dim)))
        # out = torch.relu(self.deconv1(out))
        # out = torch.relu(self.deconv2(out))
        # out = torch.relu(self.deconv3(out))
        # out = torch.relu(self.deconv4(out))
        # out = out.reshape(-1, 16*12*12)
        # out = torch.relu(self.linear1(out))
        # out = torch.tanh(self.linear2(out))

        return out


class Converntional_Skip_Connection(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""
	def __init__(self):
		super(Converntional_Skip_Connection, self).__init__()
		self.channel = 128
		self.conv2d_x4 = Sequential(Conv2d(in_channels=128, out_channels=256, kernel_size=1))

		self.channel = 64
		self.conv2d_x3 = Sequential(Conv2d(in_channels=64, out_channels=128, kernel_size=1))

		self.channel = 32
		self.conv2d_x2 = Sequential(Conv2d(in_channels=32, out_channels=64, kernel_size=1))

		self.channel = 16
		self.conv2d_x1 = Sequential(Conv2d(in_channels=16, out_channels=32, kernel_size=1))
		
		self.conv_in_x6 = Sequential(Conv2d(in_channels=512, out_channels=128, kernel_size=1))
		self.conv_in_x7 = Sequential(Conv2d(in_channels=256, out_channels=64, kernel_size=1))
		self.conv_in_x8 = Sequential(Conv2d(in_channels=128, out_channels=32, kernel_size=1))
		self.conv_in_x9 = Sequential(Conv2d(in_channels=64, out_channels=16, kernel_size=1))
		
		self.conv1 = Sequential(Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
		 ReLU(),
		 MaxPool2d(2, stride=2), # 8 * 2 * 2
		)
		self.conv2 = Sequential( Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
		 ReLU(),
		 MaxPool2d(2, stride=2), # 8 * 2 * 2
		)
		
		
		self.conv3 = Sequential( Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
		ReLU(),
		 MaxPool2d(2, stride=2), # 8 * 2 * 2
		)
		self.conv4 = Sequential(Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
		 ReLU(),
		 MaxPool2d(2, stride=2), # 4 * 1 * 1
		)

		self.conv5 = Sequential(
				Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
				ReLU(),
				MaxPool2d(2, stride=2), # 256 * 8 * 8
		)

		self.deconv1 = Sequential(ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=1),)  # 8 * 5 * 5
		self.deconv2 = Sequential(ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=1),) # 8 * 5 * 5
		self.deconv3 = Sequential(ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),) # 4 * 15 * 15
		self.deconv4 = Sequential(ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),) # 4 * 15 * 15
		
		self.linear1 = Sequential(Linear(16*12*12, 2700))
		self.linear2 = Sequential(Linear(2700, cfg.SAMPLE_SIZE*3),)

		
		


	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x4 = self.conv4(x3)
		x5 = self.conv5(x4)
		self.channel= 128
		x4_conv = self.conv2d_x4(x4)
		# print('x5',x5.shape)
		# print('x4',x4_conv.shape)
		cat_1 = torch.cat((x5.permute(1,0,2,3),x4_conv[:,:,4:12,4:12].permute(1,0,2,3)),0).permute(1,0,2,3) #eq=512
		x6 = self.deconv1(cat_1)
		
		# x6 = self.conv_in_x6(cat_1) #eq=128
			
		# print('x6.shape',x6.shape)
		x3_conv = self.conv2d_x3(x3) #64 ->128
		
		# print(x3_conv.shape,'x3_conv')
		# print('cat_x7',torch.cat((x6.permute(1,0,2,3),x3_conv[:,:,3:13,3:13].permute(1,0,2,3)),0).permute(1,0,2,3).shape)
		x7 = self.deconv2(torch.cat((x6.permute(1,0,2,3),x3_conv[:,:,3:13,3:13].permute(1,0,2,3)),0).permute(1,0,2,3))

		# print(x7.shape,'x7.shape')
		x2_conv = self.conv2d_x2(x2) #32 -> 64
		x8 = self.deconv3(torch.cat((x7.permute(1,0,2,3),x2_conv[:,:,26:38,26:38].permute(1,0,2,3)),0).permute(1,0,2,3))
		# print(x8.shape,'x8.shape')
		x1_conv = self.conv2d_x1(x1)
		x9 = self.deconv4(torch.cat((x8.permute(1,0,2,3),x1_conv[:,:,58:70,58:70].permute(1,0,2,3)),0).permute(1,0,2,3))
		# print('x9.shape',x9.shape)
		x10 = self.linear1(torch.flatten(x9, 1))
		# print('x10.shape',x10.shape)
		x11 = self.linear2(x10)
		# print('x11.shape',x11.shape)
		x12 = torch.tanh(x11)
		return x12


class ViT_CNN(Module):
	def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_points, patch_size, num_patches, num_points_conv, dropout=0.0):
		super().__init__()
		self.patch_size = patch_size
		self.embed_dim = embed_dim
		self.num_patches = num_patches
		self.ConvEncoder = 	Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
		)
		self.ConvDecoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, num_points_conv*3),
			Linear(num_points_conv*3, num_points_conv*3),
			Tanh()
		)
		self.dropout = Dropout(dropout)
		# Parameters/Embeddings
		self.pos_embedding = torch.nn.Parameter(torch.randn(1,num_patches,embed_dim))
		# Layers/Networks
		self.input_layer = Linear(num_channels*(patch_size**2), embed_dim)
		self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
		self.layerNorm = LayerNorm(embed_dim)
		# Reconstruction head (FC)
		self.fc1 = Linear(num_patches*embed_dim, int(num_patches*embed_dim/2))
		self.fc2 = Linear(int(num_patches*embed_dim/2), cfg.SAMPLE_SIZE)
		self.fc3 = Linear(cfg.SAMPLE_SIZE, (cfg.SAMPLE_SIZE-num_points_conv)*3)

		
	def forward(self, x):
		###################### CNN stream ###############################
		convFeatures = self.ConvEncoder(x)
		convOutput = self.ConvDecoder(convFeatures)
		###################### Transformer stream #######################
		# Preprocess input -> Convert the input image to patches
		x = imageToPatches(x, self.patch_size, True)        
		B, T, _ = x.shape 
		# Linear projection of flattened patches       
		x = self.input_layer(x)	
		# Add positional encoding
		x = x + self.pos_embedding
		# Apply Transforrmer
		x = self.dropout(x)
		x = x.transpose(0, 1) # Shape: (patch_size, batch_size, embed_dim)
		transformerFeatures = self.transformer(x)
		# Reconstruction head (FC)
		out = self.layerNorm(x)
		out = out.transpose(0, 1)
		out = out.reshape(-1, self.num_patches*self.embed_dim)
		out = torch.relu(self.fc1(out))
		out = torch.relu(self.fc2(out))
		out = torch.tanh(self.fc3(out))
		transformerOutput = out
		###################### Merge two streams #######################
		out = torch.cat((transformerOutput, convOutput), 1)
		return out


class TCNN(Module):
	def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_points, patch_size, num_patches, num_points_conv, dropout=0.0):
		super().__init__()
		self.patch_size = patch_size
		self.num_patches = num_patches
		self.embed_dim = embed_dim
		self.num_patches = num_patches
		self.ConvEncoder = 	Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2),
		)
		self.ConvDecoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, num_points_conv*3),
			Linear(num_points_conv*3, num_points_conv*3),
			Tanh()
		)
		self.dropout = Dropout(dropout)
		# Parameters/Embeddings
		# self.pos_embedding = torch.nn.Parameter(torch.randn(1,num_patches,embed_dim))
		# Layers/Networks
		self.input_layer = Conv2d(num_channels, embed_dim, patch_size, patch_size)
		self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
		self.layerNorm = LayerNorm(embed_dim)
		# Reconstruction head (FC)
		self.fc1 = Linear(num_patches*embed_dim, int(num_patches*embed_dim/2))
		self.fc2 = Linear(int(num_patches*embed_dim/2), cfg.SAMPLE_SIZE)
		self.fc3 = Linear(cfg.SAMPLE_SIZE, (cfg.SAMPLE_SIZE-num_points_conv)*3)

		
	def forward(self, x):
		###################### CNN stream ###############################
		convFeatures = self.ConvEncoder(x)
		convOutput = self.ConvDecoder(convFeatures)
		###################### Transformer stream #######################
		# Linear projection of flattened patches
		x = self.input_layer(x)
		# Add positional encoding
		# x = x.permute(0, 2, 3, 1)
		x = x.view(-1, self.num_patches, self.embed_dim)
		# x = x + self.pos_embedding
		# Apply Transforrmer
		x = self.dropout(x)
		transformerFeatures = self.transformer(x)
		# Reconstruction head (FC)
		out = self.layerNorm(x)
		out = out.reshape(-1, self.num_patches*self.embed_dim)
		out = torch.relu(self.fc1(out))
		out = torch.relu(self.fc2(out))
		out = torch.tanh(self.fc3(out))
		transformerOutput = out
		###################### Merge two streams #######################
		out = torch.cat((transformerOutput, convOutput), 1)
		return out


class PureTransformer(Module):
	def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_points, patch_size, num_patches, num_points_conv, dropout=0.0):
		super().__init__()
		self.patch_size = patch_size
		self.num_patches = num_patches
		self.embed_dim = embed_dim
		self.num_patches = num_patches
		self.dropout = Dropout(dropout)
		# Parameters/Embeddings
		# self.pos_embedding = torch.nn.Parameter(torch.randn(1,num_patches,embed_dim))
		# Layers/Networks
		self.input_layer = Conv2d(num_channels, embed_dim, patch_size, patch_size)
		self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
		self.layerNorm = LayerNorm(embed_dim)
		# Reconstruction head (FC)
		self.fc1 = Linear(num_patches*embed_dim, int(num_patches*embed_dim/2))
		self.fc2 = Linear(int(num_patches*embed_dim/2), cfg.SAMPLE_SIZE)
		self.fc3 = Linear(cfg.SAMPLE_SIZE, (cfg.SAMPLE_SIZE-num_points_conv)*3)

		
	def forward(self, x):
		###################### Transformer stream #######################
		# Linear projection of flattened patches
		x = self.input_layer(x)
		# Add positional encoding		
		x = x.view(-1, self.num_patches, self.embed_dim)
		# x = x + self.pos_embedding
		# Apply Transforrmer
		x = self.dropout(x)
		transformerFeatures = self.transformer(x)
		# Reconstruction head (FC)
		out = self.layerNorm(x)
		out = out.reshape(-1, self.num_patches*self.embed_dim)
		out = torch.relu(self.fc1(out))
		out = torch.relu(self.fc2(out))
		out = torch.tanh(self.fc3(out))
		return out


class DL_course_leakyrel_sig(Module):
	def __init__(self):
		super(DL_course_leakyrel_sig, self).__init__()
		
		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 4 * 1 * 1
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 256 * 8 * 8
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			LeakyReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			LeakyReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			LeakyReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			LeakyReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Sigmoid()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded


class DL_course_leakyrel_tanh(Module):
	def __init__(self):
		super(DL_course_leakyrel_tanh, self).__init__()
		
		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 4 * 1 * 1
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			MaxPool2d(2, stride=2), # 256 * 8 * 8
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			LeakyReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			LeakyReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			LeakyReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			LeakyReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded


class DL_course_rel_sigmoid(Module):
	def __init__(self):
		super(DL_course_rel_sigmoid, self).__init__()
		
		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 4 * 1 * 1
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 256 * 8 * 8
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Sigmoid()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded


class DL_course_leakyrel_tanh_stride(Module):
	def __init__(self):
		super(DL_course_leakyrel_tanh_stride, self).__init__()
		
		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), # 8 * 4 * 4
			LeakyReLU(),
			
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			LeakyReLU(),
			
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			LeakyReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			LeakyReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			LeakyReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			LeakyReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded


class DL_course_rel_tanh(Module):
	def __init__(self):
		super(DL_course_rel_tanh, self).__init__()
		
		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 8 * 4 * 4
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 8 * 2 * 2
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 4 * 1 * 1
			Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 4 * 2 * 2
			ReLU(),
			MaxPool2d(2, stride=2), # 256 * 8 * 8
		)
		
		self.decoder = Sequential(
			ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 8 * 5 * 5
			ReLU(),
			ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), # 4 * 15 * 15
			ReLU(),
			Flatten(),
			Linear(16*12*12, 2700),
			Linear(2700, cfg.SAMPLE_SIZE*3),
			Tanh()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded


class PreTrainedTransformer(Module):
	def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_points, patch_size, num_patches, num_points_conv, dropout=0.0):
		super().__init__()
		self.patch_size = patch_size
		self.num_patches = num_patches
		self.embed_dim = embed_dim
		self.num_patches = num_patches
		self.dropout = Dropout(dropout)
		# Parameters/Embeddings
		# self.pos_embedding = torch.nn.Parameter(torch.randn(1,num_patches,embed_dim))
		# Layers/Networks
		self.input_layer = Conv2d(num_channels, embed_dim, patch_size, patch_size)
		self.transformer = ViT('B_16_imagenet1k', pretrained=True)
		# self.transformer = Sequential(*[PreLayerNormAttention(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
		self.layerNorm = LayerNorm(embed_dim)
		# Reconstruction head (FC)
		self.fc1 = Linear(num_patches*embed_dim, int(num_patches*embed_dim/2))
		self.fc2 = Linear(int(num_patches*embed_dim/2), cfg.SAMPLE_SIZE)
		self.fc3 = Linear(cfg.SAMPLE_SIZE, (cfg.SAMPLE_SIZE-num_points_conv)*3)

		
	def forward(self, x):
		###################### Transformer stream #######################
		# Linear projection of flattened patches
		# x = self.input_layer(x)
		# Add positional encoding		
		# x = x.view(-1, self.num_patches, self.embed_dim)
		# x = x + self.pos_embedding
		# Apply Transforrmer
		# x = self.dropout(x)
		# print(x.shape)
		# transformerFeatures = self.transformer(x)
		transformerFeatures = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
		inputs = transformerFeatures(x.cpu(), return_tensors="pt")
		print(inputs)
		# Reconstruction head (FC)
		# print(transformerFeatures.shape)
		out = transformerFeatures.reshape(-1, self.num_patches*self.embed_dim)
		out = torch.relu(self.fc1(out))
		out = torch.relu(self.fc2(out))
		out = torch.tanh(self.fc3(out))
		return out


class Pixel2Point(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""

	def __init__(self):
		super(Pixel2Point, self).__init__()

		self.FEATURES_NUM = 256

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=128, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
		)
		
		self.decoder = Sequential(
			Flatten(),
			Linear(self.FEATURES_NUM, self.FEATURES_NUM),
			Linear(self.FEATURES_NUM, self.FEATURES_NUM),
			Linear(self.FEATURES_NUM, self.FEATURES_NUM),
			ReLU(),
			Linear(self.FEATURES_NUM, cfg.SAMPLE_SIZE*3),
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class Pixel2Point_InitialPC(Module):
	"""
	For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’ 
	our output image dimension will be [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].
	"""

	def __init__(self):
		super(Pixel2Point_InitialPC, self).__init__()

		self.FEATURES_NUM = 256
		self.INITIAL_SPHERE_POINTS = 16

		self.encoder = Sequential(
			Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=64, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
			Conv2d(in_channels=self.FEATURES_NUM, out_channels=self.FEATURES_NUM, kernel_size=3, stride=2, padding=1), # 4 * 2 * 2
			ReLU(),
		)
		
		self.decoder = Sequential(
			Flatten(),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM)),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM)),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM)),
			ReLU(),
			Linear(self.INITIAL_SPHERE_POINTS*(3+self.FEATURES_NUM), cfg.SAMPLE_SIZE*3),
		)

		self.initialSphere = torch.tensor([
										0.382683, 0.0, 0.92388,
										-0.382683, 0.0, 0.92388,
										0.92388, 0.0, 0.382683,
										0.46194, 0.800103, 0.382683,
										-0.46194, 0.800103, 0.382683,
										-0.92388, 0.0, 0.382683,
										-0.46194, -0.800103, 0.382683,
										0.46194, -0.800103, 0.382683,
										0.92388, 0.0, -0.382683,
										0.46194, 0.800103, -0.382683,
										-0.46194, 0.800103, -0.382683,
										-0.92388, 0.0, -0.382683,
										-0.46194, -0.800103, -0.382683,
										0.46194, -0.800103, -0.382683,
										0.382683, 0.0, -0.92388,
										-0.382683, 0.0, -0.92388
												]).reshape((self.INITIAL_SPHERE_POINTS, 3)).to(device="cuda")


	def forward(self, x):
		encoded = self.encoder(x)
		encoded = torch.flatten(encoded, 2).reshape(-1, self.FEATURES_NUM)
		encoded = torch.transpose(encoded.unsqueeze(2).expand(-1, -1, self.INITIAL_SPHERE_POINTS), 1, 2)
		sphere = self.initialSphere
		# sphere = sphere.unsqueeze(0).expand(1, -1, -1).to('cuda')					   # Run this for testing stage
		sphere = sphere.unsqueeze(0).expand(encoded.shape[0], -1, -1)					# Run this for training stage		
		# print(sphere.shaape, encoded.shape)
		encoded = torch.concat([sphere, encoded], dim=-1)
		# print(encoded.shape)
		decoded = self.decoder(encoded)
		# print(decoded.shape)
		return decoded

