import os
import numpy
import matplotlib.pyplot as plt
import sys
import time
from PIL import Image

args = sys.argv[1:]
Train_file = str(args[0])
Test_file = str(args[1])


################################### Read Train Images #####################################

Train = []
Train_imgnum = 0
label = []

with open(Train_file,"r") as file:
	for i in file.readlines():
		dataset = i.split(" ")
		imagePath = dataset[0]
		lab = dataset[1]
		lab_split = lab.split("\n")
		label.append(lab_split[0])
		im = Image.open(imagePath).convert('L')
		img = im.resize((64, 64), Image.ANTIALIAS)
		Train_ar = numpy.array(img).reshape(1, -1)
		Train.append(Train_ar)
		Train_imgnum += 1


Train = numpy.asarray(Train).squeeze(axis=1)
# print (label)
set_a  = set(label)
label_num = [label.count(i) for i in set_a]
tot_label = len(label)


#################################### Read Test Images ####################################

Test = []
Test_imgnum = 0

with open(Test_file,"r") as file:
	for i in file.readlines():
		dataset = i.split("\n")
		imagePath = dataset[0]
		#print (imagePath)
		im = Image.open(imagePath).convert('L')
		img = im.resize((64, 64), Image.ANTIALIAS)
		Test_ar = numpy.array(img).reshape(1, -1)
		Test.append(Test_ar)
		Test_imgnum += 1

Test = numpy.asarray(Test).squeeze(axis=1)

######################################################################################


################################ PCA on Train Images #################################


Train_mean = numpy.mean(Train,axis = 0)
Train_arr = Train - Train_mean
#print (Train_mean)
#print (Train_mean.shape)

Train_trans = numpy.transpose(Train_arr)
Train_array = numpy.matmul(Train_arr,Train_trans)
# print (Train_array)
# print (Train_array.shape)

Train_U, Train_S, Train_V = numpy.linalg.svd(Train_array)
Train_U = numpy.matmul(numpy.transpose(Train_arr),Train_U)
#print (Train_S)
#print (Train_S.shape)


for i in range(Train_imgnum):
    Train_U[:, i] = Train_U[:, i] / numpy.linalg.norm(Train_U[:, i])
    
indx = numpy.argsort(-Train_S)
Train_U = Train_U[:, indx]
Train_S = Train_S[indx]

# print (Train_U)
# print (Train_U.shape)

Train_S = Train_S[:32].copy()
Train_U = Train_U[:, :32].copy()

Train_final = numpy.matmul(Train_arr,Train_U)
# print (Train_final)
# print (Train_final.shape)

for i in range(len(Train_final)):
    Train_final[i, :] = Train_final[i, :] / numpy.linalg.norm(Train_final[i, :])

Train_final = numpy.matrix(Train_final)
# print (Train_final)
# Train_final = Train_final.T

#########################################################################################


####################################### Classes #########################################

# for i in range(1000)
	
k = 0
classes = {}
for i in range(Train_imgnum):

	if (label[i] not in classes):
		classes[label[i]] = [k]
		k = k+1
	classes[label[i]].append(Train_final[i, :])


################################### PCA on Test Images ###################################

Test_mean = numpy.mean(Test,axis = 0)
Test_arr = Test - Test_mean
#print (Test_mean)
#print (Test_mean.shape)

Test_trans = numpy.transpose(Test_arr)
Test_array = numpy.matmul(Test_arr,Test_trans)
# print (Test_array)
# print (Test_array.shape)

Test_U,Test_S,Test_V = numpy.linalg.svd(Test_array)
Test_U = numpy.matmul(numpy.transpose(Test_arr),Test_U)
#print (Test_S)
#print (Test_S.shape)


for i in range(Test_imgnum):
    Test_U[:, i] = Test_U[:, i] / numpy.linalg.norm(Test_U[:, i])
    
indx = numpy.argsort(-Test_S)
Test_U = Test_U[:, indx]
Test_S = Test_S[indx]

# print (Test_U)
# print (Test_U.shape)

Test_S = Test_S[:32].copy()
Test_U = Test_U[:, :32].copy()

Test_final = numpy.matmul(Test_arr,Test_U)


for i in range(len(Test_final)):
    Test_final[i, :] = Test_final[i, :] / numpy.linalg.norm(Test_final[i, :])



##########################################################################################

######################################## Softmax #########################################
# print (label_num)

w = numpy.random.rand(32,len(label_num))
y = [[0 for x in range(Train_imgnum)] for y in range(len(label_num))] 
y  = numpy.matrix(y)

y_cl = []
for cls_label, i in classes.items():
	for j in range(len(i)-1):
		y_cl.append(i[0])

targets = numpy.array(y_cl).reshape(-1)
y = numpy.eye(len(label_num))[targets]

# print (y)

itr = 100000 
eta = 0.000000000001

for i in range(itr):

	xw = numpy.matmul(Train_final,w)

	y_err = y - xw

	J = -numpy.matmul(Train_final.T,y_err)

	w = w - eta*J

# print (w.shape)

for i in range(Test_imgnum):
	out = numpy.matmul(Test_final,w)

p = []

for i in range(Test_imgnum):
	p = numpy.argmax(out,axis = 1)

p = numpy.array(p).squeeze(axis = 1)

final = []

for i in range(Test_imgnum):
	for cls_nam, val in classes.items():
		if(p[i] == val[0]):
			final.append(cls_nam)


for i in range(Test_imgnum):
	print (final[i])