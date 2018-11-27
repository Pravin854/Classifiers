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

classes = {}
for i in range(Train_imgnum):

	if (label[i] not in classes):
		classes[label[i]] = []

	classes[label[i]].append(Train_final[i, :])

# class_mean = {}
# class_arr = numpy.array(classes).squeeze(axis=0)
# print (classes)

###################################### Class Mean ########################################


class_mean = {}

for cls_val, vec in classes.items():
	if(cls_val not in class_mean):
		class_mean[cls_val] = []
	# print(len(vec))
	vec = numpy.array(vec).squeeze(axis=1)
	mn = numpy.mean(vec,axis = 0)
	class_mean[cls_val].append(mn)

# print (class_mean)

##########################################################################################

###################################### Class Cov #########################################

var = {}

for cls_val, vec in classes.items():
	if (cls_val not in var):
		var[cls_val] = [] 
	vec = numpy.array(vec).squeeze(axis=1)
	varr = numpy.var(vec,axis = 0)       
	var[cls_val].append(varr)

# print (var)

##########################################################################################

######################################### Prior ##########################################

prior = []
for i in range(len(label_num)):
	prior.append(label_num[i]/tot_label)	
# print (prior)

##########################################################################################


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
# print (Test_final)
# print (Test_final.shape)

for i in range(len(Test_final)):
    Test_final[i, :] = Test_final[i, :] / numpy.linalg.norm(Test_final[i, :])

# Test_final = Test_final.T

##########################################################################################

####################################### Probability ######################################

# print(len(Train_final))
# print (Test_final.shape)
# print (class_mean)
# print (Test_final)

# print (Test_final.shape)
# print (Train_final.shape)

# print (numpy.array(Test_final[1,:]))


sp = Test_final.shape[1]

# 

for k in range(Test_imgnum):
	final_label = None
	final_prob = 0
	for cls_val1, mns in class_mean.items():
		for cls_val2, varrs in var.items():
			if (cls_val1 == cls_val2):
				test_k = numpy.array(Test_final[k,:])
				probab = 1
				for i in range(sp):
					x = test_k[i]
					# print (len(mns))
					u = mns[0][i]
					sig = varrs[0][i]
					# print (x,"\n")	
					# print (u)
					# print (sig)
					# print ("next")
					# print ("class 1",cls_val1)
					# print ("class 2",cls_val2)

					expp = numpy.exp(-((x-u)**2)/(2*sig))
					probab *= (1 / (numpy.sqrt(2*3.14*sig))) * expp

				f_prob = numpy.prod(probab)*(len(classes[cls_val1]))              
				if(f_prob<0):
					f_prob = -f_prob
# 
				# print (f_prob)
				if f_prob > final_prob:
					final_prob = f_prob
					final_label = cls_val1
					# print ("label", final_label)

	print (final_label)


##########################################################################################