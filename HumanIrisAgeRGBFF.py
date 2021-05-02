import numpy as np
import csv
import math
import os
from PIL import Image
from pymsgbox import *
from datetime import datetime

class DeepNNetwork:
    def __init__(self):

        # load the dataset from the CSV file
        self.file_dir = "C:/Users/CONALDES/Documents/FUTAFTT/"       

        print("### Input Data ###")
        print("==================")
        
        img_list = []
        y_data = []
        
        files = os.listdir(self.file_dir)
        for f in files:
            file_name = self.file_dir + f
            if os.path.isfile(file_name):
                if file_name[-3:] == "bmp":
                    img_list.append(f)
                elif file_name[-3:] == "png":
                    img_list.append(f)
                    if file_name[-6:-4].isdigit():
                        agestr = file_name[-6:-4]
                    elif file_name[-5:-4].isdigit():
                        agestr = file_name[-5:-4]
                    y_data.append(agestr)
                elif file_name[-3:] == "jpg":
                    img_list.append(f)
                    if file_name[-6:-4].isdigit():
                        agestr = file_name[-6:-4]
                    elif file_name[-5:-4].isdigit():
                        agestr = file_name[-5:-4]
                    y_data.append(agestr)
                    #print("agestr: " + str(agestr))
                    
        print("img_list: " + str(img_list))
        print("                            ") 
        print("ages_list: " + str(y_data))
        ages_data = []
        allages = np.array(y_data).astype("float")
        for i in range(len(allages)):
            ages_data.append([allages[i]])
        #print("ages_data: " + str(ages_data))
        print("                            ")    
        self.data_y = np.vstack(ages_data)
        #print("ages: " + str(self.data_y))

        print("Image conversion to grayscale going on ......")
        print("                            ") 
        
        self.imgLstLen = len(img_list)   
        self.pxelArray = getAgeRGBArrays(self.file_dir, img_list)

        self.data_x = np.concatenate((self.pxelArray, np.ones((self.pxelArray.shape[0],1))), axis=1)
        #self.data_y = self.ages[:,1:]
        
        #print("self.pxelArray: " + str(self.pxelArray))
        #print("self.data_x: " + str(self.data_x))
        #self.data_x = np.zeros(self.pxelArray.shape)
        #self.data_x = self.pxelArray
        #self.datay = self.pxelArray.shape[0]
        
        print("                              ")
        print("### RGBA Attributes (data_x with biases) and Age (data_y) ###")
        print("=============================================================")
        print("data_x: " + str(self.pxelArray))
        print("data_y: " + str(self.data_y))
        print("                              ")

        #print("self.data_x.shape: " + str(self.data_x.shape))
        nrow, self.ncol = self.data_x.shape

        print("                              ")
        print("### Normalisation process going on .......... ###")
        print("                              ")    

        meanRgb = {}
        stdRgb = {}
        self.mean_RGB = {}
        self.std_RGB = {}
        
        for l in range(0, (self.ncol - 1)):
            meanRgb[l], stdRgb[l], self.mean_RGB[l], self.std_RGB[l] = (None,)*4

        for j in range(0, (self.ncol - 1)):    
            tempval = self.data_x[:,j]            
            meanRgb[j] = np.mean(tempval)
            stdRgb[j] = np.std(tempval)
            for row in range(len(self.data_x)):
                temp = (self.data_x[row, j] - meanRgb[j])/stdRgb[j]
                self.data_x[row, j] = temp
                
        self.datay = []
        
        self.min_age = np.min(self.data_y )
        self.max_age = np.max(self.data_y )
        for row in range(len(self.data_x)):
            temp = (self.data_y[row] - self.min_age)/(self.max_age - self.min_age)
            self.datay.append(temp)
            
        self.predict = self.saveDataSetMetadata(meanRgb, stdRgb, self.min_age, self.max_age)
        
        self.data_yy = np.array(self.datay) 

        # we set a threshold at 80% of the data
        self.m = float(self.pxelArray.shape[0])
        self.m_train_set = int(self.m * 0.8)
        
        print("### Traning Set (80%) and Testing Set (20%) ###")
        print("===============================================")
        #print("m_train_set: " + str(roundup(self.m_train_set,0)))
        #print("m_test_set: " + str(roundup((self.m - self.m_train_set),0)))
        print("m_train_set: " + str(self.m_train_set))
        print("m_test_set: " + str(int(self.m - self.m_train_set)))   
        print("                              ")

        # we split the train and test set using the threshold
        self.x, self.x_test = self.data_x[:self.m_train_set,:], self.data_x[self.m_train_set:,:]            
        self.y, self.y_test = self.data_yy[:self.m_train_set,:], self.data_yy[self.m_train_set:,:]

        #print("self.data_x: " + str(self.data_x))
        #print("self.data_yy: " + str(self.data_yy))

        print("### Normalized Traning Data (x with biases) ###")
        print("===============================================")
        print("x: " + str(self.x))
        print("y: " + str(self.y))                   
        print("                              ")
        print("### Normalized Testing Data (x_test with biases) ###")
        print("====================================================")
        print("x_test: " + str(self.x_test))
        print("y_test: " + str(self.y_test))                       
        print("                              ")

        # we init the network parameters
        self.z2, self.a2, self.z3, self.a3, self.z4, self.a4 = (None,) * 6
        self.delta2, self.delta3, self.delta4 = (None,) * 3
        self.djdw1, self.djdw2, self.djdw3 = (None,) * 3
        self.gradient, self.numericalGradient, self.chkedgradt = (None,) * 3
        self.Lambda = 0.1   # For regularization
        self.learning_rate = 0.1

        #parameters
        inputSize = 9
        hidden1Size = 3
        hidden2Size = 2
        outputSize = 1        

        # init weights
        np.random.seed(0)
        WT1 = np.random.rand(inputSize, hidden1Size)    # (9x3) weight matrix from input to hidden layer
        WT2 = np.random.rand(hidden1Size, hidden2Size)  # (3x2) weight matrix from hidden to output layer
        WT3 = np.random.rand(hidden2Size, outputSize)   # (2x1) weight matrix from hidden to output layer

        ww1 = np.append(WT1, [[0.1, 0.1, 0.1]], axis=0)
        ww2 = np.append(WT2, [[0.1, 0.1]], axis=0)
        ww3 = np.append(WT3, [[0.1]], axis=0)
        
        # convert weights to matrix
        self.w1 = np.matrix(ww1)
        self.w2 = np.matrix(ww2)
        self.w3 = np.matrix(ww3)
        
        print("### Weights Generated (with biases) ###")
        print("=======================================")
        print("w1: " + str(self.w1))
        print("w2: " + str(self.w2))
        print("w3: " + str(self.w3))
        print("                              ")

        print("self.x.shape: " + str(self.x.shape))
        print("self.w1.shape: " + str(self.w1.shape))
        print("self.w2.shape: " + str(self.w2.shape))
        print("self.w3.shape: " + str(self.w3.shape))

        self.gflag = 1
        
    def forward(self):   
        # first layer
        self.z2 = np.dot(self.x, self.w1)
        #print("forward self.x: " + str(self.x))
        #print("forward self.w1: " + str(self.w1))
        self.a2 = np.tanh(self.z2)

        # we add the 1 unit (bias) at the output of the first layer
        ba2 = np.ones((self.x.shape[0], 1))
        self.a2 = np.concatenate((self.a2, ba2), axis=1)

        # second layer
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = np.tanh(self.z3)

        # we add the 1 unit (bias) at the output of the second layer
        ba3 = np.ones((self.a3.shape[0], 1))
        self.a3 = np.concatenate((self.a3, ba3), axis=1)

        # output layer, prediction of our network
        self.z4 = np.dot(self.a3, self.w3)
        self.a4 = np.tanh(self.z4)
        #print("self.a4: " + str(self.a4))
        
    # back propagation with regularisation
    def backward(self):
        
        # gradient of the cost function with regards to W3
        self.delta4 = np.multiply(-(self.y - self.a4), tanh_prime(self.z4))
        self.djdw3 = ((self.a3.T * self.delta4) / self.m_train_set) + self.Lambda * self.w3/self.m_train_set
        
        # gradient of the cost function with regards to W2
        self.delta3 = np.multiply(self.delta4 * self.w3.T, tanh_prime(np.concatenate((self.z3, np.ones((self.z3.shape[0], 1))), axis=1)))
	#np.delete(self.delta3, 2, axis=1) removes the bias term from the backpropagation
        self.djdw2 = ((self.a2.T * np.delete(self.delta3, 2, axis=1)) / self.m_train_set) + self.Lambda * self.w2/self.m_train_set
        
        # gradient of the cost function with regards to W1
	#np.delete(self.delta3, 2, axis=1) removes the bias term from the backpropagation
        self.delta2 = np.multiply(np.delete(self.delta3, 2, axis=1) * self.w2.T, tanh_prime(np.concatenate((self.z2, np.ones((self.z2.shape[0], 1))), axis=1)))
        #np.delete(self.delta2, 3, axis=1) removes the bias term from the backpropagation
        self.djdw1 = ((self.x.T * np.delete(self.delta2, 3, axis=1)) / self.m_train_set) + self.Lambda * self.w1/self.m_train_set
        
    def update_gradient(self):
        # division by self.m_train_set taken care of in back propagation
        # Gradient descent learning rule. Weight (w) rescaled by a factor (1 - (self.Lambda * self.learning_rate)/self.m_train_set) called weight decay.         
        
        self.w1 += - self.learning_rate * (self.djdw1 + self.Lambda * self.w1/self.m_train_set)
        self.w2 += - self.learning_rate * (self.djdw2 + self.Lambda * self.w2/self.m_train_set)
        self.w3 += - self.learning_rate * (self.djdw3 + self.Lambda * self.w3/self.m_train_set)

    def cost_function(self):
        # quadratic cost function plus sum of the squares of all the weights in the network.
        return (0.5 * sum(np.square((self.y - self.a4))))/self.m_train_set + ((0.5*self.Lambda) / self.m_train_set) * (
            np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)) + np.sum(np.square(self.w3)))

    def set_weights(self, weights):
        self.w1 = np.reshape(weights[0:30], (10, 3))
        self.w2 = np.reshape(weights[30:38], (4, 2))
        self.w3 = np.reshape(weights[38:41], (3, 1))

    def compute_gradients(self):
        # reval is a real value of complex number
        self.gradient = np.concatenate((self.djdw1.ravel(), self.djdw2.ravel(), self.djdw3.ravel()), axis=1).T

    def compute_numerical_gradients(self):
        weights = np.concatenate((self.w1.ravel(), self.w2.ravel(), self.w3.ravel()), axis=1).T

        self.numericalGradient = np.zeros(weights.shape)
        perturbation = np.zeros(weights.shape)
        e = 1e-4
        #print("Weights size: " + str(len(perturbation)))
        #print("================")
        #print("weights: " + str(weights))
        for p in range(len(perturbation)):
            # Set perturbation vector
            perturbation[p,0] = e	#All elements of perturbation take 0.0001 one after the other in the iteration cycle
            #print("perturbation: " + str(p) + ", " + str(perturbation))            
            self.set_weights(weights + perturbation)
            #print("weights + perturbation: " + str(p) + ", " + str(weights + perturbation))
            self.forward()
            loss2 = self.cost_function()

            self.set_weights(weights - perturbation)
            #print("weights - perturbation: " + str(p) + ", " + str(weights - perturbation))
            self.forward()
            loss1 = self.cost_function()

            self.numericalGradient[p,0] = (loss2 - loss1) / (2 * e)
            
            #print("weight disturbed " + str(p + 1) + ": (index " + str(p) + ") ")
            #print("weights:     weights + perturbation:   weights - perturbation:   numericalGradient:")
            #print(str(roundup(weights[p,0],8)) + "       " + str(roundup((weights + perturbation)[p,0],8)) + "                  " + str(
            #    roundup((weights + perturbation)[p,0],8)) + "            " + str(roundup(self.numericalGradient[p,0],8)))

            perturbation = np.zeros(weights.shape)            

        self.set_weights(weights)   # Reset weights to normal, i.e. without perturbation        

    def check_gradients(self):
        self.compute_gradients()
        self.compute_numerical_gradients()
        #print("                             ")
        #print("Vector of elements of gradients computed during backpropagation (V1) and ")
        #print("Vector of elements of numerical gradients (V2) compared: ")
        #print("       V1,                      V2")
        #for p in range(len(self.gradient)):
        #    print(str(self.gradient[p,0]) + ",      " + str(self.numericalGradient[p,0]))
        
        self.chkedgradt = np.linalg.norm(self.gradient - self.numericalGradient) / np.linalg.norm(
            self.gradient + self.numericalGradient)
        
        #print("                             ")
        print("Gradient checked: " + str(self.chkedgradt))

    #def predict(self, X):
    #    self.x = X
    #    self.forward()
    #    return self.a4

    def saveDataSetMetadata(self, mean_rgb, std_rgb, min_age, max_age):
        for j in range(0, (self.ncol - 1)):    
            self.mean_RGB[j] = mean_rgb[j]
            self.std_RGB[j] = std_rgb[j] 
        
        self.min_age = min_age
        self.max_age = max_age        
   
    def input(self, data_xx):
        nrow, ncol = data_xx.shape
        #print("data_xx.shape: " + str(data_xx.shape))
        
        RGB = {}
        for l in range(0, ncol):
            RGB[l] = (None,)
            
        for j in range(0, ncol):    
            RGB[j] = (data_xx[0, j] - self.mean_RGB[j]) / self.std_RGB[j]

        rgbvals = []
        for i in range(0, ncol):
            rgbvals.append(RGB[i])
            
        #print("np.matrix([rgbvals]): " + str(np.matrix([rgbvals])))
        return np.matrix([rgbvals])

    def output(self, age):
        return age * (self.max_age - self.min_age) + self.min_age

    def r2(self):
        y_mean = np.mean(self.y)
        ss_res = np.sum(np.square(self.y - self.a4))
        ss_tot = np.sum(np.square(self.y - y_mean))
        return 1 - (ss_res / ss_tot)

    def summary(self, step):
        print("Loss %f" % (self.cost_function()))
        print("RMSE: " + str(np.sqrt(np.mean(np.square(self.a4 - self.y)))))
        print("MAE: " + str(np.sum(np.absolute(self.a4 - self.y)) / self.m_train_set))
        #print("R2: " + str(self.r2()))

    def predict_age(self, file_dir, img_name, gflag):
        #print("predict_age -> file_dir: " + str(file_dir))
        #print("predict_age -> img_name: " + str(img_name))        
        pxel_array = getImageAvgRGB(file_dir, img_name, gflag)        
        #print("pxel_array: " + str(pxel_array))
        self.x = np.concatenate((self.input(pxel_array), np.ones((1, 1))), axis=1)
        nn.forward()
        #print("Predicted Age: " + str(roundup(self.output(self.a4[0,0]), 2)))
        return roundup(self.output(self.a4[0,0]), 2)        

def getAgeRGBArrays(file_dir, img_list):
    pxel_array = []
    img_list_len = len(img_list)
    
    for i in range(0, img_list_len):
        file_name = file_dir + img_list[i]
        img = Image.open(file_name, "r")
        pix_val = list(img.getdata())
        #pix_val_flat = [x for sets in pix_val for x in sets]
        #print("pix_val: " + str(pix_val))
        #pix_val_flat               
        img_array1 = []
        img_array2 = []
        img_array3 = []
        #combined_array = []
        temp_array = []
        
        listlen = len(pix_val)
        #print("listlen: " + str(listlen))
        n1rd = int(listlen/3)
        #print("1st 3rd listlen: " + str(n1rd))
        n2rd = int((listlen - n1rd)/2)
        #print("2nd 3rd: " + str(n2rd))
        n3rd = listlen - n1rd - n2rd
        #print("3rd 3rd: " + str(n3rd))
        trd = n1rd + n2rd + n3rd
        #print("Total 1rd2rd3rd: " + str(trd))
        #print("trd == listlen: " + str(trd == listlen))

        sum_elem0 = 0
        sum_elem1 = 0
        sum_elem2 = 0 
        for l in range(0, n1rd):
            sum_elem0 += pix_val[l][0]
            #print("sum_elem0: " + str(l) + ", " + str(0) + ", " + str(sum_elem0))
            sum_elem1 += pix_val[l][1]
            #print("sum_elem1: " + str(l) + ", " + str(1) + ", " + str(sum_elem1))
            sum_elem2 += pix_val[l][2]
            #print("sum_elem2: " + str(l) + ", " + str(2) + ", " + str(sum_elem2))
            
        img_array1.append([sum_elem0,sum_elem1,sum_elem2])            

        sum_elem0 = 0
        sum_elem1 = 0
        sum_elem2 = 0 
        for l in range(n1rd, (n1rd + n2rd)):
            sum_elem0 += pix_val[l][0]
            sum_elem1 += pix_val[l][1]
            sum_elem2 += pix_val[l][2]
             
        img_array2.append([sum_elem0,sum_elem1,sum_elem2])            

        sum_elem0 = 0
        sum_elem1 = 0
        sum_elem2 = 0 
        for l in range((n1rd + n2rd), listlen):
            sum_elem0 += pix_val[l][0]
            sum_elem1 += pix_val[l][1]
            sum_elem2 += pix_val[l][2]
             
        img_array3.append([sum_elem0,sum_elem1,sum_elem2])

        #print("img_array1: " + str(img_array1))
        #print("img_array2: " + str(img_array2))
        #print("img_array3: " + str(img_array3))

        temp_array_flat = []
        for k in range(0, len(img_array1)):
            lst1 = img_array1[k]
            for j in range(0, len(lst1)):
                temp_array.append(lst1[j])
                                  
            lst2 = img_array2[k]
            for j in range(0, len(lst2)):
                temp_array.append(lst2[j])
            
            lst3 = img_array3[k]
            for j in range(0, len(lst3)):
                temp_array.append(lst3[j])
                
        #print("temp_array: " + str(temp_array))   
            
        pxel_array.append(temp_array)

    pxelArray = np.matrix(pxel_array)    
    #feature_set = np.vstack(pxel_array)
    #print("pxelArray.shape: " + str(pxelArray.shape))
    #print("feature_set.shape: " + str(feature_set.shape))
    #print("pxelArray: " + str(pxelArray))
    #print("feature_set: " + str(feature_set))
    return pxelArray

def getImageAvgRGB(file_dir, img_name_ext, gflag):
    #print("getImageAvgRGB -> file_dir: " + str(file_dir))
    #print("getImageAvgRGB -> img_name_ext: " + str(img_name_ext))
    pxel_array = []    
    #file_name = file_dir + img_name + ".bmp"       
    file_name = file_dir + img_name_ext
    #print("getImageAvgRGB -> file_name: " + str(file_name))
    #file_name = file_dir + img_list[0]
    img = Image.open(file_name, "r")
    pix_val = list(img.getdata())
    if (gflag == 1):
        print("grayscale: " + str(pix_val))
    img_array1 = []
    img_array2 = []
    img_array3 = []
    #combined_array = []
    temp_array = []
        
    listlen = len(pix_val)
    n1rd = int(listlen/3)
    n2rd = int((listlen - n1rd)/2)
    n3rd = listlen - n1rd - n2rd
    trd = n1rd + n2rd + n3rd

    sum_elem0 = 0
    sum_elem1 = 0
    sum_elem2 = 0 
    for l in range(0, n1rd):
        sum_elem0 += pix_val[l][0]
        sum_elem1 += pix_val[l][1]
        sum_elem2 += pix_val[l][2]
            
    img_array1.append([sum_elem0,sum_elem1,sum_elem2])            

    sum_elem0 = 0
    sum_elem1 = 0
    sum_elem2 = 0 
    for l in range(n1rd, (n1rd + n2rd)):
        sum_elem0 += pix_val[l][0]
        sum_elem1 += pix_val[l][1]
        sum_elem2 += pix_val[l][2]
             
    img_array2.append([sum_elem0,sum_elem1,sum_elem2])            

    sum_elem0 = 0
    sum_elem1 = 0
    sum_elem2 = 0 
    for l in range((n1rd + n2rd), listlen):
        sum_elem0 += pix_val[l][0]
        sum_elem1 += pix_val[l][1]
        sum_elem2 += pix_val[l][2]
             
    img_array3.append([sum_elem0,sum_elem1,sum_elem2])
        
    temp_array_flat = []
    for k in range(0, len(img_array1)):
        lst1 = img_array1[k]
        for j in range(0, len(lst1)):
            temp_array.append(lst1[j])
                                  
        lst2 = img_array2[k]
        for j in range(0, len(lst2)):
            temp_array.append(lst2[j])
            
        lst3 = img_array3[k]
        for j in range(0, len(lst3)):
            temp_array.append(lst3[j])
           
    pxel_array.append(temp_array)

    pxelArray = np.matrix(pxel_array) 
    return pxelArray

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def roundup(a, digits=0):
    #n = 10**-digits
    #return round(math.ceil(a / n) * n, digits)
    return round(a, digits)

def tanh_prime(x):
    return 1.0 - np.square(np.tanh(x))

def getImageName():
    t_file_dir = "C:/Users/CONALDES/Documents/tempPics/"
    files = os.listdir(t_file_dir)
    #print("files: " + str(files))
    imageName = ""
    file_name = ""
    person_name = ""
    file_exist = False
    for f in files:
        file_name = t_file_dir + f
        if os.path.isfile(file_name):
            if file_name[-3:] == "bmp":
                person_name = f[0:-4]
                file_exist = True
            elif file_name[-3:] == "png":
                person_name = f[0:-4]
                file_exist = True
            elif file_name[-3:] == "jpg":
                person_name = f[0:-4]
                file_exist = True
    if file_exist:
        imageName = files[0]
        
    #print("t_file_dir: " + str(t_file_dir))
    #print("imageName: " + str(imageName))
    return t_file_dir, imageName, person_name

def moveImage(t_file_dir, imageName, predicted_age):
    #t_file_dir = "C:/Users/CONALDES/Documents/tempPics/"
    r_file_dir = "C:/Users/CONALDES/Documents/rejectedPics/"
    a_file_dir = "C:/Users/CONALDES/Documents/acceptedPics/"
    file_name = t_file_dir + imageName
    if predicted_age < 18:
        try:
            image = Image.open(file_name, "r")
            #Saved in the same relative location 
            image.save(r_file_dir + imageName)
            # show moved image
            image.show()                     
        except IOError: 
            pass
    elif predicted_age >= 18:        
        try:
            image = Image.open(file_name, "r")
            #Saved in the same relative location 
            image.save(a_file_dir + imageName)
            # show moved image
            image.show()                     
        except IOError: 
            pass

    files = os.listdir(t_file_dir)
    img_list = []
    file_exist = False
    for f in files:
        file_name = t_file_dir + f
        if os.path.isfile(file_name):
            if file_name[-3:] == "bmp":
                file_exist = True
            elif file_name[-3:] == "png":
                img_list.append(f)
                file_exist = True
            elif file_name[-3:] == "jpg":
                img_list.append(f)
                file_exist = True

    if file_exist:            
        os.remove (file_name)
    
nn = DeepNNetwork()

# current date and time
now = datetime.now()
starting_time = now.strftime("%H:%M:%S")
timestamp1 = datetime.timestamp(now)

done = False

nb_it = 10000000
for step in range(nb_it):
    nn.forward()
    nn.backward()
    nn.update_gradient()

    nn.check_gradients()
    if nn.chkedgradt < 5.35e-06:
        done = True
        now = datetime.now()
        stopping_time = now.strftime("%H:%M:%S")
        print("                              ")
        print("Back Propagation computes correctly the gradients:")
        print("==================================================")
        print("If gradient checked is in the order of 10e-06, it is a good approximation")
        print("Iteration: %d, " % (step + 1))
        print("Starting Time = ", starting_time)
        print("Stopping Time = ", stopping_time)
        timestamp2 = datetime.timestamp(now)
        print("Time Elapsed = " + str(roundup((timestamp2 - timestamp1), 2)) + "secs")
        print("                             ")
        break

if not done:    
    now = datetime.now()
    stopping_time = now.strftime("%H:%M:%S")
    print("                              ")
    print("Iteration: %d, " % (step + 1))
    print("Starting Time = ", starting_time)
    print("Stopping Time = ", stopping_time)
    timestamp2 = datetime.timestamp(now)
    print("Time Elapsed = " + str(roundup((timestamp2 - timestamp1), 2)) + "secs")
    print("                             ")

print("### Weights After Training (with biases) ###")
print("============================================")
print("w1: " + str(nn.w1))
print("w2: " + str(nn.w2))
print("w3: " + str(nn.w3))
print("                              ")

print("### Computed Output (y), Network Generated Output (a4) and Difference ###")
print("=========================================================================")
print("   y        (a4)      (y - a4)")
print("==============================")
for i in range(nn.m_train_set):
    nn.y[i,0] = (nn.max_age - nn.min_age)*nn.y[i,0] + nn.min_age
    nn.a4[i,0] = (nn.max_age - nn.min_age)*nn.a4[i,0] + nn.min_age
for i in range(nn.m_train_set):    
    print("  " + str(roundup(nn.y[i,0], 2)) + "     " + str(roundup(nn.a4[i,0],2)) + "       " + str(roundup((nn.y[i,0]-nn.a4[i,0]),2)))

print("                              ")

nn.x = nn.x_test
nn.y = nn.y_test
nn.forward()

print("### Testing summary ###")
print("=======================")
nn.summary(nb_it)
print("                              ")
print("### Predict ###")
print("===============")

predicted_age = 0.00
while True:
    status = input("Predict a person's age (yes or no)? ")
    if status.lower() == "yes":
        t_file_dir, image_name, person_name = getImageName()
        #print("t_file_dir: " + str(t_file_dir))
        #print("image_name: " + str(image_name))
        try:
            predicted_age = nn.predict_age(t_file_dir, image_name, nn.gflag)      # + "." + image_ext)
            msg = "Name: " + str(person_name) + "\nAge: "  + str(predicted_age) + " years\n"
            if predicted_age < 18:
                msg = msg + "Voting Status: You are not eligible to vote" 
            elif predicted_age >= 18:
                msg = msg + "Voting Status: You are eligible to vote"
            alert(msg, "Screening Information", button='OK')
            #print("Predicted Age: " + str(predicted_age))
            #print("                              ")
            nn.gflag = 2
        except Exception as e:
            print(e)
        moveImage(t_file_dir, image_name, predicted_age)
           
    else:       # status.lower() == "no":
        break
    


