import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')   # only in macOS (slove backend problem of matplotlib 3.0.3 in macOS)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # matplotlib sub-library, only for 3d-visulation

class PLA():
    def __init__ (self,trainSet,testSet):
        self.trainSet = trainSet
        self.testSet = testSet

    def train(self,maxIter=20):
        '''model training'''
        m = self.trainSet.shape[0] # number of samples in dataset
        n = self.trainSet.shape[1] # n (dim) number of features in dataset
        
        # randomise dataset
        np.random.shuffle(self.trainSet) # shuffle

        # initialise
        self.w = np.ones((1, n-1)) # weight
        self.b = 0 # bias
        eta = 1 # learning rate (optional)
        i = 0 # itr_account
        for j in range(maxIter):
            # for x1, x2, ..., xj
            for k in range(m):
                # update - SGD
                if self.trainSet[k][-1] * (np.sum(self.w * self.trainSet[k,0:-1],) + self.b) <0:
                    # if: y * a <= 0
                    self.w = self.w + eta * self.trainSet[k][-1] * self.trainSet[k,0:-1]
                    # w(k+1) = w(k) + eta * y * x
                    self.b = self.b + eta * self.trainSet[k][-1]
                    # b(k+1) = b(k) * y
                    i += 1
        
        # identify the convergence
        if i>maxIter:
            self.status = 'UNCONVERGED'
        else:
            self.status = 'converged' 

        return self.w, self.b, self.status

    def test(self):
        '''model test'''
        self.TP,self.TN,self.FP,self.FN = 0,0,0,0
        self.accuracy, self.precision, self.recall = 0,0,0
        self.FPR, self.TPR, self.F1Score = 0,0,0
        self.y_pred, self.y_fact = [], [] # store prediction 

        for i,element in enumerate(self.testSet):
            a = np.sign(np.dot(element[0:-1],self.w[0][:])+self.b)
            if a == element[-1]: 
                if a==1: self.TP+=1
                else: self.TN+=1
            else:
                if a == 1: self.FP+=1
                else: self.FN+=1
            self.y_pred.append('+1' if int(a)==1 else '-1')
            self.y_fact.append('+1' if int(self.testSet[i,-1])==1 else '-1')
            # print('\t pred:fact  {}:{}'.format(int(a),int(self.testSet[i,-1])))
        
        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        try:
            self.precision = self.TP / (self.TP + self.FP)
            self.recall = self.TP / (self.TP + self.FN)
            self.FPR = self.FP / (self.FP + self.TN)
            self.TPR = self.TP / (self.TP + self.FN)
            self.F1Score = (2 * self.precision * self.recall) / (self.precision + self.recall)
        except ZeroDivisionError as e:
            print ("\nCAUTION: " + str(e) + "\n")
            return self.accuracy,0,0,0

        return self.accuracy, round(self.FPR,2),round(self.TPR,2),round(self.F1Score,2)

def data_load(dataPath): 
    '''load specific data set from specific path'''
    dataSetCounter = len(open(dataPath,'rU').readlines()) # counting total inputs of training data set
    x = []
    Set = open(dataPath,'r')
    for i in range(0, dataSetCounter):
        data = Set.readline().replace('\n','').split(',') # split one data into a list
        get_data = [float(i) for i in data] # convert list into float-type 
        x.append(get_data)
    arr = np.asarray(x) #convert list into n-dim-array
    Set.close()
    return arr #return processed training data set

def batchLoad(Classifier):
    trainSet = data_load("data/train_{}.data".format(Classifier))
    testSet  = data_load("data/test_{}.data".format(Classifier))
    return trainSet,testSet

def Visualisation3D(Classifier,num,dim1=1,dim2=2,dim3=3):
    '''3d visualisation part'''
    # Equation: w1x1+w2x2+w3x3+w4x4-b=0 (attributes must be continuous)
    x1 = Classifier.trainSet[:,dim1-1] # x-axis in 3D space
    x2 = Classifier.trainSet[:,dim2-1] # y-axis in 3D space
    X1, X2 = np.meshgrid(x1,x2)
    w = Classifier.w
    b = Classifier.b
    X3 = (b - w[0][dim1-1]*X1 - w[0][dim2-1]*X2) / w[0][dim3-1] # z-axis in 3D space
    # PlaneEquation: w1x1 + w2x2 + w3x3 = w0x0

    # create 3d figure
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.set_title('Perceptron Result and Hyperplane - Classifier#({})'.format(num))
    plt.xlabel('x' + str(dim1))
    plt.ylabel('x' + str(dim2))
    labels = ['class1', '', 'class2']
    markers = ['o','','x']
    colors = ['g','','r']

    for i in range(3):
        idx = np.where(Classifier.trainSet[:,-1]==i-1)   # find the samples with same instances
        ax1.scatter(Classifier.trainSet[idx, dim1-1], Classifier.trainSet[idx, dim2-1], \
                        Classifier.trainSet[idx, dim3-1], \
                        marker=markers[i], color=colors[i], \
                        label=labels[i], s=10)

    ax1 = fig.gca(projection='3d')
    ax1.plot_surface(X1, X2, X3,color='grey', alpha=0.002) 
    plt.legend(loc = 'upper right')
    plt.tight_layout()    
    plt.show()

def visualise():
    Visualisation3D(Classifier_a,'a',dim1=2,dim2=3,dim3=4) 
    # Visualisation3D(trainSet_a,'a',w_1,b_1,dim1=1,dim2=2,dim3=3) # alternative one
    Visualisation3D(Classifier_b,'b',dim1=2,dim2=3,dim3=4) 
    Visualisation3D(Classifier_c,'c',dim1=2,dim2=3,dim3=4) 

def output(Classifier,num):
    print('\nClassifier#({}): {}'.format(str(num),Classifier.status))
    print('PLA_weight: {}\nPLA_bias  :   {}'.format(Classifier.w.squeeze(),Classifier.b))
    print('Prediction: {}\nTarget    : {}'.format(Classifier.y_pred,Classifier.y_fact))    
    print('accruacy  : {}% \nFPR: {}, TPR: {}, F1Score: {}, Precision : {}, Recall : {}'.format(\
          Classifier.accuracy*100,Classifier.FPR,Classifier.TPR,round(Classifier.F1Score,2), \
          round(Classifier.precision,2),Classifier.recall))  


if __name__ == "__main__":
    '''
           data_loading
                            '''
    trainSet_a,testSet_a = batchLoad('a')
    trainSet_b,testSet_b = batchLoad('b')
    trainSet_c,testSet_c = batchLoad('c')
    
    maxIter = 20

    '''
            Classifier_a
                            '''
    Classifier_a = PLA(trainSet_a,testSet_a)
    # train
    Classifier_a.train(maxIter) 
    # test
    Classifier_a.test()
    # output result
    output(Classifier_a,'a')


    '''
            Classifier_b
                            '''
    Classifier_b = PLA(trainSet_b,testSet_b)
    # train
    Classifier_b.train(maxIter) 
    # test
    Classifier_b.test()
    # output result
    output(Classifier_b,'b')


    '''
            Classifier_c
                            '''
    Classifier_c = PLA(trainSet_c,testSet_c)
    # train
    Classifier_c.train(maxIter) 
    # test
    Classifier_c.test()
    # output result
    output(Classifier_c,'c')


    '''
            visualise ALL
                            '''
    visualise()    
    # !!! remember to drag the picture for rotating 3D model !!!



