import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')   # only in macOS (slove backend problem of matplotlib 3.0.3 in macOS)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # matplotlib sub-library, only for 3d-visulation


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

def PLA_train(dataSet,maxIter=20):
    '''model training'''
    m = dataSet.shape[0] # number of samples in dataset
    n = dataSet.shape[1] # n (dim) number of features in dataset
    
    # randomise dataset
    np.random.shuffle(dataSet) # shuffle

    # initialise
    w = np.ones((1, n-1)) # weight
    b = 0 # bias
    eta = 1 # learning rate
    i = 0 # itr
    
    for j in range(maxIter):

        # update - SGD
        for k in range(m):
            if dataSet[k][-1] * (np.sum(w * dataSet[k,0:-1],)+ b) <0:
                # if: y * a <= 0
                w = w + eta * dataSet[k][-1] * dataSet[k,0:-1]
                # w(k+1) = w(k) + eta * y * x
                b = b + eta * dataSet[k][-1]
                # b(k+1) = b(k) * y
                i += 1
    return w, b

def PLA_test(w,b,testSet):
    '''model test'''
    TP,TN,FP,FN = 0,0,0,0
    accuracy, precision, recall = 0,0,0
    FPR, TPR, F1Score = 0,0,0

    for i,element in enumerate(testSet):
        a = np.sign(np.dot(element[0:-1],w[0][:])+b)
        if a == element[-1]: # TP,TN
            if a==1: TP+=1 
            else: TN+=1
        else: # FP FN
            if a==1: FP+=1
            else: FN+=1
        
        print('\t pred:fact  {}:{}'.format(int(a),int(testSet[i,-1])))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        F1Score = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError as e:
        print ("\nCAUTION: " + str(e) + "\n")
        return accuracy,0,0,0

    return accuracy, round(FPR,2),round(TPR,2),round(F1Score,2)


def Visualisation3D(dataSet,w,b,dim1=1,dim2=2,dim3=3):
    '''3d visualisation part'''
    x1 = dataSet[:,dim1-1] # x-axis in 3D space
    x2 = dataSet[:,dim2-1] # y-axis in 3D space
    X1, X2 = np.meshgrid(x1,x2)
    X3 = (b - w[0][dim1-1]*X1 - w[0][dim2-1]*X2) / w[0][dim3-1] # z-axis in 3D space
    # PlaneEquation: w1x1 + w2x2 + w3x3 = w0x0

    # create 3d figure
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.set_title('Perceptron Result and Hyperplane')
    plt.xlabel('x' + str(dim1))
    plt.ylabel('x' + str(dim2))
    labels = ['class1', '', 'class2']
    markers = ['o','','x']
    colors = ['g','','r']

    for i in range(3):
        idx = np.where(dataSet[:,-1]==i-1)   # find the samples with same instances
        ax1.scatter(dataSet[idx, dim1-1], dataSet[idx, dim2-1], dataSet[idx, dim3-1],
            marker=markers[i], color=colors[i], label=labels[i], s=10)

    ax1 = fig.gca(projection='3d')
    ax1.plot_surface(X1, X2, X3,color='grey', alpha=0.002) 
    plt.legend(loc = 'upper right')
    plt.tight_layout()    
    plt.show()


if __name__ == "__main__":
    '''main part'''

    # load high-dimentional dataset
    trainSet = data_load("data/train.data")
    testSet  = data_load("data/test.data")

    # model training part 
    w,b = PLA_train(trainSet,maxIter=20)
    print('PLA_weight:',w.squeeze())
    print('PLA_bias  :',b)

    # # model visualisation part
    # Visualisation3D(trainSet,w,b,dim1=1,dim2=2,dim3=3) # alternative one for test
    # !!! please drag the picture to rotate 3D model !!!
    Visualisation3D(trainSet,w,b,dim1=2,dim2=3,dim3=4) 
    # # Equation: w1x1+w2x2+w3x3+w4x4-b=0 (attributes must be continuous)

    # testing part (and return accruacy)
    accuracy,FPR,TPR,F1Score = PLA_test(w,b,testSet)
    print('accruacy: %d%%' % (accuracy*100)) # show percentage
    print('FPR: {},TPR: {},F1Score: {}'.format(FPR,TPR,F1Score))




