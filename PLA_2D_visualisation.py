import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')   # only in macOS (slove backend problem of matplotlib 3.0.3 in macOS)
import matplotlib.pyplot as plt

def genLinearSeparableData(w,b, numlines):
    w = np.asarray(w) # make weight as a ndArray
    numFeatures = len(w) # get dimisions
    x = np.random.rand(numlines, numFeatures) * 10  # generate data set
    cls = np.sign(np.sum(w*x,axis=1)+b)    # use wx+b=0 to classify data set
    dataSet = np.column_stack((x,cls))
    # create optLine for Standard
    x = np.linspace(0, 10, 999)      # use 999 plots to generate the opt(std) result
    y1 = -w[0] / w[1] * x - b / w[1] # plane equation w1x1+w2x2-w0x0=0
    rows = np.column_stack((x.T, y1.T,np.zeros((999, 1)))) # combine into one array
    dataSet = np.row_stack((dataSet, rows))
    return dataSet

def visualise2D(dataSet):
    '''model visualisation'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = ['class_1', 'OptBoundary', 'class_2', 'PLA_Result']
    markers = ['o','.','x','.']
    colors = ['green','y','r','b']
    for i in range(4):
        idx = np.where(dataSet[:,2]==i-1)   
        # [-1] is Negative data, [+1] is Positive data, [0] is OptResult, [2] is PLA_result
        ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker=markers[i], \
                                color=colors[i], label=labels[i], s=10)
    plt.legend(loc = 'upper right')
    plt.show()


def PLA_train(dataSet,plot=False):
    '''model training'''

    numLines = dataSet.shape[0] # m
    numFeatures = dataSet.shape[1] # n (dim)
    
    # initalise
    w = np.ones((1, numFeatures-1)) # weight
    b = 0 # bias
    eta = 1 # learning rate
    i = 0 # itr

    # update - SGD
    while i<numLines:
        if dataSet[i][-1] * (np.sum(w * dataSet[i,0:-1],)+ b) < 0:
            w = w + eta * dataSet[i][-1] * dataSet[i,0:-1]
            b = b + eta * dataSet[i][-1]
            i = 0
        else:
            i +=1

    # generate PLA result (Hyperplane)
    x = np.linspace(0,10,999)    # use 999 plots to create Hyperplane
    y = -w[0][0]/w[0][1]*x - b/w[0][1] # plane equation
    rows = np.column_stack((x.T,y.T,2*np.ones((999,1)))) #combine arrays
    dataSet = np.row_stack((dataSet,rows))

    # visualise
    if plot == True: visualise2D(dataSet) # call visualise function

    return w, b

def PLA_test(w,b,testSet,dataVolume):
    '''model test'''
    TP,TN,FP,FN = 0,0,0,0
    accuracy, precision, recall = 0,0,0
    FPR, TPR, F1Score = 0,0,0

    for i,element in enumerate(testSet[0:dataVolume]):
        if i<dataScale*0.2:
            a = np.sign(np.dot(element[0:2],w[0][:])+b)
            if a == element[-1]: 
                if a==1: TP+=1
                else: TN+=1
            else:
                if a == 1: FP+=1
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


if __name__ == "__main__":
    '''main part'''

    dataScale = 200
    # generate Linear Separable Data Set
    trainSet = genLinearSeparableData([1,-2],7,int(dataScale*0.8)) # get a 2-dim vector data set
    testSet  = genLinearSeparableData([1,-2],7,int(dataScale*0.2)) # use 20 percent for testing
    
    # training part (and visualisation)
    w,b = PLA_train(trainSet,plot=True) 
    print('weight: ',w.squeeze())
    print('bias: ',b)

    # testing part (and return accruacy)
    accuracy,FPR,TPR,F1Score = PLA_test(w,b,testSet,int(dataScale*0.2))
    print('accuracy: %d%%' % (accuracy*100)) # show percentage
    print('FPR: {},TPR: {},F1Score: {}'.format(FPR,TPR,F1Score))


