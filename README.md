## Perceptron Implementation ## 

                                      Author: Siyu Fang 
                                      University of Liverpool

                                      Python version: 3.0
                                      Matplotlib version: 3.0.3
                                      Numpy version: 1.16.2
                                      IDE: Sublime Text

# 1. PLA and visualisation

The first and second .py files are named `PLA_2D_visualisation` and 

`PLA_3D_visualisation` respectively. Which contain a PLA model for two different 

datasets for implement and validation (one is a 2-d dataset, one is high dimensional dataset). 

# 1.1. 2D visualisation

In the first one, it contains a function `genLinearSeparableData(w,b, numlines)` 

for creating random linear separable data for PLA model, and a `PLA_train()` and `PLA_test()`

for training model, visualisation and model evaluation. for the `PLA_train(trainSet,plot=True)`,

It can return the final weight and bias for the Hyperplane, and decided to visualise the 2D dataset

and the decision region or not. `PLA_test(w,b,testSet,DataVolume)` receives the previous trained 

'weight' and 'bias', and use a LinearSeparableDataSet to test the model, compute TP,TN,FP,FN

accuracy, FPR, TPR, precision, recall, and F-Score, then return the final evaluation results separately.

and the way to use this algorithm has been shown below: 

```python

dataScale = 20
trainSet = genLinearSeparableData([1,-2],7,int(dataScale*0.8)) # get a 2-dim vector data set
testSet  = genLinearSeparableData([1,-2],7,int(dataScale*0.2)) # use 20 percent for testing
w,b = PLA_train(trainSet,plot=True) 
accuracy,FPR,TPR,F1Score = PLA_test(w,b,testSet,int(dataScale*0.2))

```

# 1.2. 3D visualisation

In the second file, it's called `PLA_3D_visualisation`, which specifically designed for

3D visualisation of a high-dimensional Dataset and Hyperplane for validating the PLA model in a more 

intuition way. The overall Algorithm is pretty same as the 1.1., But we use the training dataset from the

Assignment released, so that we build a function named `data_load(dataPath)`, which receive the path of 

data file to process and return an N-D_Array for our PLA model. `Visulation3D()` function is designed for 

visualise the data and Hyperplane in a 3-Dimensional way, which receives the trainSet, trained weight, bias,

and the 3 different dimensions you want to choose. Then to plot a 3-D space for visualising the overall model.

The basic of the hyperplane construction is based on the plane equation `w1x2+w2x2+w3x3-w0x0=0` in which z-axis

equals `(bias - w1x1 - w2x2)/w3)`, and the way to use it has been shown as follow:

```python

trainSet = data_load("train.data") # load 4-dim dataSet from Iris' data
testSet  = data_load("test.data")
w,b = PLA_train(trainSet)
Visualisation3D(trainSet,w,b,dim1=1,dim2=2,dim3=3) # or use dim1=2,dim2=3,dim3=4
accuracy,FPR,TPR,F1Score = PLA_test(w,b,testSet)

```

# 2. Binary Perceptron Class

The .py file of this part is named `myPerceptron.py`

This part is designed for classify the Assignment dataset 'class-1','class-2', and 'class-3'.

It aims to indentify 'class-1' and 'class-2', 'class1' and 'class-3', 'class-2' and 'class-3' respectivily.

The overall workflow like the previous two PLA models, but we built a `PLA()` class be used to create 

each classifier. And `PLA.train()` function will return a `status` to identify the final status of current PLA 

after 20 iterations (or more, you can change the itr times). Because 'maybe' one of these classifiers cannot

go converge forever (data may not belongs to linear separable). and the way to use as the following example:

```python

trainSet_a,testSet_a = batchLoad('a') # only for classifier(a), but samiliar to (b) and (c)

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


```


## If the code cannot be run successfully ##

1. BrokenPipeError
	delete print() function and replaced by an output Stream or run in Anaconda or Colab

2. ERROR at `import matplotlib.pyplot as plt`
	add a code before import plt, `mpl.use('TkAgg')` for changing the backend of plt 3.0.3 in macOS

3. Unknown Error
	contact the author tofangsiyu@gmail.com



