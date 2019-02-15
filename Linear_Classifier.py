# Importing required libraries
import numpy as np
import operator
import math
from decimal import Decimal
from scipy.special import expit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# reading the train data
X = np.loadtxt("X.csv",delimiter=",")
y = np.loadtxt("y.csv",delimiter=",")

# consolidating the data
data = np.hstack((X,np.reshape(y,(4600,1))))

# splitting the data into 10 parts
Xy_part = {}
for i in range(1,11):
    np.random.shuffle(data)
    Xy_part[i]=data[:460,:]
    np.delete(data,np.s_[0:460],axis=0)

# function to calculate the mle of pi
def pi_mle(y_train):
    # getting labels and counts
    unique_lab, count = np.unique(y_train, return_counts=True)
    output = {}
    # saving the mle for pi
    for i, label in enumerate(unique_lab):
        output[int(label)] = float(count[i])/len(y_train)
    return output

# function to calculate the mle of lamba
def lambda_mle(x_train,y_train):
    # initializing empty lists for storing lambda1 and lambda2
    lambda_0 = []
    lambda_1 = []
    # calculating lambda1 and lambda2 for each dimension
    for i in range(0, x_train.shape[1]):
        lambda_0.append((sum(x_train[y_train == 0][:,i])+1)/(sum(y_train == 0)+1))
        lambda_1.append((sum(x_train[y_train == 1][:,i])+1)/(sum(y_train == 1)+1))
    return lambda_0, lambda_1

#  poisson likelihood function
def PoissonDist(x_test, lambda_0, lambda_1):
    poisson_0 = []
    poisson_1 = []
    for j in range(0,x_test.shape[0]):
        p0=Decimal(1)
        p1=Decimal(1)
        for i in range(0, x_test.shape[1]):
            p0=p0*Decimal(math.exp(-lambda_0[i])) * (Decimal((lambda_0[i] ** x_test[j,i]))/(Decimal(math.factorial(x_test[j,i]))))
            p1=p1*Decimal(math.exp(-lambda_1[i])) * (Decimal((lambda_1[i] ** x_test[j,i]))/(Decimal(math.factorial(x_test[j,i]))))
        poisson_0.append(float(p0))
        poisson_1.append(float(p1))
    return poisson_0, poisson_1


def NaiveBayes(X_train, y_train, X_test, y_test):
    pi = pi_mle(y_train)
    lambda_0, lambda_1 = lambda_mle(X_train,y_train)
    poisson_0, poisson_1 = PoissonDist(X_test,lambda_0,lambda_1)
    prediction = []
    # calculating y_o for lambda_0 and lambda_1
    for i in range(0,len(poisson_0)):
        if ((poisson_0[i]*pi[0])>(poisson_1[i]*pi[1])):
            prediction.append(0)
        else:
            prediction.append(1)
    return prediction

def crossvalidation(data_part):
    pred=[]
    act=[]
    for i in range(1,11):
        temp=[]
        # select 9 out of 10 partitions for training
        for j in range(1,11):
            if(i!=j):
                temp.append(data_part[j])
        train = []
        for sublist in temp:
            for item in sublist:
                train.append(item)
        train = np.array(train)
        # assigning partitions to train and test
        X_train = train[:,0:54]
        y_train = train[:,54]
        X_test = np.array(data_part[i][:,0:54])
        y_test = np.array(data_part[i][:,54])
        pred.append(NaiveBayes(X_train, y_train, X_test, y_test))
        # storing actual and predicted values
        predicted = []
        for sublist in pred:
            for item in sublist:
                predicted.append(item)
        act.append(y_test)
    actual = []
    for sublist in act:
        for item in sublist:
            actual.append(item)
    return predicted, actual

def confusion_matrix(tp, fp, tn, fn ,class_labels=['0','1']):

    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_subplot(111)

    # Drawing the grid lines
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(1.5,-0.5)

    ax.plot([-0.5,1.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,1.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,1.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,1.5], '-k', lw=2)

    # Setting the x labels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_coords(0.5,1.16)

    # Setting the ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1])
    ax.yaxis.set_label_coords(-0.09,0.45)


    # showing the true and false positive and well as the true and false negatives
    ax.text(0,0,'True Negative: %d'%(tn),va='center',ha='center')
    ax.text(0,1,'False Negative: %d'%fn,va='center',ha='center')
    ax.text(1,0,'False Positive: %d'%fp,va='center',ha='center')
    ax.text(1,1,'True Positive: %d'%(tp),va='center',ha='center')

    #  printing the prediction accuracy
    ax.text(2, 0.5, r'Prediction Accuracy = (2300 + 1690)/4600 = 0.8674 or 86.74%', fontsize=15)
    plt.tight_layout()
    plt.show()

def prediction_accuracy(predicted, actual):
    predicted_classes = []
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0,len(predicted)):
        if predicted[i] == actual[i]:
            total += 1
        if (predicted[i] == 0) and (actual[i] == 0):
            tn += 1
        if (predicted[i] == 0) and (actual[i] == 1):
            fn += 1
        if (predicted[i] == 1) and (actual[i] == 0):
            fp += 1
        if (predicted[i] == 1) and (actual[i] == 1):
            tp += 1

    return float(total)/len(predicted), tp, fp, tn, fn

predicted, actual = crossvalidation(Xy_part)

accuracy, tp, fp, tn, fn = prediction_accuracy(predicted, actual)

confusion_matrix(tp, fp, tn, fn)

# calculating the average poisson parameter values
def avg_lambda(data_part):
    l_0= [0] * 54
    l_1= [0] * 54
    for i in range(1,11):
        temp=[]
        # select 9 out of 10 partitions for training
        for j in range(1,11):
            if(i!=j):
                temp.append(data_part[j])
        train = []
        for sublist in temp:
            for item in sublist:
                train.append(item)
        train = np.array(train)
        # assigning partitions to train and test
        X_train = train[:,0:54]
        y_train = train[:,54]
        X_test = np.array(data_part[i][:,0:54])
        y_test = np.array(data_part[i][:,54])
        # calculating lambda values
        lambda_0, lambda_1 = lambda_mle(X_train,y_train)
        for i in range(0,54):
            l_0[i]=l_0[i]+lambda_0[i]
            l_1[i]=l_1[i]+lambda_1[i]
    for i in range(0,54):
        l_0[i] = l_0[i]/10
        l_1[i] = l_1[i]/10

    return l_0, l_1


lambda_0, lambda_1 = avg_lambda(Xy_part)
# plotting the stem plot for the poisson parameters
plt.close('all')
f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
markerline, stemlines, baseline = ax1.stem(range(1, 55), lambda_0, '-.')
plt.setp(baseline, 'color', 'g', 'linewidth', 2)
plt.setp(markerline, 'markerfacecolor', 'r')
plt.setp(stemlines, 'color', 'r')
ax1.set_xlabel("Dimensions")
ax1.set_ylabel("Poisson Parameters")
markerline, stemlines, baseline = ax1.stem(range(1, 55), lambda_1, '-')
plt.setp(baseline, 'color', 'g', 'linewidth', 2)
plt.setp(markerline, 'markerfacecolor', 'y')
plt.setp(stemlines, 'color', 'y','alpha', 0.5)
plt.legend(["Class 0", "Class 1"], loc='best', numpoints=2)
plt.title("Plot for Poisson Parameters when y = 0 (Red) and y = 1 (Yellow)")
plt.show()


print("The average poisson parameters for dimension 16 are: class 0 = ",lambda_0[15]," class 1 = ", lambda_1[15])
print("The average poisson parameters for dimension 52 are: class 0 = ",lambda_0[51]," class 1 = ", lambda_1[51])

# function to compute l1 distance
def l1_distance(v1, v2):
    return np.sum(np.absolute(v1-v2))

# function to get the nearest neighbours
def get_neighbours(X_train, X_test_inst, k):
    dist = []
    nbrs = []
    # finding the distances between data points
    for i in range(0, X_train.shape[0]):
        distance = l1_distance(X_train[i], X_test_inst)
        dist.append((i, distance))
    # storing the top k nearest distances
    dist.sort(key=operator.itemgetter(1))
    for x in range(k):
        nbrs.append(dist[x][0])
    return nbrs

# function for storing and sorting votes based on nearest neighbours
def kNN_predictClass(op, y_train):
    Votes = {}
    for i in range(len(op)):
        if y_train[op[i]] in Votes:
            Votes[y_train[op[i]]] += 1
        else:
            Votes[y_train[op[i]]] = 1
    # sorting the votes
    sortedVotes = sorted(Votes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# function for crossvalidating and testing the KNN
def kNN_test(data_part, k):
    pred = []
    act = []
    for i in range(1,11):
        temp=[]
        # select 9 out of 10 partitions for training
        for j in range(1,11):
            if(i!=j):
                temp.append(data_part[j])
        train = []
        for sublist in temp:
            for item in sublist:
                train.append(item)
        train = np.array(train)
        # assigning partitions to train and test
        X_train = train[:,0:54]
        y_train = train[:,54]
        X_test = np.array(data_part[i][:,0:54])
        y_test = np.array(data_part[i][:,54])
        for i in range(0, X_test.shape[0]):
            op = get_neighbours(X_train, X_test[i], k)
            predictedClass = kNN_predictClass(op, y_train)
            pred.append(predictedClass)
        act.append(y_test)
    actual = []
    for sublist in act:
        for item in sublist:
            actual.append(item)
    return pred, actual

# function to get the accuracy
def prediction_accuracy(pred, actual):
    count = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            count += 1
    return float(count)/len(pred)

# storing the predictions and accuracies for k = 1 to 20
predicted = {}
actual = {}
accuracies = {}
for k in range(1, 21):
    predicted[k], actual[k] = kNN_test(Xy_part, k)
    accuracies[k] = prediction_accuracy(predicted[k], actual[k])

plt.figure(figsize=(15, 6))
plt.plot(accuracies.keys(), accuracies.values())
plt.ylim([0, 1])
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Plot of the prediction accuracy of KNN Classifier as a function of k (Number of Neighbours)")
plt.show()


# add a column of 1s to the data
data_1 = np.hstack((X,np.reshape(y,(4600,1))))

append_ones = np.ones((data.shape[0],1))
data_1 = np.hstack((append_ones, data))

# replacing the 0s with -1
np.place(data_1[:,55], data_1[:,55] == 0, -1)

# splitting the data into 10 parts
Xy_part_1 = {}
for i in range(1,11):
    np.random.shuffle(data_1)
    Xy_part_1[i]=data_1[:460,:]
    np.delete(data,np.s_[0:460],axis=0)

# function to calculate update
def calc_upd(X_train, y_train, w, sig_iter):
    update = np.zeros(len(w))
    for i in range(0, X_train.shape[0]):
        update += y_train[i] * (1 - sig_iter[i]) * X_train[i]
    return update

# calculating the objective function
def calc_objFun(X_train, y_train, w):
    op = 0
    sig_iter = []
    for i in range(0, X_train.shape[0]):
        sig_val = expit(y_train[i] * np.dot(X_train[i], w))
        sig_iter.append(sig_val)
        op = op + np.log(sig_val)
    return op, sig_iter

# performing crossvalidation
obj_val = {}
for i in range(1,11):
    obj_val[i] = []
    temp=[]
    # select 9 out of 10 partitions for training
    for j in range(1,11):
        if(i!=j):
            temp.append(Xy_part_1[j])
    train = []
    for sublist in temp:
        for item in sublist:
            train.append(item)
    train = np.array(train)
    # assigning partitions to train and test
    X_train = train[:,0:55]
    y_train = train[:,55]
    X_test = np.array(Xy_part_1[i][:,0:55])
    y_test = np.array(Xy_part_1[i][:,55])
    w = np.zeros(X_train.shape[1])
    for t in range(1, 10001):
        learning_rate = 0.01 / 4600
        iter_obj_val, sig_iter = calc_objFun(X_train, y_train, w)
        obj_val[i].append(iter_obj_val)
        w = w + (learning_rate * calc_upd(X_train, y_train, w, sig_iter))

# plotting the objective functions for all cv folds
colors = ['b','g','r','c','m','y','b','sienna','olive','darkcyan']
lab = ['cv1','cv2','cv3','cv4','cv5','cv6','cv7','cv8','cv9','cv10']
plt.figure(figsize=(12, 8))
for i in range(1,11):
    plt.plot(obj_val[i],color = colors[i-1], label = lab[i-1])
plt.xlabel("Iterations")
plt.ylabel("Objective Training Function")
plt.legend(loc='best')
plt.title("Logistic Regression objective training function L per iteration for t = 1, 2, ..., 10000")
plt.show()
