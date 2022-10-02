import os
import numpy
from numpy import random
import scipy
from scipy.special import softmax
import mnist
import pickle
import time
from random import sample


# you can use matplotlib for plotting
import matplotlib
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = numpy.ascontiguousarray(Xs_tr)
        Ys_tr = numpy.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    return - y @ numpy.log(softmax(W  @ x)) + (gamma / 2) * numpy.sum(W **2)

# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the loss with respect to the model parameters W
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    return numpy.outer((softmax(W @ x) - y), x) + gamma * W


# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    d,n = Xs.shape
    c,n = Ys.shape
    print(softmax(W @ Xs))
    M = softmax(W @ Xs)
    idxs= numpy.argmax(M, axis= 0)
    R = numpy.zeros((c,n))
    for i in range(len(idxs)):
        R[idxs[i], i] = 1
    # print(R != Ys)
    # print(n)
    return numpy.sum(R != Ys) / (n * c)
    #return numpy.sum(-0.5 * (numpy.sign(softmax(W @ Xs) * Ys)-1)) / n

# compute the gradient of the multinomial logistic regression objective on a batch, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# ii        indices of the batch (an iterable or range)
#
# returns   the gradient of the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, gamma, W, ii = None):
    if ii is None:
        ii = range(Xs.shape[1])
    # TODO students should implement this
    # a starter solution using an average of the example gradients

    # (softmax(W @ Xs) - Ys) @ Xs.T + gamma * W
    # for i in ii:
        # acc += multinomial_logreg_grad_i(Xs[:, i], Ys[:, i], gamma, W)
    return ((softmax(W @ Xs[:,ii], axis= 0) - Ys[:,ii]) @ Xs[:,ii].T + gamma * W) / len(ii)

# compute the cross-entropy loss of the classifier on a batch, with regularization
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# ii        indices of the batch (an iterable or range)
#
# returns   the model cross-entropy loss
def multinomial_logreg_batch_loss(Xs, Ys, gamma, W, ii = None):
    if ii is None:
        ii = range(Xs.shape[1])
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    # (d, n) = Xs.shape
    # acc = 0.0
    # for i in ii:
    #     acc += multinomial_logreg_loss_i(Xs[:, i], Ys[:, i], gamma, W)
    # return acc / len(ii)
    return - numpy.sum(Ys[:,ii] * numpy.log(softmax(W  @ Xs[:,ii], axis= 0))) / len(ii) + (gamma / 2) * numpy.sum(W **2)
    

# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (c * n)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    Wt = W0
    out = [W0]
    for i in range(num_iters):
        Wt = Wt - alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, Wt)
        if ((i % monitor_freq) + 1) == 0:
            #out.append((i,Wt))
            out.append(Wt)
    return out



# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
#                   to do this, you'll want code like the following:
#                     models = []
#                     models.append(W0.copy())   # (you may not need the copy if you don't mutate W0)
#                     ...
#                     for sgd_iteration in ... :
#                       ...
#                       # code to compute a single SGD update step here
#                       ...
#                       if (it % monitor_period == 0):
#                         models.append(W)
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    _, n = Xs.shape 
    Wt = W0
    models = [W0]
    for i in range(num_epochs):
        for sgd_iteration in range(n // B):
            batch_indices = numpy.random.choice(n, B)
            Wt = Wt - alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, Wt, batch_indices)
            if ((i * (n // B) + sgd_iteration) + 1) % monitor_period== 0:
                models.append(Wt)

    #models.append(Wt)
    return models


# ALGORITHM 2: run stochastic gradient descent with minibatching and sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    _, n = Xs.shape
    assert (n % B) == 0
    Wt = W0
    models = [W0]
    for i in range(num_epochs):
        for j in range(int(n / B)):
            Wt = Wt - alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, Wt, numpy.arange(j * B,(j + 1) * B))
            if ((i * (n / B) + j) + 1) % monitor_period == 0:
                models.append(Wt)
    #models.append(Wt)
    return models

# ALGORITHM 3: run stochastic gradient descent with minibatching and without-replacement sampling
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
def sgd_minibatch_random_reshuffling(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    _, n = Xs.shape
    assert (n % B) == 0
    Wt = W0
    models = [W0]
    for i in range(num_epochs):
        batch_indices = numpy.arange(n)
        numpy.random.shuffle(batch_indices)
        for j in range(n // B):
            Wt = Wt - alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, Wt, batch_indices[(j*B):(j+1)*B])
            if ((i * (n // B) + j) + 1) % monitor_period == 0:
                models.append(Wt)
    #models.append(Wt)
    return models

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    d,n = Xs_tr.shape
    c,n = Ys_tr.shape
    #print(multinomial_logreg_grad_i(numpy.array([3, 6, 2]), numpy.array([1, 4]), 1, numpy.array([[1, 12, 3],[1, 45, 4]])))
    
    #time start
    #start = time.time()

    #run
    #g = gradient_descent(Xs_tr, Ys_tr, 0.0001, numpy.zeros((c,d)), 1, 100, 10)
    
    #time end
    #print(time.time() - start)

    #graph
    #x,y = list(zip(*g))
    #y_loss = [multinomial_logreg_batch_loss(Xs_tr, Ys_tr, 0.0001, yi) for yi in y]
    #print(x)
    #print(y_loss)
    #pyplot.plot(x,y_loss)
    #pyplot.show()

    #error test
    #print(multinomial_logreg_error(numpy.array([[3, 6, 2],[1,1,1]]), numpy.array([[1, 4,1],[2,2,1]]), numpy.array([[1, 12],[1, 45]])))

    #print(sgd_minibatch(Xs_tr, Ys_tr, 0.0001, numpy.zeros((c,d)), 1, 10, 3, 10))
    #print(sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, 0.0001, numpy.zeros((c,d)), 1, 10, 3, 10))
    print(sgd_minibatch_random_reshuffling(Xs_tr, Ys_tr, 0.0001, numpy.zeros((c,d)), 1, 10, 3, 10))
