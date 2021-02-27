from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf


housing=fetch_california_housing()


print(housing.data.shape,housing.target.shape)
print(housing.data[0],housing.target[0])

ones_mat=np.ones([20640,1])

X=np.concatenate((housing.data,ones_mat),axis=1)
y=housing.target
print(X.shape,y.shape)

X=tf.constant(X)
y=tf.constant(y)
XT=tf.transpose(X)
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)


print(W)