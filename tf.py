import tensorflow as tf
import numpy as np

my_data = np.genfromtxt('dataset.csv',delimiter=',',dtype=float)
x_data = np.array(my_data[:,0])
y_data = np.array(my_data[:,1])
print(x_data)
print(y_data)
weights = tf.Variable(tf.random_uniform([1],-1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
#initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in range(20100):
   sess.run(train)
   if step % 2 == 0:
       print(step,'weights',sess.run(weights),'biases', sess.run(biases))
