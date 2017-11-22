import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

#Numpy
img = np.random.rand(2,2,2)
a = np.random.rand(1)
b = np.random.rand(1)
test = np.array([img,a,b])
print(a.shape,b.shape,img.ravel().shape)

sample = np.concatenate([img.ravel(), a, b])[:, None]

print(sample.shape)


batch = np.repeat(np.expand_dims(sample, axis=0), 5, axis=0) #shape (32,3,1)

#TF

input = tf.placeholder(dtype=tf.float32, shape=[None] + list(sample.shape))

#tf_img = input
tf_img = input[:, :-2] #get image batch from input
#tf_a = input[:, -2] #get a batch from input
#tf_b = tf.#get b batch from input

#out = layers.convolution2d(input,num_outputs=64,kernel_size=8,stride=2,activation_fn=tf.nn.relu)
#out = layers.flatten(out)
#out = tf.concat([out,tf_a,tf_b])
#out = layers.fully_connected(out,10,activation_fn=tf.nn.relu)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    res = sess.run(tf_img,feed_dict={input:batch})


    print(res - batch[:, :-2])



