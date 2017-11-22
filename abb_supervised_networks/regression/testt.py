

import tensorflow as tf

"""
var = tf.constant([1.0,2.0,3.0])
var = tf.cast(var,dtype=tf.float32)

c1 = tf.constant([0.5])
c2 = tf.constant([0.25])
c3 = tf.constant([0.125])




v1 = tf.ones_like(var)*c1

v2 = tf.ones_like(var)*c2

v3 = tf.ones_like(var)*c3


test = var[1]

i = tf.constant(0)

c = lambda i,v1: tf.less(i,2)

def body(i,v1):

    v1[i] = tf.case({tf.equal(v1[i],tf.constant(1.0)): lambda: tf.constant(3.0)},default=lambda:v1[i])

    return tf.add(i,1),v1

args = ()

r = tf.while_loop(c,body,(i,v1))



with tf.Session() as sess:

    v,v1,v2,v3,t = sess.run([var,v1,v2,v3,test])

    print(t)

    rr = sess.run(r)

    print(rr[1])

"""



