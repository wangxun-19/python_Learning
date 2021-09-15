import tensorflow as tf

print(tf.__version__)

# 创建一个常量

a = tf.constant([1,2],name='a')
b = tf.constant([1,2],name='b')

#两个向量相加

result = tf.add(a,b)

print(result)