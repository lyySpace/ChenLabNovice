import tensorflow as tf

'''
1. Learn more about the attribution of tf.reshape, and 
try to reshape a Tensor with shape of (3,2,5) into shape of (3,10); 
see the change of value and attributions (e.g., rank) of the Tensor.
'''

original = tf.zeros(shape=(3, 2, 5))
reshaped = tf.reshape(original, (3,10))

print("Original tensor")
print(f"Shape: {original.shape}, Rank: {len(original.shape)}\n")

print("Reshaped Tensor:")
print(f"Shape: {reshaped.shape}, Rank: {len(reshaped.shape)}")


'''
2. Implement a function that may show the shape of input Tensor, 
and also return the square of the Tensor. Please use the tf.function.
'''

@tf.function #  Decorator
def tensor_square(x):
    tf.print("Input Tensor Shape", tf.shape(x))
    return tf.square(x)

example = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
result = tensor_square(example)

'''
3. Illustrate a Graph for your function
'''
logdir = "logs" # 指定日誌除存路徑 # tensorboard --logdir=logs
writer = tf.summary.create_file_writer(logdir)

with writer.as_default():
    tf.summary.graph(tensor_square.get_concrete_function(example).graph)

