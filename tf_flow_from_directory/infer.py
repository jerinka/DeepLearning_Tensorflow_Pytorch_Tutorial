import numpy as np
import keras.utils as image
import tensorflow as tf

test_image = image.load_img('train/dog/dog.100.jpg')
test_image = tf.image.resize(test_image, [64,64])
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

cnn = tf.keras.models.load_model('model.h5')

result = cnn.predict(test_image)

# {'cats': 0, 'dogs': 1}

if result[0][0] == 1:
	print('dog')
else:
	print('cat')
