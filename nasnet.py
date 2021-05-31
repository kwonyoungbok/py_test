import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import time

if __name__ == '__main__':
    model = tf.keras.applications.NASNetLarge(
                        input_shape=None,
                        include_top=True,
                        weights="imagenet",
                        input_tensor=None,
                        pooling=None,
                        classes=1000)


    for i in range(5):
        img_path = str(i+1)+'.jpg'
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        st= time.time()
        preds = model.predict(x)
        print("inf",i,": ", time.time()-st)

    print('Predicted:', decode_predictions(preds))
