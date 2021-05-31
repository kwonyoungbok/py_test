import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

if __name__ == '__main__':
    model = InceptionV3(include_top=True, weights='imagenet')

    img_path = 'elephant.jpeg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))