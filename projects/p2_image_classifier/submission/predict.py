import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import argparse

image_size = 224
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def where(needles,haystack):
    classes = []
    for i in needles:
        found = np.where(i == haystack)
        classes.append(str(found[0][0]+1))
    
    return classes
        
from PIL import Image
def predict(image_path,model,top_k=5):
    #process image
    im = Image.open(image_path)
    image = np.asarray(im)
    
    processed_image = process_image(image)
    #print(f"Shape before: {processed_image.shape}")
    processed_image = np.expand_dims(processed_image,axis=0)
    #print(f"Shape after: {processed_image.shape}")
    ps = model.predict(processed_image)

    predictions = ps[0]
    #top_k probalities
    probs = np.sort(predictions)[::-1][:top_k]

    #top_k classes index
    classes = where(probs,predictions)
        
    return probs,classes

#args
parser = argparse.ArgumentParser()
parser.add_argument("image_path",type=str,help="image path")
parser.add_argument("model",type=str,help="model name to load & use")
parser.add_argument('--top_k', action="store", dest="top_k",type=int,help="top (n) matched",default=5)
parser.add_argument('--category_names', action="store",dest="category_names",help="category name in JSON")
args = parser.parse_args()

reloaded_keras_model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer': hub.KerasLayer},compile=False)

#the prediction
probs, classes = predict(args.image_path, reloaded_keras_model, args.top_k)

print(f"Top {args.top_k} probabilities: {probs}")

if args.category_names:
    #args contains --category_names
    import json
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        
    classes_matched = [class_names[i] for i in classes]
    print(f"its classes: {classes_matched}")
else:
    #no --category_names in args
    print(f"its classes: {classes}")
    
#e.g. calling
#python predict.py test_images/cautleya_spicata.jpg my_model_1613927278.h5 --category_names label_map.json
    
    
