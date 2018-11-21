import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras
from keras.preprocessing import image
from keras.models import load_model
from net import Net
import sys 

# load image and convert it to a tensor
def load_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    print("Display the current image")
    plt.imshow(img)                           
    plt.axis('off')
    plt.show()
    img_tensor = image.img_to_array(img)                
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    return img_tensor

def predict(weights_path, image_path):
    model = Net.build(32, 32, 3, weights_path)
    image = load_image(image_path) 
    class_ = model.predict(image)
    output_indice = -1 
    
    # get class index having maximum predicted score
    for i in range(36):
        if(i == 0):
            max = class_[0][i]
            output_indice = 0
        else:
            if(class_[0][i] > max):
                max = class_[0][i]
                output_indice = i
    
    # append 26 characters (A to Z) to list characters
    characters = []
    for i in range(65, 65+26):
        characters.append(chr(i))
    if(output_indice > 9):
        final_result = characters[(output_indice - 9) - 1]
        print("Predicted: ", final_result)

    else:
        print("No result can be fulfilled")
