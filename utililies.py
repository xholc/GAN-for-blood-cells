import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

##### bit planes
def get_bit_plane(image, bit):
    # Extract the bit plane
    bit_plane = np.bitwise_and(image, 2**bit)
    # Scale it to 0 or 255
    bit_plane = np.array(bit_plane * 255 / 2**bit).astype(np.uint8)

    return bit_plane

def get_transform_image(image, rb, gb, bb):
    img = cv2.imread(image)
    # Split the image into BGR channels
    b, g, r = cv2.split(img)
    bit_plane_r = get_bit_plane(r, rb) #4
    kr = (bit_plane_r+1)* (2**rb)-1
    bit_plane_g = get_bit_plane(g, gb) #4
    kg = (bit_plane_g+1)* (2**gb)-1
    bit_plane_b = get_bit_plane(b, bb) #5
    kb = (bit_plane_b+1)* (2**bb)-1
    result_fig = np.stack((kr, kg, kb), axis=-1)

    return result_fig


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

#test function
'''
for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)  '''
    
    