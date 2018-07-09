from struct import unpack
import numpy as np


def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

# train_img_file = './data/mnist/train-images.idx3-ubyte'
# train_lbl_file = './data/mnist/train-labels.idx1-ubyte'
# test_img_file = './data/mnist/t10k-images.idx3-ubyte'
# test_lbl_file = './data/mnist/t10k-labels.idx1-ubyte'
#
# train_img, train_lbl = loadmnist(train_img_file, train_lbl_file)
# test_img, test_lbl = loadmnist(test_img_file, test_lbl_file)

# np.save('./data/mnist/train_img.npy', train_img)
# np.save('./data/mnist/train_lbl.npy', train_lbl)
# np.save('./data/mnist/test_img.npy', test_img)
# np.save('./data/mnist/test_lbl.npy', test_lbl)