import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
##################################
#
# Nome: Matheus Santos Almeida
# Matricula: 201600092332
# E-mail: matheussa@dcomp.ufs.br
#
##################################

def imagem(file):
    v = imread(file) if type(file) is str else file
    return v


def v_l(res, n):
    return res if n > 1 else res[0]
# Q.2


def imread(filename):
    img = mpimg.imread(filename)
    if img.dtype == np.float32:
        return np.array([[list(map(lambda a: int(255*a), x)) for x in row] for row in img],
                        dtype=np.uint8)
    return img


# Q.3
def nchannels(file):
    img = imagem(file)
    if len(img.shape) == 2:
        return 1
    else:
        return img.shape[2]

# Q.4


def size(file):
    img = imagem(file)
    return img.shape[1], img.shape[0]

# Q.5


def rgb2gray(file):
    img = imagem(file)
    if nchannels(img) == 1:
        print("A imagem já está em escala de cinza")
        return img
    if img.shape[2] == 3:
        res = np.array([[int(0.299*x[0] + 0.587*x[1] + 0.114*x[2]) for x in row] for row in img], dtype=np.uint8)
        return res
    print("A imagem não está na escala RGB")

# Q.6


def imreadgray(file):
    img = imagem(file)
    if nchannels(img) == 1:
        return img
    else:
        return rgb2gray(img)

# Q.7


def imshow(file):
    img = imagem(file)
    if nchannels(img) == 1:
        plt.imshow(img, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()

# Q.8


def tresh(file, treshold):
    img = imagem(file)
    return (img > treshold)*255


# Q.9


def negative(file):
    img = imagem(file)
    return 255 - img

# Q.10


def contrast(file, r, m):
    img = imread(file)
    n = nchannels(file)
    return np.array([[v_l([min(max(r * (x[i] - m) + m, 0), 255) for i in range(n)]) for x in row] for row in img], dtype=np.uint8)

# Q.11


def hist(file):
    img = imagem(file)
    tv = np.zeros((256, nchannels(img)), dtype=np.uint32)
    if nchannels(img) == 1:
        arr = img.reshape(img.shape[0]*img.shape[1])
        unique, counts = np.unique(arr, return_counts=True)
        l = list(zip(unique, counts))
        for ls in l:
            tv[ls[0]][0] = ls[1]
    else:
        for row in img:
            for r in row:
                for i in range(3):
                    tv[r[i]][i] += 1
    return tv


def gera_bar(hs, bin, pos=0 , cor='blue'):
    res = [sum(hs[i:i + bin]) for i in range(0, 256, bin)]
    plt.bar((np.arange(res))+pos, res, width=0.25,align='center', color=cor)

# Q.12
# Q.13


def showhist(hs, bin=1):
    colors, somas = ['red', 'green', 'blue'], [0, 0.25, 0.5]
    res = np.array([[sum(hs[i:i + bin, j]) for i in range(0, 256, bin)] for j in range(hs.shape[1])])
    for i in range(hs.shape[1]):
        plt.bar((np.arange(res.shape[1])) + somas[i], res[0], width=0.25, align='center', color=colors[i])
    plt.show()

# Q.14


def histeq(file):
    img = imagem(file)
    hs = hist(img).reshape((1, 256))
    total = size(img)[1] * size(img)[0]
    tr = [sum(hs[0, :i+1])/float(total) for i in range(256)]
    return np.array([[tr[a]*255 for a in row] for row in img], dtype=np.uint8)

# Q.15


def f(img, x, y):
    return img[min(max(0, x), img.shape[0]-1), min(max(0, y), img.shape[1]-1)]


def convolve(file, mask):
    return er(file, mask, sum)

# Q.16


def maskblur():
    return 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1,2, 1]])


# Q.17


def blur(file):
    return convolve(file, maskblur())

# Q.18


def seSquare():
    return np.ones([3, 3], np.uint8)

# Q.19


def seCross3():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

# Q.20


def er_pix(img,x, y, S, w, a, b, n, func):
    rgb = np.array([f(img, x + s[0], y + s[1]) * w[s[0] + a, s[1] + b] for s in S]).reshape(S.shape[0],n)
    print(rgb.shape)
    return v_l([func(rgb[:, i]) for i in range(n)], n)


def er(file, mask, func):
    img = imagem(file)
    n = nchannels(img)
    a, b = np.array(mask.shape)//2
    S = np.array([(x, y) for x in range(-b, b + 1) for y in range(-a, a + 1) if mask[x + a][y + a] != 0])
    arr = [[er_pix(img, x, y, S, mask, a, b, n,func) for y in range(size(img)[0])] for x in range(size(img)[1])]
    return np.array(arr, np.uint8)


def erode(file, eb):
    return er(file, eb, np.amin)

# Q.21


def dilate(file, eb):
    return er(file, eb, np.amax)

