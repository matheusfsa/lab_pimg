import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def imagem(file):
    v = imread(file) if type(file) is str else file
    return v

# Q2


def imread(filename):
    img =  mpimg.imread(filename)
    if img.dtype == np.float32:
        return np.array([[list(map(lambda a: int(255*a), x)) for x in row] for row in img],
                        dtype=np.uint8)
    return img


# Q3
def nchannels(file):
    img = imagem(file)
    if len(img.shape) == 2:
        return 1
    else:
        return img.shape[2]

# Q4
def size(file):
    img = imagem(file)
    return img.shape[1], img.shape[0]

# Q5
def rgb2gray(file):
    img = imagem(file)
    if nchannels(img) == 1:
        print("A imagem já está em escala de cinza")
        return img
    if img.shape[2] == 3:
        res = np.array([[int(0.299*x[0] + 0.587*x[1] + 0.114*x[2]) for x in row] for row in img], dtype=np.uint8)
        return res
    print("A imagem não está na escala RGB")

# Q6


def imreadgray(file):
    img = imagem(file)
    if nchannels(img) == 1:
        return img
    else:
        return rgb2gray(img)

# Q7


def imshow(file):
    img = imagem(file)
    if nchannels(img) == 1:
        plt.imshow(img, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()

# Q8


def tresh(file, treshold):
    img = imagem(file)
    return (img > treshold)*255


# Q9
def negative(file):
    img = imagem(file)
    return 255 - img

# Q10


def contrast(file, r, m):
    img = imagem(file)
    n = nchannels(file)

    def g(a):
        if n == 3:
            res = []
            for i in range(3):
                v = r * (a[i] - m) + m
                res.append(min(max(v, 0), 255))
            return res
        else:
            v = r * (a - m) + m
            return min(max(v, 0), 255)
    v = np.array([[g(x) for x in row] for row in img], dtype=np.uint8)
    return v

# Q11


def hist(file):
    img = imagem(file)
    if nchannels(img) == 1:
        arr = img.reshape(img.shape[0]*img.shape[1])
        unique, counts = np.unique(arr, return_counts=True)
        l = list(zip(unique, counts))
        tv = np.zeros((256, 1), dtype=np.uint32)
        for ls in l:
            tv[ls[0]][0] = ls[1]

    else:
        tv = np.zeros((256, 3), dtype=np.uint32)
        for row in img:
            for r in row:
                tv[r[0]][0] += 1
                tv[r[1]][1] += 1
                tv[r[2]][2] += 1
    return tv


def gera_bar(hs, bin,pos=0 , cor='blue'):

    lst_v = 0
    n = int(256/bin)
    res = np.zeros(n)
    for i in range(n):
        if i + bin <= 256:
            res[i] = sum(hs[lst_v:lst_v + bin])
            lst_v += bin
        else:
            res[i] = sum(hs[lst_v:256])
            lst_v = 255

    plt.bar(np.arange(n)+pos, res, width=0.25,align='center', color=cor)
# Q12
# Q13


def showhist(file, bin=1):
    img = imagem(file)
    if nchannels(img) == 1:
        hs = hist(img).reshape(256)
        gera_bar(hs, bin)
    else:
        hs = hist(img)
        gera_bar(hs[:, 0], bin,pos=0, cor='red')
        gera_bar(hs[:, 1], bin, pos=0.25, cor='green')
        gera_bar(hs[:, 2], bin, pos=0.5, cor='blue')
    plt.show()

# Q14


def histeq(file):
    img = imagem(file)
    hs = hist(img).reshape((1, 256))
    total = size(img)[1] * size(img)[0]
    tr = [sum(hs[0, :i+1])/float(total) for i in range(256)]
    return np.array([[tr[a]*255 for a in row] for row in img], dtype=np.uint8)

# Q15


def f(img, x, y):
    return img[min(max(0, x), img.shape[0]-1), min(max(0, y), img.shape[1]-1)]


def convolve(file, mask):
    return er(file, mask, sum)

# Q16


def maskblur():
    return 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1,2, 1]])


# Q17


def blur(file):
    return convolve(file, maskblur())

# Q18


def seSquare():
    return np.ones([3,3],np.uint8)

# Q19


def seCross3():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

# Q20


def er_pix(img,x, y, S, w,a,b,func):
    if nchannels(img) == 1:
        return func([f(img, x+s[0], y+s[1])*w[s[0]+a, s[1]+b] for s in S])
    else:
        rgb = np.array([f(img, x + s[0], y + s[1]) * w[s[0] + a, s[1] + b] for s in S])
        return [func(rgb[:,0]), func(rgb[:,1]), func(rgb[:,2])]


def er(file, mask, func):
    img = imagem(file)
    n, m = mask.shape
    a = int((n - 1) / 2)
    b = int((m - 1) / 2)
    S = [(x, y) for x in range(-b, b + 1) for y in range(-a, a + 1) if mask[x + a][y + a] != 0]
    arr = [[er_pix(img, x, y, S, mask, a, b, func) for y in range(size(img)[0])] for x in range(size(img)[1])]
    return np.array(arr, np.uint8)


def erode(file, eb):
    return er(file, eb, np.amin)

# Q21


def dilate(file, eb):
    return er(file, eb, np.amax)

