from collections import deque
import numpy as np
import pimg as p

def viz(x, y, adj):
    if adj == 4:
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    if adj == 8:
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x-1, y-1), (x+1,y+1), (x-1, y+1), (x+1, y-1)]


def v(img,x, y):
    if x >= img.shape[0] or x < 0 or y >= img.shape[1] or y < 0:
        return -1
    return img[x, y]


def rotulacao(img, adj):
    img = img.copy()
    rotulo = 1
    queue = deque([])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] == 255:
                queue.append((x, y))
                img[x,y] = rotulo
                while queue:
                    cord = queue.popleft()
                    for vizinho in viz(cord[0], cord[1],adj):
                        if v(img,vizinho[0], vizinho[1]) == 255:
                            img[vizinho[0], vizinho[1]] = rotulo
                            queue.append(vizinho)
                rotulo += 1
    return np.array(img, np.uint8)

def rotToRgb(img):
    res = np.array([[[(pix*59)%255, (pix*73)%255, (pix*83)%255] for pix in row] for row in img], np.uint8)
    return res

def gera_img(tresh, adj):
    img_g = p.imreadgray('images/mario.jpg')
    img_t = p.tresh(img_g, tresh)

    img = rotulacao(img_t, adj)
    return rotToRgb(img)


def grad_morf(img, b, n):
    if n == 1:
        return img - p.erode(img, b)
    elif n == 2:
        return p.dilate(img, b) - img
    else:
        return p.dilate(img, b) - p.erode(img, b)


def dil_cond(img, b, m):
    return np.multiply(p.dilate(img, b), m)


def equal(a, b):
    return np.array_equal(a, b)

def create_xk()

def comp_ext(img, )