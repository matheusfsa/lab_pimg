import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dog = 'images/dog.jpg'

def imagem(file):
    if type(file) is str:
        return imread(file)
    else:
        return file


'''
2- Crie uma função chamada imread que recebe um nome de arquivo e retorna a imagem lida
'''


def imread(filename):
    img =  mpimg.imread(filename)
    if img.dtype == np.float32:
        return np.array([[list(map(lambda a: int(255*a), x)) for x in row] for row in img],
                        dtype=np.uint8)
    return img


'''
3- Crie uma função chamada nchannels que retorna o número de canais da imagem de entrada
'''


def nchannels(file):
    img = imagem(file)
    if len(img.shape) == 2:
        return 1
    else:
        return img.shape[2]


"""
4  -  Crie  uma  função  chamada  size  que  retorna  um  vetor  onde  a  primeira  posição  é  a  largurae  a  segunda 
é  a  altura  em  pixels  da  imagem  de  entrada.
"""


def size(file):
    img = imagem(file)
    return img.shape[1], img.shape[0]


"""
5  -  Crie  uma  função  chamada  rgb2gray  que  recebe  uma  imagem  RGB  e  retorna  outra  imagem  igual  à  imagem  
de  entrada  convertida  para  escala  de  cinza.  Para  converter  um  pixel  de  RGB  para  escala  de  cinza,  faça  
a  média  ponderada  dos  valores  (R,  G,  B)  com  os  pesos  (0.299,  0.587,  0.114)  respectivamente.
"""


def rgb2gray(file):
    img = imagem(file)
    if nchannels(img) == 1:
        print("A imagem já está em escala de cinza")
        return img
    if img.shape[2] == 3:
        res = np.array([[int(0.299*x[0] + 0.587*x[1] + 0.114*x[2]) for x in row] for row in img], dtype=np.uint8)
        return res
    print("A imagem não está na escala RGB")


"""
6  -  Crie  uma  função  chamada  imreadgray  que  recebe  um  nome  de  arquivo  e  retorna  a  imagem  lida  em  
escala  de  cinza.  Deve  funcionar  com  imagens  de  entrada  RGB  e  escala  de  cinza.
"""


def imreadgray(file):
    img = imagem(file)
    if nchannels(img) == 1:
        return img
    else:
        return rgb2gray(img)

"""
7  -  Crie  uma  função  chamada  imshow  que  recebe  uma  imagem  como  parâmetro  e  a  exibe.  Se  a  imagem  for  
em  escala  de  cinza,  exiba  com  colormap  gray.  Sempre  usar  interpolação  nearest  para  que  os  pixels  
apareçam  como  quadrados  ao  dar  zoom  ou  exibir  imagens  pequenas.
"""


def imshow(file):
    img = imagem(file)
    if nchannels(img) == 1:
        plt.imshow(img, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()


'''
8  -  Crie  uma  função  chamada  thresh  que  recebe  uma  imagem  e  um  valor  de  limiar.  Retornauma  imagem  onde  
cada  pixel  tem  intensidade  máxima  onde  o  pixel  correspondemte  da  imagem  de  entrada  tiver  intensidade  maior
ou  igual  ao  limiar,  e  intensidade  mínima  caso  contrário.
'''


def tresh(file, treshold):
    img = imagem(file)
    if nchannels(img) == 1:
        return np.array([list(map(lambda a: 255 if a >= treshold else 0, row.s)) for row in img],
                        dtype=np.uint8)
    return np.array([[list(map(lambda a: 255 if a >= treshold else 0, x)) for x in row] for row in img],
                    dtype=np.uint8)


'''
9  -  Crie  uma  função  chamada  negative  que  recebe  uma  imagem  e  retorna  sua  negativa.
'''


def negative(file):
    img = imagem(file)
    return np.array([[list(map(lambda a: 255 - a, x)) for x in row] for row in img],
                    dtype=np.uint8)


'''
10 - Crie uma função chamada contrast que recebe uma imagem f, real r e um real m.
Retorna uma imagem g = r(f - m) + m
'''
def g(r, a, m, n):
    if n == 3:
        res = []
        for i in range(3):
            v = int(r*(a[i]-m) + m)
            res.append(min(max(v,0), 255))
        return res
    else:
        v = int(r * (a - m) + m)
        return min(max(v, 0), 255)

def contrast(file, r, m):
    img = imagem(file)
    n = nchannels(file)
    return np.array([[g(r,x,m,n) for x in row] for row in img],
                    dtype=np.uint8)


'''
11 - Crie uma função chamada hist que retorna uma matriz coluna onde cada posição
contém o número de pixels com cada intensidade de cinza. Caso a imagem seja RGB,
retorne uma matriz com 3 colunas.
'''


def hist(file):
    img = imagem(file)
    if nchannels(img) == 1:
        arr = img.reshape(img.shape[0]*img.shape[1])
        i = 0
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


'''
12 - Crie uma função chamada showhist que recebe a saída da função anterior e mostra
um gráfico de barras. Caso a matriz recebida tenha três colunas, ou seja, se referente a uma
imagem RGB, desenhe para cada intensidade uma barra com cada uma das três cores.

13 - Altere a função anterior, adicionando um segundo parâmetro opcional chamado bin.
Seu valor padrão deve ser 1, o tipo é inteiro e serve para agrupar os itens do vetor recebido
no primeiro parâmetro. Ou seja, se bin = 5, cada barra corresponderá a um grupo de 5
intensidades consecutivas.
'''


def gera_bar(hs, bin, cor='blue'):

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

    plt.bar(np.arange(n), res, width=1, color=cor)


def showhist(file, bin=1):
    img = imagem(file)
    if nchannels(img) == 1:
        hs = hist(img).reshape(256)
        gera_bar(hs, bin)
    else:
        hs = hist(img)
        gera_bar(hs[:, [0]], bin, cor='red')
        gera_bar(hs[:, [1]], bin, cor='green')
        gera_bar(hs[:, [2]], bin, cor='blue')
    plt.show()


'''
14 - Crie uma função chamada histeq que calcula a equalização do histograma da imagem
de entrada e retorna a imagem resultante. Deve funcionar para imagens em escala de cinza.
'''


def histeq(file):
    img = imagem(file)
    if nchannels(img) == 1:
        hs = hist(img).reshape((1, 256))
        total = size(img)[1] * size(img)[0]
        tr = [sum(hs[0, :i+1])/float(total) for i in range(256)]
        return tr
        return np.array([[tr[a]*255 for a in row] for row in img], dtype=np.uint8)
    else:
        hs = hist(img)
        total = size(img)[1] * size(img)[0]
        tr = np.array([[sum(hs[:i + 1, j]) / float(total) for i in range(256)] for j in range(3)])
        return np.array([[[tr[0, a[0]] * 255, tr[1, a[1]] * 255,tr[2, a[2]] * 255]for a in row] for row in img], dtype=np.uint8)


'''
15 - Crie uma função chamada convolve, que recebe uma imagem de entrada e uma
máscara com valores reais. Retorna a convolução da imagem de entrada pela máscara.
Nesta e nas próxomas questões, quando necessário extrapolar, use o valor do pixel mais
próximo pertencente à borda.
'''


def f(img, x, y):
    return img[min(max(0, x), img.shape[0]-1), min(max(0, y), img.shape[1]-1)]


def conv_pix(img, x, y, w, a, b):
    pix = 0
    for s in range(-a, a+1):
        for t in range(-b, b+1):
            pix += w[s+a, t+b]*f(img, x+s, y+t)
    return pix


def convolve(file, mask):
    img = imagem(file)
    mask = np.array(mask)
    n, m = mask.shape
    a = int((n-1)/2)
    b = int((m-1)/2)
    return np.array([[conv_pix(img,x, y, mask, a, b) for y in range(size(img)[0])] for x in range(size(img)[1])],dtype=np.uint8)


def clamp(value, L):
	return min(max(value,0), L-1)


def convolve2(file, mask):
    image = imagem(file)
    convolution = np.ndarray(image.shape, dtype='uint8')
    a = int((mask.shape[0]-1)/2)
    b = int((mask.shape[1]-1)/2)
    altura = image.shape[0]
    largura = image.shape[1]
    for x in range(altura):
        for y in range(largura):
            if nchannels(image) == 1:
                soma = 0
            else:
                soma = [0,0,0]
            for s in range(-a,a+1):
                for t in range(-b,b+1):
                    w = mask[s+1,t+1]
                    f = image[clamp(x+s,altura),clamp(y+t,largura)]
                    soma += w * f
            convolution[x,y] = soma
    return convolution

'''
16 - Crie uma função chamada maskBlur que retorna a máscara 1/16 * [[1, 2, 1], [2, 4, 2], [1,
2, 1]].
'''


def maskblur():
    return 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1,2, 1]])


'''
17 - Crie uma função chamada blur, que convolve a imagem de entrada pela máscara
retornada pela função maskBlur.
'''


def blur(file):
    return convolve(file, maskblur())


'''
18 - Crie uma função chamada seSquare3, que retorna o elemento estruturante binário [[1,1, 1], [1, 1, 1], [1, 1, 1]].
'''


def seSquare():
    return np.ones([3,3],np.uint8)


'''
19 - Crie uma função chamada seCross3, que retorna o elemento estruturante binário [[0, 1,0], [1, 1, 1], [0, 1, 0]].
'''


def seCross3():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


'''
20 - Crie uma função chamada erode, que recebe uma imagem e um elemento estruturante
binário. Retorna uma imagem onde cada pixel (i, j) da saída é igual ao menor valor presente
no conjunto de pixels definido pelo elemento estruturante centrado no pixel (i, j) da entrada.
São considerados apenas os pixels correspondentes a posições diferentes de zero no
elemento estruturante.
'''


def v(f, m):
    if m == 0:
        if type(f) == np.ndarray:
            return np.array([-1, -1, -1])
        else:
            return None
    return f


def eb_pix(img, mask, x, y, a, b, op):
    '''
    xi = max(0, x-a)
    xf = min(x + a + 1, img.shape[0])
    yi = max(0, y - b)
    yf = min(y + b + 1, img.shape[0])
    a = np.multiply(img[xi:xf, yi:yf], mask)
    '''
    if nchannels(img) == 1:
        if op == 0:
            res  = np.Inf
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    if mask[s + a, t + b] == 1:
                        v = f(img, x + s, y + t)
                        res = min(res,v)
        else:
            res = 0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    if  mask[s + a, t + b] == 1:
                        v = f(img, x + s, y + t)
                        res = max(res,v)
    else:
        if op == 0:
            res = np.array([255, 255, 255], np.uint8)
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    if  mask[s + a, t + b] == 1:
                        v = f(img, x + s, y + t)
                        res[0] = min(res[0], v[0])
                        res[1] = min(res[1], v[1])
                        res[2] = min(res[2], v[2])
        else:
            res = np.zeros(3,dtype=np.uint8)
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    if mask[s + a, t + b] == 1:
                        v = f(img, x + s, y + t)
                        res[0] = max(res[0], v[0])
                        res[1] = max(res[1], v[1])
                        res[2] = max(res[2], v[2])
    return res


def erode(file, eb):
    img = imagem(file)
    n, m = eb.shape
    a = int((n - 1) / 2)
    b = int((m - 1) / 2)
    arr = [[eb_pix(img, eb, x, y, a, b, 0) for y in range(size(img)[0])] for x in range(size(img)[1])]
    return np.array(arr)




'''
21 - Crie uma função chamada dilate, semelhande à erode da questão anterior, retornando
porém o maior valor no lugar do menor.
'''


def dilate(file, eb):
    img = imagem(file)
    n, m = eb.shape
    a = int((n - 1) / 2)
    b = int((m - 1) / 2)
    arr = [[eb_pix(img, eb, x, y, a, b, 1) for y in range(size(img)[0])] for x in range(size(img)[1])]
    return np.array(arr)

