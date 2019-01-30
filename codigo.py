from collections import deque
def viz(x, y, adj):
	if adj == 4:
		return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
	if adj == 8:
		return [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x-1, y-1), (x+1,y+1), (x-1, y+1), (x+1, y-1)]

def v(img,x, y):
	if x >= img.shape[0] or x < 0 or y >= img.shape[1] or y < 0:
		return img[x,y]
	return -1

def rotulacao(img, adj):
	img = image(img).copy()
	rotulo = 1
	queue = deque([])
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			if img[x,y] == 255:
				img[x,y] = rotulo
				queue.append((x, y))
				while(queue.size()):
					cord = queue.dequeue()
					for vizinho in viz(cord[0], cord[1],adj):
						if v(img,vizinho[0], vizinho[1]) == 255:
							img[vizinho[0], vizinho[1]] = rotulo_atual
							queue.append(vizinho)
				rotulo += 1
	return img

def rotToRgb(img):
	max_rot = np.amax(img)
	for i in range(max_rot):
		img[img == i] = [img == i]*np.rand(255)



