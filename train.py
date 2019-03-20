import scipy.io
coords = scipy.io.loadmat("train_data/male/image_001.mat")
print(coords["x"])
print(coords["y"])
