import os
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
from torch.utils.serialization import load_lua  
from PIL import Image 
import yaml
import io
import pdb

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

images_path = config['birds_images_path']
embedding_path = config['birds_embedding_path']
datasetDir = config['birds_dataset_path']

val_classes = open(config['val_split_path']).read().splitlines()
train_classes = open(config['train_split_path']).read().splitlines()
test_classes = open(config['test_split_path']).read().splitlines()

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

for _class in sorted(os.listdir(embedding_path)):
	split = ''
	if _class in train_classes:
		split = train
	elif _class in val_classes:
		split = valid
	elif _class in test_classes:
		split = test

	data_path = os.path.join(embedding_path, _class)
	for example in sorted(glob(data_path + "/*.t7")):
		example_data = load_lua(example)
		img_path = example_data['img']
		embeddings = example_data['txt'].numpy()
		example_name = img_path.split('/')[-1][:-4]

		img_path = os.path.join(images_path, img_path)
		img = open(img_path, 'rb').read()

		embeddings = embeddings[np.random.choice(range(10), 5)]

		for c, e in enumerate(embeddings):
			ex = split.create_group(example_name + '_' + str(c))
			ex.create_dataset('name', data=example_name)
			ex.create_dataset('img', data=np.void(img))
			ex.create_dataset('embeddings', data=e)
			ex.create_dataset('class', data=_class)

		print(example_name)



