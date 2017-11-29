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

images_path = config['flowers_images_path']
embedding_path = config['flowers_embedding_path']
text_path = config['flowers_text_path']
datasetDir = config['flowers_dataset_path']

val_classes = open(config['flowers_val_split_path']).read().splitlines()
train_classes = open(config['flowers_train_split_path']).read().splitlines()
test_classes = open(config['flowers_test_split_path']).read().splitlines()

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
	txt_path = os.path.join(text_path, _class)
	for example, txt_file in zip(sorted(glob(data_path + "/*.t7")), sorted(glob(txt_path + "/*.txt"))):
		example_data = load_lua(example)
		img_path = example_data['img']
		embeddings = example_data['txt'].numpy()
		example_name = img_path.split('/')[-1][:-4]

		f = open(txt_file, "r")
		txt = f.readlines()
		f.close()

		img_path = os.path.join(images_path, img_path)
		img = open(img_path, 'rb').read()

		txt_choice = np.random.choice(range(10), 5)

		embeddings = embeddings[txt_choice]
		txt = np.array(txt)
		txt = txt[txt_choice]
		dt = h5py.special_dtype(vlen=str)

		for c, e in enumerate(embeddings):
			ex = split.create_group(example_name + '_' + str(c))
			ex.create_dataset('name', data=example_name)
			ex.create_dataset('img', data=np.void(img))
			ex.create_dataset('embeddings', data=e)
			ex.create_dataset('class', data=_class)
			ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

		print(example_name, txt[1], _class)



