install:
	pip install --upgrade pip
	pip install -r requirements.txt


train:
	python train_model.py

run:
	python model.py
