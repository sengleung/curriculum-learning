cd ~
mkdir curriculum-learning/ConvNet/image_classification/datasets
mv emnist_balanced.zip curriculum-learning/ConvNet/image_classification/datasets/emnist_balanced.zip
sudo apt-get install python-virtualenv unzip -y
cd ~/curriculum-learning/ConvNet/image_classification/datasets
unzip emnist_balanced.zip
cd ~
virtualenv --python=python3.5 python3.5venv
