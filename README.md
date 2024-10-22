# NYC_TAXI_PROJECT
## Motivation:
This repository is made for the fourth homework of the course Tools for reproducible data science (STOR 674) in the semester of  Fall 2024, the purpose of this homework is to learn
encapsulation techniques, pytorch and get some practice with version control in github.

## Description;
In this project we get the data from taxi drives from NYC, to exemplify the usage of pytorch a model is created were the fareamout is predicted from the variables of pick-up and drop-off, further improvement in the model can be done.

## Basic components:
- data: This folder contain the data used in the example. We use the NYC dataset up to the point of January of 2024. The dataset can be downloaded [here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- models: This folder contains the models fitted from the train_model.py file, the models are saved in .pth extension.
- Notebooks: This folder may be empty, this is inteded to store some notebooks we used for exploration of the dataset.
- scripts: This folder contains two scripts, utils.py and model_utils.py. In the first one we do the data cleaning and preprocessing and in the later we define the neural network architecture.
- train_model.py: This is the script were all the action is happening. This is script trains the model using the model defined in scripts/model_utils.py
- requirements.txt: This is the file of requirements of python packages to run this repository.

## How to use it:
- Download this repository and keep it with the same structure as shown in here.
- Install all dependencies required: To do this you can do 
```console
conda create --name myenv python=3.8

conda activate myenv

pip install -r requirements.txt
```
- Run the code: For example, to run the code with parameters of learning rate 0.01 number of  epochs 1 and batch size 32 you can use the terminal of your preference with the folowing command.

```console
python train_model.py --learning_rate 0.01 --num_epochs 1 --batch_size 32
```

The learn 
