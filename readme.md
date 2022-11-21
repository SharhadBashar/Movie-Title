# Predicting the type of movie or tv show from its title

This code is a simple NLP classifier which predicts the type of movie or tv show from its title

The classes are:
1. Entertainment
2. News
3. Sports

### Setup:
1. Clone the repository:
`git clone https://github.com/SharhadBashar/Movie-Title.git`

2. Navigate to the python folder:
`cd python`

3. Install the required dependencies:
`pip install -r requirements.txt`

#### You are now ready to train a model and make predictions

### To run:
#### Train
To train a model from a data source, first make sure there is a file with a list of movie titles and corresponding classes in the format:

movie_title | class
------------|------

To train, you must call `main.py` and specify the `train` or `t` command, along with the path to the data:
```
python3 main.py train train.csv
```

This will take your data, clean it and store it in a file called `model_train.csv`. This file can be deleted after

It will then create a folder called `model` and the trained model will be stored in there as `model.pkl`

### To predict
To make predictions, first make sure there is a file with a list of movie titles:

movie_title |
------------|

To predict, you must call `main.py` and specify the `predict` or `p` command, along with the path to the data:
```
python3 main.py p predict.csv
```

This will take your inputs, clean it and store it in a file called `model_predict.csv`. This file is deleted after

It will then make the predictions and store them in a file called `predictions.csv`, which contains the original movie titles and the predicted classes:

movie_title | class
------------|-----
