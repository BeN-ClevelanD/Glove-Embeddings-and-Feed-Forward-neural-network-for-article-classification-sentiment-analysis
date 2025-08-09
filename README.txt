In order to run this code, a requirements.txt has been provided. It is suggested to use a python venv or a conda venv. 

These are the dependencies:
Python version: 3.12.11

packages:
optuna==4.4.0
numpy==2.2.5
scikit-learn==1.6.1
torch==2.7.0+cu126
pandas==2.2.3
matplotlib==3.10.3
umap-learn==0.5.7
seaborn==0.13.2

To create a venv for this:

python3.12 -m venv environment
source environment/bin/activate
pip install -r requrements.txt

running "python main.py" will run everything relevant. 
classifier_pipeline.py holds all code for the classifier. 
glove_utils.py holds all code for the glove implementation and training
old_normalise_classification_data.py implements text normalisation for the classifier task
optuna_tuna_classifier.py implements OPtuna hyperparameter tuning for the classifier
text_normaliser implements normalisation for the word embeddings

If there is already a glove_embeddings.txt present, the main.py will skip over retraining glove, and will just perform the visual analysis as well as
the training and testing of the classifier. to retrain the glove embeddings, simply remove this file. 

PLEASE MAKE SURE YOU HAVE THE FOLLOWING FOLDERS BEFORE RUNNING:
ag_news_csv
data/normalised/
results/

these should be one layer deeper than the base directly, with the folders being on the samelevel main.py for example.