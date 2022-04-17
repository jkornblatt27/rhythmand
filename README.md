# rhythmand
"rhythm and" is a web app that tells you the dominant emotion of an album and recommends five albums that are similar.

The potential emotions are: anger, joy, sadness, and surprise.

The app can be found here: http://rhythmand.herokuapp.com/

## Directory Structure

```
├── README.md                         <- You are here
├── src/                              <- Contains the code used for cleaning data, fitting models, and creating the database of albums
    ├── cleaning.py                   <- Cleaning the GoEmotions dataset
    ├── create_dataset.py             <- Populating the dataset with Genius/Spotify features
    ├── create_db.py                  <- Creating final database used for app
    ├── helpers.py                    <- Helper functions for model fitting
    ├── naive_bayes.py                <- Experiments in model fitting using Naive Bayes models
    ├── svm.py                        <- Experiments in model fitting using linear support vector classifiers 
├── templates/                        <- HTML used in web app
    ├── home.html                     <- Homepage
    ├── results.html                  <- Displays results of homepage query
│
├── app.py                            <- Flask wrapper for querying app database
├── Procfile                          <- Commands for executing app on Heroku
├── requirements.txt                  <- Python package dependencies
├── runtime.txt                       <- Specifies Python version for running app on Heroku
```

## Data Used

The data used for model fitting is the GoEmotions dataset, which can be found here: https://github.com/google-research/google-research/tree/master/goemotions

The albums that are included in the app database are taken from a dataset of albums that were reviewed by Pitchfork, found here: https://components.one/datasets/pitchfork-reviews-dataset

## Sources

Thompson, A. (2019). *20,783 Pitchfork Reviews* [Data set]. https://components.one/datasets/pitchfork-reviews-dataset

Demszky, Dorottya κ.ά. ‘GoEmotions: A Dataset of Fine-Grained Emotions’. *58th Annual Meeting of the Association for Computational Linguistics (ACL)*. N.p., 2020. Print.

Paul Ekman. 1992b. An argument for basic emotions. *Cognition & Emotion*, 6(3-4):169–200.

