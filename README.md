# Car Purchase Decision: 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Installation](#installation)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Demo
Link: [https://car-purchase-decision.herokuapp.com/](htttps://flight-price-prediction-api.herokuapp.com/tps://car-purchase-decision.herokuapp.com/)

[![](https://i.imgur.com/R1g2wvC.png)](https://flight-price-prediction-api.herokuapp.com/)

[![](https://i.imgur.com/p0aeL6c.png)](https://flight-price-prediction-api.herokuapp.com/)

## Overview
This is a Flask web app which helps customer decision making while purchasing car

## Motivation
What to do when you are at home due to this pandemic situation? I started to learn Machine Learning model to get most out of it. I came to know mathematics behind all supervised models. Finally it is important to work on application (real world application) to actually make a difference.

## Installation
The Code is written in Python 3.8.6. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.

[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── static 
│   ├── css
│       ├── styles.css
├── templates
│   ├── home.html
|── cluster_files
|   ├── cluster_iter_imputer.pkl
|   ├── cluster_knn_imputer.pkl
|── model_files
|   ├── model_iter_imputer.pkl
|   ├── model_knn_imputer.pkl
|── modules
|   ├── load_clusters.py
|   ├── load_models.py
|   ├── transformation.py
├── Procfile
├── README.md
├── app.py
├── Car_Purchase_Decision.ipynb
├── requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) [<img target="_blank" src="https://mljar.com/images/machine-learning/xgboost_v2.png" width=200>](https://xgboost.readthedocs.io/en/latest/) [<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=200>](https://matplotlib.org/) [<img target="_blank" src="https://imbalanced-learn.org/stable/_static/logo_wide.png" width=200>](https://imbalanced-learn.org/stable/) 


## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Mandal-21/Flight-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope

* Use of Deep Learning
* Optimize Flask app.py
* Front-End 