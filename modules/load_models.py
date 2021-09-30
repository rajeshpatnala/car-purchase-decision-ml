import os
import pickle

parent_dir = os.path.normpath(os.getcwd())
models_dir = parent_dir + os.path.sep + 'model_files'

mdl_knn = pickle.load(open(models_dir + os.sep + "model_iter_imputer.pkl", "rb"))
mdl_iter = pickle.load(open(models_dir + os.sep + "model_knn_imputer.pkl", "rb"))
