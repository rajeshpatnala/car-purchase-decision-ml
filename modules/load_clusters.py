import os
import pickle

parent_dir = os.path.normpath(os.getcwd())
cluster_dir = parent_dir + os.path.sep + 'cluster_files'

cls_knn = pickle.load(open(cluster_dir + os.sep + "cluster_iter_imputer.pkl", "rb"))
cls_iter = pickle.load(open(cluster_dir + os.sep + "cluster_knn_imputer.pkl", "rb"))
