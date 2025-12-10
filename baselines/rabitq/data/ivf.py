
import numpy as np
import faiss
import struct
import os
import sys
from utils.io import *

# source = './'

if __name__ == '__main__':

    # dataset = 'siftsmall'
    # source = sys.argv[1]
    # dataset = sys.argv[2];
    # print(f"Clustering - {dataset}")
    # path
    # path = os.path.join(source, dataset)
    data_path   = sys.argv[1]
    dim         = int(sys.argv[2])
    store_path  = sys.argv[3]
    dataset     = sys.argv[4]
    K           = sys.argv[5]

    X = read_fvecs(data_path, dim)
    D = X.shape[1]
    # K = 4096
    centroids_path = os.path.join(store_path, f'{dataset}_centroid_{K}.fvecs')
    dist_to_centroid_path = os.path.join(store_path, f'{dataset}_dist_to_centroid_{K}.fvecs')
    cluster_id_path = os.path.join(store_path, f'{dataset}_cluster_id_{K}.ivecs')

    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid ** 0.5

    to_fvecs(dist_to_centroid_path, dist_to_centroid)
    to_ivecs(cluster_id_path, cluster_id)
    to_fvecs(centroids_path, centroids)
