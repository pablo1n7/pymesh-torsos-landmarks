# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import decomposition

def aling_shapes(ls, m_ls):
    ls = scale(ls)
    m_ls = scale(m_ls)
    assert len(ls) == len(m_ls)
    N = ls.shape[0]
    centroid_ls = get_centroid(ls)
    centroid_m_ls = get_centroid(m_ls)
     # centre the points
    AA = ls - np.tile(centroid_ls, (N, 1))
    BB = m_ls - np.tile(centroid_m_ls, (N, 1))
     # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_ls.T + centroid_m_ls.T

    A2 = (R*ls.T) + np.tile(t, (1, N))
    A2 = A2.T

    return np.array(A2)


def get_centroid_size(landmarks):
    """ Obtienee el centroid size de una configuración de landmarks
    (Raiz cuadrada de la suma de las distancias al cuadrado)"""
    centroid = get_centroid(landmarks)
    diff = landmarks-centroid
    diff_cuadrado = (diff**2.).sum()
    return np.sqrt(diff_cuadrado)

def scale(landmarks):
    print("")
    landmarks = np.array(landmarks)
    return np.matrix(landmarks / get_centroid_size(landmarks))


def get_centroid(landmarks):
    """ Obtiene el centroide de una configuración de landmarks"""
    return landmarks.mean(0)

def plot_landmarks2d(landmarks):
    """ Plot de los landmarks con su centroide"""
    fig = plt.figure()
    plt.axes().set_aspect('equal', 'datalim')
    axe = fig.add_subplot(111)
    for lands in landmarks:
        #centroid = get_centroid(lands)
        axe.scatter(lands[:, 0], lands[:, 1], c="blue", marker="*")
        #axe.scatter(centroid[0], centroid[1], centroid[2], c="red", marker="o")
    plt.show()

def plot_landmarks3d(landmarks):
    """ Plot de los landmarks con su centroide"""
    fig = plt.figure()
    #plt.axes().set_aspect('equal', 'datalim')
    axe = fig.add_subplot(111, projection='3d')
    for lands in landmarks:
   #     centroid = get_centroid(lands)
        lands = np.array(lands)
        axe.scatter(lands[:,0], lands[:, 1], lands[:, 2], c="red", marker="*")
    #    axe.scatter(centroid[0], centroid[1], centroid[2], c="red", marker="o")
    plt.show()

def consensus(landmarks):
    """ """
    return landmarks.mean(0)

def procrustes(landmarks):
    """ """
    ls_consensus = np.matrix(consensus(landmarks))
    #ls_consensus = np.matrix(landmarks[0])
    procrustres_array = map(lambda x: aling_shapes(np.matrix(x),ls_consensus),landmarks)
    #plot_landmarks3d([landmarks[0],landmarks[1],ls_consensus])
    return procrustres_array


def main():
    """ Funcion Principal """
    import pandas as pd

    landmarking = pd.read_csv("../data_artificial_cites/landmarking_automatico_f&m.csv",index_col=0)
    ids = landmarking["id"].values
    header = []
    for i in range(15):
        header.append("S0{}{}_X".format(i//10,i%10))
        header.append("S0{}{}_Y".format(i//10,i%10))
        header.append("S0{}{}_Z".format(i//10,i%10))
    
    landmarks = landmarking[header]
    landmarks_array = np.ones(landmarks.shape)

    #aligns = procrustes(np.array([A,B]))
    #plot_landmarks3d(aligns)

    for i in range(0,landmarks.shape[0]):
        landmarks_array[i] = landmarks.iloc[[i]].values
    
    landmarks_array = landmarks_array.reshape(300,15,3)
    landmarks_procrustes = np.array(list(procrustes(landmarks_array)))

    # ret_R, ret_t = aling_shapes(A, B)
    # A2 = (ret_R*A.T) + np.tile(ret_t, (1, 15))
    # A2 = A2.T

    #A2 = aling_shapes(A, B)
    plot_landmarks3d(landmarks_procrustes)
    #plot_landmarks3d([A2,B])

    dim = 3 * 15 - 7
    pca = decomposition.PCA(n_components=dim,svd_solver='full')
    pca.fit(landmarks_procrustes.reshape(300,45))
    print ("Con n_components = {} el valor es: {}".format(dim,pca.explained_variance_ratio_.sum()))

    x_out = pca.transform(landmarks_procrustes.reshape(300,45))
    x_out.shape

    fig = plt.figure()

    plt.scatter(x_out[0:150,0],x_out[0:150,1],c="pink",marker="o")
    plt.scatter(x_out[150:300,0],x_out[150:300,1],c="blue",marker="o")
    plt.show()


if __name__ == '__main__':
    main()

