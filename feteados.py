import pymesh
import numpy as np
import pandas as pd
import math
import glob

fs_landmarks = sorted(glob.glob("../../Datos/data_artificial_cites/landmarks_reales/*.pts"))
fs_mesh = sorted(glob.glob("../../Datos/data_artificial_cites/reales/*.ply"))

for landmarks_f, mesh_f in zip(fs_landmarks[94:], fs_mesh[94:]):
    landmarks = pd.read_csv(landmarks_f,header=None,index_col=[0],skiprows=range(2),delimiter=r"\s+")
    mesh = pymesh.load_mesh(mesh_f)
    ls0 = landmarks.iloc[0].tolist()
    ls1 = landmarks.iloc[4].tolist()
    ls2 = landmarks.iloc[9].tolist()
    ls3 = landmarks.iloc[10].tolist()
    head = landmarks.iloc[14].tolist()
    foot_left = landmarks.iloc[7].tolist()
    foot_right = landmarks.iloc[6].tolist()
    landmarks_torso = np.array([ls0,ls1,ls2,ls3])
    centroide = landmarks_torso.mean(0)
    b = pymesh.meshutils.generate_icosphere(0.70,centroide)
    output_mesh = pymesh.boolean(b, mesh, operation="intersection",engine="auto")
    print(mesh_f.split("/")[-1])
    pymesh.save_mesh("out/torsos/"+mesh_f.split("/")[-1], output_mesh)
