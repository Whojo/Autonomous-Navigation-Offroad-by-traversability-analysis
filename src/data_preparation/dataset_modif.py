# Python librairies
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

dataset_dir = "/home/gabriel/PRE/datasets/dataset_multimodal_siamese_png_filtered_hard"
output_dir = "/home/gabriel/PRE/datasets"
name = "/dataset_multimodal_siamese_png_filtered_hard_modified"

df = pd.read_csv(dataset_dir + "/traversal_costs.csv")

df_1 = df[df['linear_velocity'] <= df["linear_velocity"].mean()]

nb_1 = df_1.shape[0]

df_2 = df[df['linear_velocity'] <= df["linear_velocity"].mean()].sample(nb_1)

df_total = pd.concat([df_1, df_2])

df_total.plot.scatter(x='linear_velocity', y='traversal_cost')

print(df_total["linear_velocity"].mean(), df_total.shape[0], df.shape[0])

plt.show()

try:
    os.mkdir(output_dir + name)
    print(output_dir + name + " folder created\n")
except OSError :
    try:
        shutil.rmtree(output_dir + name, ignore_errors=True)
        print(output_dir + name + " foldel deleted\n")
        os.mkdir(output_dir + name)
        print(output_dir + name + " folder created\n")
    except OSError :
        print("Aborting\n")
        sys.exit(1)

df_total.to_csv(output_dir + name + "/traversal_costs.csv", index=False)