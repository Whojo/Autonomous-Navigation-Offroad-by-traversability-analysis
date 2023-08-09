"""

Takes a dataset and allows to modify it as you fish by applying a statistical filtering to balance it.
Just launch it as a standalone and it'll do the job.

"""

# Python librairies
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
absolute_path = os.path.dirname(__file__)
import shutil
from sklearn.model_selection import train_test_split
import params
import params.dataset
import params.traversal_cost
import params.learning

#Setup output dataset metadata
name = "/dataset_multimodal_siamese_png_filtered_hard_modified"
dataset_dir = os.path.join(absolute_path, "../../datasets/dataset_multimodal_siamese_png_filtered_hard")
output_dir = os.path.join(absolute_path, "../../datasets" + name)

#Reading the base dataset
df = pd.read_csv(dataset_dir + "/traversal_costs.csv", converters={"image_id": str})

#Applying the statistical filtering you want
df_1 = df[df['linear_velocity'] <= df["linear_velocity"].mean()]
nb_1 = df_1.shape[0]
df_2 = df[df['linear_velocity'] > df["linear_velocity"].mean()].sample(nb_1)
df_total = pd.concat([df_1, df_2])
df_total.plot.scatter(x='linear_velocity', y='traversal_cost')
df.plot.scatter(x='linear_velocity', y='traversal_cost')
print(df_total["linear_velocity"].mean(), df_total.shape[0], df.shape[0])
plt.show()


#Creating the output dir
try:
    os.mkdir(output_dir)
    print(output_dir + " folder created\n")
except OSError :
    try:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(output_dir + " foldel deleted\n")
        os.mkdir(output_dir)
        print(output_dir + " folder created\n")
    except OSError :
        print("Aborting\n")
        sys.exit(1)

#Exporting the total resulting dataset
df_total.to_csv(output_dir + "/traversal_costs.csv", index=False)
image_dir = dataset_dir + "/images/"
os.mkdir(output_dir + "/images/")
dest_image_dir = output_dir + "/images/"

for _, row in df_total.iterrows() :
    image_file = os.path.join(image_dir, row["image_id"])
    shutil.copy(image_file + ".png", dest_image_dir)
    shutil.copy(image_file + "d.png", dest_image_dir)
    shutil.copy(image_file + "n.png", dest_image_dir)

#Doing the train_test_split of sklearn

test_dir = output_dir + "/images_test"
train_dir = output_dir + "/images_train"
os.mkdir(train_dir)
os.mkdir(test_dir)

df_train, df_test = train_test_split(df_total,
                                    train_size=params.learning.TRAIN_SIZE +
                                               params.learning.VAL_SIZE,
                                    stratify=df_total["traversability_label"]
                                    if params.dataset.STRATIFY else None,
                                    )


for _, row in df_train.iterrows() :
    image_file = os.path.join(image_dir, row["image_id"])
    shutil.copy(image_file + ".png", train_dir)
    shutil.copy(image_file + "d.png", train_dir)
    shutil.copy(image_file + "n.png", train_dir)

for _, row in df_test.iterrows() :
    image_file = os.path.join(image_dir, row["image_id"])
    shutil.copy(image_file + ".png", test_dir)
    shutil.copy(image_file + "d.png", test_dir)
    shutil.copy(image_file + "n.png", test_dir)

df_train.to_csv(output_dir + "/traversal_costs_train.csv", index=False)
df_test.to_csv(output_dir + "/traversal_costs_test.csv", index=False)