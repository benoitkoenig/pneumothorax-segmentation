import matplotlib.pyplot as plt
import pandas as pd

from pneumothorax_segmentation.tracking import get_segmentation_dataframes

pd.plotting.register_matplotlib_converters()

#########################################
# Initializing dataframes and variables #
#########################################

df = get_segmentation_dataframes()
nb_rows = df["index"].count()
print("Dataframe size: {}".format(nb_rows))

chunk_size = nb_rows // 6
df_chunks = []
for i in range(6):
    df_chunk = df.tail(chunk_size * (i + 1)).head(chunk_size)
    df_chunks.append(df_chunk)
sane_pictures_in_chunk = [df_chunk["IoU"].dropna().index.values for df_chunk in df_chunks]

############
# Plotting #
############

fig = plt.figure(figsize=(18, 12))
fig.canvas.set_window_title("U-Net Pneumothorax Segmentation tracking - training over {} rows".format(nb_rows))

# IoU distribution for images with a disease
plt.subplot(1, 2, 1)
values_IoU = [df_chunk["IoU"].dropna().values for df_chunk in df_chunks]
parts = plt.violinplot(values_IoU)
plt.xticks([1, 6], ["Latest", "Oldest"])
plt.ylim(0., 1.)
plt.yticks([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
for pc in parts["bodies"]:
    pc.set_alpha(1)
parts["cmins"].set_alpha(0)
parts["cmaxes"].set_alpha(0)
parts["cbars"].set_alpha(0)
plt.title("IoU distribution for images with a disease")

# Size of the predicted area on sane subject images
plt.subplot(1, 2, 2)
values_wrong_diagnosis = []
for (i, df_chunk) in enumerate(df_chunks):
    val = df_chunk.drop(sane_pictures_in_chunk[i])
    val = val["prediction_area"].tolist()
    values_wrong_diagnosis.append(val)
parts = plt.violinplot(values_wrong_diagnosis)
plt.xticks([1, 6], ["Latest", "Oldest"])
for pc in parts["bodies"]:
    pc.set_alpha(1)
parts["cmins"].set_alpha(0)
parts["cmaxes"].set_alpha(0)
parts["cbars"].set_alpha(0)
plt.title("Size of the predicted area on sane subject images")

plt.show()
