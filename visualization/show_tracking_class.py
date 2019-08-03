import matplotlib.pyplot as plt
import pandas as pd

from pneumothorax_segmentation.tracking import get_classification_dataframes

pd.plotting.register_matplotlib_converters()

###########
# Methods #
###########

def get_probs_distribution(df_chunks, is_there_pneumothorax):
    "Returns the right probability distribution, in chunks, for sane / non-sane subjects"
    chunks_probs = []
    for i in range(6):
        chunk_ids = (df_chunks[i]["is_there_pneumothorax"] == is_there_pneumothorax).index.values
        chunk = df_chunks[i].drop(chunk_ids[i])
        probs = chunk["probs"].values
        probs = [eval(prob)[int(is_there_pneumothorax)] for prob in probs]
        chunks_probs.append(probs)
    return chunks_probs

def draw_distribution(plt, chunks):
    "Draw the probability distribution for the given chunks"
    parts = plt.violinplot(chunks)
    plt.xticks([1, 6], ["Latest", "Oldest"])
    plt.ylim(0., 1.)
    plt.yticks([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    for pc in parts["bodies"]:
        pc.set_alpha(1)
    parts["cmins"].set_alpha(0)
    parts["cmaxes"].set_alpha(0)
    parts["cbars"].set_alpha(0)

#########################################
# Initializing dataframes and variables #
#########################################

df = get_classification_dataframes()
nb_rows = df["index"].count()
print("Dataframe size: {}".format(nb_rows))

chunk_size = nb_rows // 6
df_chunks = []
for i in range(6):
    df_chunk = df.tail(chunk_size * (i + 1)).head(chunk_size)
    df_chunks.append(df_chunk)

############
# Plotting #
############

fig = plt.figure(figsize=(18, 12))
fig.canvas.set_window_title("U-Net Pneumothorax Segmentation tracking - training over {} rows".format(nb_rows))

# Right Probability of sane subjects
plt.subplot(1, 2, 1)
chunks = get_probs_distribution(df_chunks, False)
draw_distribution(plt, chunks)
plt.title("Right Probability of sane subjects")

# Right Probability of subjects with pneumothorax
plt.subplot(1, 2, 2)
chunks = get_probs_distribution(df_chunks, True)
draw_distribution(plt, chunks)
plt.title("Right Probability of subjects with pneumothorax")

plt.show()
