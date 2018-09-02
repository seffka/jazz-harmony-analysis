import plots
import lowLevelFeatures as ll
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from plots import degrees
import os
import numpy as np

os.chdir(os.environ['JAZZ_HARMONY_DATA_ROOT'] + "/JazzHarmonyCorpus")

cep = ll.ChromaEvaluationParameters(stepSize=2048, smoothingTime=1.2)
chromaEvaluator = ll.AnnotatedChromaEvaluator(cep)
chromas = chromaEvaluator.loadChromasForAnnotationFileListFile('ready.txt')
dMaj = pd.DataFrame(data=preprocessing.normalize(chromas.chromas[chromas.kinds == 'maj'], norm='l1'),  columns=degrees)

m = np.mean(dMaj)

plt.plot(m)
plt.show()

sns.violinplot(data=dMaj, inner="point")
plt.show()

maj = preprocessing.normalize(chromas.chromas[chromas.kinds == 'maj'], norm='l1')
fig, ax = plt.subplots(figsize=(9, 8), dpi= 90, facecolor='w', edgecolor='k')
#plots.plotMajHexagram(ax, maj, step = 60)
plots.plotMajHexagram(ax, maj, 60, labelSize=18)
plt.show()

