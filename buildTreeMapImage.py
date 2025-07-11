import sys
#sys.path.insert(0, '../src')
import pandas as pd
import matplotlib.pyplot as plt
#!pip install squarify
from mpl_extra import treemap as tr
sizex = int(sys.argv[1])

df = pd.read_csv('tempfolder/SelectedGenesNamesCat.csv')
#df = pd.read_csv('tempfolder/treemapRes.csv')
df.head()
fig, ax = plt.subplots(figsize=(sizex,sizex),
                       gridspec_kw=dict(left=0, right=1, top=1, bottom=0),
                       dpi=49,squeeze=True, subplot_kw=dict(aspect=1))
ax.set_axis_off() # remove the axis
ax.set_frame_on(False)
ax.margins(x=0)
ax.margins(y=0)
trc = tr.treemap(ax, df, area='AREA', fill='hdi',
                 labels='gname',norm_x=sizex,norm_y=sizex,
           levels=['L1','L2','L3','gname']
           ,
           textprops={'c':'w', 'reflow':True,
                      'place':'top left', 'max_fontsize':1.5},
           )
print("...................")
fig.savefig('tempfolder/my_plot.png')
print(df.shape[0])
f = open("tempfolder/generect2.txt", "w")
for y in range(0,df.shape[0], 1):
  f.write(str(y)+"\n"+trc.texts["gname"][y]._origin_text+"\n"
  +str(trc.patches["gname"][y].get_x())+"\n"
  +str(trc.patches["gname"][y].get_y())+"\n"
  +str(trc.patches["gname"][y].get_width())+"\n"
  +str(trc.patches["gname"][y].get_height())+"\n")
f.close()





