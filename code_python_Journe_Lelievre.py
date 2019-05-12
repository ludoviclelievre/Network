# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 01:53:28 2017

@author: KD5299
"""

path = 'C:\Users\KD5299\georef-bialek\final-data'

import igraph
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopy
DATA_PATH = r"C:\Users\KD5299\Documents\Dropbox\Docs Share\projet_network\bialek_grid_eu"
#DATA_PATH = '/Users/ludoviclelievre/Dropbox/Docs Share/projet_network/bialek_grid_eu'

np.random.seed(123) # for reproductibility

# import data
nodes = pd.read_csv(os.path.join(DATA_PATH,'buses.csv'))
links = pd.read_csv(os.path.join(DATA_PATH,'lines.csv'))

links = links.drop_duplicates(subset=['From Name','To Name'])
nodes.reset_index(inplace=True)
Fromlinks = pd.merge(links,nodes[['Number','index']],right_on='Number',left_on='From Number',how='left')
Tolinks = pd.merge(links,nodes[['Number','index']],right_on='Number',left_on='To Number',how='left')

# create graph
edges = list(zip(Fromlinks['index'],Tolinks['index']))
g = igraph.Graph(edges=edges, directed=False)
list(g.es)
list(g.vs)

# create graph attributes
g.vs['x'] = nodes['lon'].tolist()
g.vs['y'] = (-nodes['lat']).tolist()
g.vs['numero'] = nodes['index'].tolist()
g.vs['color']  = 'yellow'
g.vs['vertex_shape'] = 'cirlce'


g.es['color'] = 'black'
g.es['numero1'] = Fromlinks['index']
g.es['numero2'] = Tolinks['index']
g.es['width'] =2

#

g.summary()
visual_style = {}
visual_style["vertex_size"] = 4
#visual_style["vertex_label"] = map(lambda i :str(i),g.vs["numero"])
#visual_style['label_size'] = 3
#visual_style['label_color'] = 'k'
#visual_style['edge_width'] = 2



#visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
#visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in G.es["is_formal"]]
#visual_style["bbox"] = (1000, 1000)
#visual_style["margin"] = 20
igraph.plot(g,os.path.join(DATA_PATH,'images','g.png'),**visual_style)
g.write_svg(os.path.join(DATA_PATH,'g.svg'),**visual_style)


# calcul de R
### function computing R ###
g.summary()
def compute_R(g):
    g1 = g.copy()
    gplot = g.copy()
    n = len(g.vs)
    R = 0
    for i in range(n-1):
        list(g1.clusters().sizes())
        sq = float(max(g1.clusters().sizes()))/(n**2)
        max_deg = g1.vs.select(_degree = g1.maxdegree())
        list(max_deg)
        max_deg_first = max_deg[0]
        g1.delete_vertices(max_deg_first.index) 
        R+=sq
#        gplot.vs.select(numero_eq=max_deg_first['numero'])['shape'] = 'triangle-down'
#        gplot.es.select(_source=max_deg_first.index).delete()
#        
    return g1,gplot,R

g1,gplot,R = compute_R(g)
g1.summary()
igraph.plot(gplot,**visual_style)
igraph.plot(g1,**visual_style)

color = ['blue','red','green','purple','pink']


def g1_in_g2(g1,g2):
    edges_dif = g1.es.select(lambda e: (e['numero1'],e['numero2'])  in 
                 list(zip(g2.es['numero1'],g2.es['numero2']))   )
    return edges_dif

def color_subgraph(g1):
    for gs in g1.clusters().subgraphs():
        col= color[np.random.randint(0,5)]
#        g1.vs.select(numero_in=gs.vs['numero'])['color'] = col
#        g1.es.select(numero_in=gs.es['numero'])['color'] = col
        g1_in_g2(g1,gs)['color'] = col
    return g1

g1 = color_subgraph(g1)

igraph.plot(g1,os.path.join(DATA_PATH,'images','gclust.png'),**visual_style)
#igraph.plot(gplot,**visual_style)


import random
from geopy.distance import vincenty
def compute_distance(g,a,b):
    points = zip(g.vs[a,b]['x'],g.vs[a,b]['y'])
    # distance between a and b
    res = vincenty(*points).miles
    return res
a,b=1,2
compute_distance(g,a,b)
# better network? ->gm
delta = 10**-8
niter = 10**6
gm = g.copy()

_,_,R = compute_R(gm)
print("R at the beginning: %.10f"%R)
for it in range(niter):
    v = gm.vcount()
    i = np.random.randint(0,v)
    j = np.random.randint(0,v)
    if i==j:
        continue
    i_neig = gm.neighbors(i)
    j_neig = gm.neighbors(j)
    if not i_neig or not j_neig:
#        print("no neigbhoor")
        continue
    k =random.choice(i_neig)
    l = random.choice(j_neig)
    # double connexion?
    if (l in i_neig) or (k in j_neig):
#        print("double connexion")
        continue
    # self connexion
    if  (l == i) or (k == j):
#        print("self connexion")
        continue
    # compute distances
    eil = compute_distance(gm,i,l)
    ejk= compute_distance(gm,j,k)
    eik = compute_distance(gm,i,k)
    ejl= compute_distance(gm,j,l)
    
    if eil+ejk>eik+ejl:
        # reject
        continue
    #_,_,R_propose = compute_R(gm)
    #if R_propose > R+delta:
     #   continue
    g_test = gm.copy()
    g_test.add_edges([(i,l),(j,k)])
    g_test.delete_edges([(i,k),(j,l)])
    _,_,R_propose = compute_R(g_test)
    
    if R_propose < R+delta:
        continue
    
    print("proposition accepted")
    R = R_propose
    gm.add_edges([(i,l),(j,k)])
    gm.delete_edges([(i,k),(j,l)])
#    self.g.es.select(_between=(self.vertexSI, self.vertexSJ)).delete()

    
print("R at the end: %.10f"%R)

    
        
igraph.plot(gm,**visual_style)
list(gm.vs)
    

# basemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
fig,ax = plt.subplots(1,1)
fig.set_size_inches((7,7))
x1 = -10
x2 = 35
y1 = 35
y2 = 60
bmap = Basemap(resolution='i',projection='merc',llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,ax=ax)
bmap.drawcountries()
bmap.drawcoastlines()

for f,t in edges:
    lon = [nodes.loc[f,"lon"],nodes.loc[t,"lon"]]
    lat = [nodes.loc[f,"lat"],nodes.loc[t,"lat"]]
    x,y = bmap(lon,lat)
    color = "g"
    alpha = 0.7
    width =  1.2

    bmap.plot(x,y,color,alpha=alpha,linewidth=width)

x,y = bmap(nodes["lon"].values,nodes["lat"].values)
bmap.scatter(x,y,color='r',s=10)
fig.tight_layout()
plt.show()
file_name = "grid_eu.pdf"
print("file saved to",file_name)
fig.savefig(os.path.join(DATA_PATH,file_name))


# function computing robustess graph
def compute_graph(g):
    g1 = g.copy()
    n = len(g.vs)
    q = 0.2
    R = [1] # one cluster in initial network
    for i in range(int(round(q*n))):
        list(g1.clusters().sizes())
        sq = float(max(g1.clusters().sizes()))/n
        max_deg = g1.vs.select(_degree = g1.maxdegree())
        list(max_deg)
        max_deg_first = max_deg[0]
        g1.delete_vertices(max_deg_first.index) 
        if (i+1)%round(0.01*n)==0:
            R.append(sq)
#        gplot.vs.select(numero_eq=max_deg_first['numero'])['shape'] = 'triangle-down'
#        gplot.es.select(_source=max_deg_first.index).delete()
#        
    return R

# load, plot fraph dif
gm = igraph.read(os.path.join(DATA_PATH,'gm1.graphml'),'graphml')


fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
#avec la norme RGB
ax1.plot(compute_graph(g),color='r',marker='o',linestyle='-.',label='initial network')
ax1.plot(compute_graph(gm),color='g',marker='o',linestyle='-.',label='improved network')
ax1.fill_between(np.arange(0,20), compute_graph(g), compute_graph(gm),color='green',alpha=0.5)
ax1.set_xlim([0,20])
ax1.set_xticks(range(0,20,2))
ax1.set_xticklabels([str(l) + "%" for l in range(0,20,2)])
ax1.set_xlabel('q')
ax1.set_ylabel('s(q)')
ax1.legend(loc='best')
fig.show()

#gm.save(os.path.join(DATA_PATH,'gm.graphml'),'graphml')


def g1_notin_g2(g1,g2):
    edges_dif = g1.es.select(lambda e: (e['numero1'],e['numero2'])  not in 
                 list(zip(g2.es['numero1'],g2.es['numero2']))   )
    return edges_dif

edges_destroyed = g1_notin_g2(g,gm)
edges_created = g1_notin_g2(gm,g)
edges_created['color']='green'
edges_created['width'] = 2
igraph.plot(gm,os.path.join(DATA_PATH,'images','gm_created.png'),**visual_style)
edges_destroyed['color']='red'
edges_destroyed['width'] = 2
igraph.plot(g,os.path.join(DATA_PATH,'images','gm_destroyed.png'),**visual_style)

gc = g.copy()
list(edges_created)

gc = gc.union(gm)
gc.summary()
g.summary()
edges_created = g1_notin_g2(gc,gm)
edges_created['color']='green'
#edges_destroyed = g1_notin_g2(gc,g)
#edges_destroyed['color']='red'

igraph.plot(gc,**visual_style)

