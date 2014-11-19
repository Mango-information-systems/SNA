import logging
import networkx as nx
import numpy as np
import math
import cPickle as pickle
import os
import scipy.stats
from collections import defaultdict

logging.basicConfig(level=logging.WARN)

### Auxiliary functions




# General purpose function for slicing a graph
# G : graph to be sliced
# ntypes, etypes : (string or list of strings) node/edge types to slice 
def graph_slice(G,ntypes=[],etypes=[],inplace=False,removeedges=False):
    if isinstance(ntypes, basestring):
        ntypes = [ntypes]
    if isinstance(etypes, basestring):
        etypes = [etypes]
    if removeedges:
        if inplace:
            H = G
        else:
            H = G.copy()
        if len(ntypes)>0:
            nodelist = []
            for ntype in ntypes:
                nodelist.append(node_layer[ntype])
            H = H.subgraph(nodelist)
        if len(etypes)>0:
            H.remove_edges_from([e for e in H.edges(data=True) if e[2]['etype'] not in etypes])
    else:
        H = nx.Graph()
        if len(ntypes)>0:
            H.add_nodes_from([n for n in G.nodes(data=True) if n[2]['ntype'] in ntypes])
        else:
            H.add_nodes_from(G.nodes(data=True))
        if len(etypes)>0:
            H.add_edges_from([e for e in G.edges(data=True) if e[2]['etype'] in etypes])
        else:
            print 'not implemented'
    return H

procs = {
    'L2' : lambda v,Degs : v / np.sqrt(Degs),
    'L1' : lambda v,Degs : v / Degs,
    'noprocess' : lambda v,Degs : v}
norms = {
    'one' : lambda v : 1,
    'L1' : lambda v : sum(v),
    'L2' : np.linalg.norm,
    'gmean' : lambda v : scipy.stats.gmean(v[np.nonzero(v)])}

class Sliced_graph:
    
    def __init__(self,**kwargs):
        
        self.v0 = None if 'v0' not in kwargs else kwargs['v0']
        self.AdjMat = None if 'AdjMat' not in kwargs else kwargs['AdjMat']
        self.Degs = None if 'Degs' not in kwargs else kwargs['Degs']
        self.OrderedNodes = None if 'OrderedNodes' not in kwargs else kwargs['OrderedNodes']

    def walk(self,edge_sequence,**kwargs):
        preprocess= procs['noprocess'] if 'preprocess' not in kwargs else kwargs['preprocess']
        postprocess=procs['noprocess'] if 'postprocess' not in kwargs else kwargs['postprocess']
        norm = norms['one'] if 'norm' not in kwargs else kwargs['norm']
        if not isinstance(preprocess,list):
            preprocess = [preprocess]*len(edge_sequence)
        if not isinstance(postprocess,list):
            postprocess = [postprocess]*len(edge_sequence)

        v = self.v0
        for prepr, etype, postpr in zip(preprocess, edge_sequence, postprocess):
            v = prepr(v,self.Degs[etype])
            v = self.AdjMat[etype].dot(v)
            v = postpr(v,self.Degs[etype])
        return v / norm(v)
    
    def get_dict_from_vector(self,v):
        col={}
        v = v.flatten()
        for idx in range(len(v)):
            col[self.OrderedNodes[idx][1]] = v[idx]
        return col

    # Compute graph slices in matrix form + auxiliary data
    def compute_slice_matrices(self,G):
        #Create node and edge layers
        node_layer = defaultdict(list)
        for n in G.nodes():
            node_layer[n[0]].append(n)

        edge_layer = defaultdict(list)
        for e in G.edges(data=True):
            edge_layer[e[2]['etype']].append(e)

        ALLNTYPES = [ntype for ntype in node_layer] 
        ALLETYPES = [etype for etype in edge_layer]

        #### Transform everything into linear algebra...

        self.OrderedNodes=[]
        for ntype in ALLNTYPES:
            self.OrderedNodes = self.OrderedNodes + node_layer[ntype]
        self.NodeIndex = {}
        for idx,n in enumerate(self.OrderedNodes):
            self.NodeIndex[n]=idx

        #Construct Adjacency Matrices for various slices (single edge type)
        self.AdjMat = {}
        self.Degs = {} # Degre
        #Invdegs = {}
        for etype in ALLETYPES:
            print '--computing slice for edge type "'+etype+'"'
            H = graph_slice(G,etypes=etype)
            self.AdjMat[etype] = nx.to_scipy_sparse_matrix(H,self.OrderedNodes,format='csr')
            self.Degs[etype] = np.array([[max(1,float(H.degree(n)))] for n in self.OrderedNodes])
            #Invdegs[etype] = np.reciprocal(Degs[etype])

    def to_pickle(self,picklefile):
        #Pickle the results to file
        #first, make sure the dir exists
        d = os.path.dirname(picklefile)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(picklefile,'w') as f:
            pickle.dump(self,f)

def read_pickle(picklefile):
    with open(picklefile,'r') as f:
        return pickle.load(f)


  