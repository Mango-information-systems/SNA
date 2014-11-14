import json
import logging
from networkx.readwrite import json_graph
import networkx as nx
from collections import defaultdict
import numpy as np
import math
import pandas as pd
import pickle
import os
from scipy.stats.mstats import gmean

logging.basicConfig(level=logging.WARN)

### Auxiliary functions

def extract_topics(groups):
    topics = {}
    for group in groups:
        group['topic_keys'] = set()
        for topic in group['topics']:
            key = topic['id']
            topics[key] = topic
            group['topic_keys'].add(key)
    return topics


def extract_members(groups):
    all_members = {}
    for group in groups:
        group['member_keys'] = set()
        members = json.load(open('../crawlers/output/members/members_%s.json' % group['urlname'], 'r'))
        for member in members:
            key = member['id']
            if not key in all_members:
                #bio is different per meetup group. Ignoring that.
                all_members[key] = member
            group['member_keys'].add(key)
    return all_members

def extract_events(groups):
    all_events = {}
    for group in groups:
        group['event_keys'] = set()
        events = json.load(open('../crawlers/output/events/events_%s.json' % group['urlname'], 'r'))
        for event in events:
            key = event['id']
            if not key in all_events:
                all_events[key] = event
            group['event_keys'].add(key)
    return all_events

def extract_rsvps(groups):
    all_rsvps = {}
    for group in groups:
        try:
            group['rsvp_keys'] = set()
            rsvps = json.load(open('../crawlers/output/rsvps/rsvps_%s.json' % group['urlname'], 'r'))
            for rsvp in rsvps:
                key = rsvp['rsvp_id']
                if not key in all_rsvps:
                    all_rsvps[key] = rsvp
                group['rsvp_keys'].add(key)
        except:
            print 'could not load rsvps for meetup ' + group['urlname']
    return all_rsvps

def remove_leafs(G,recursive=True):
    foundleaf=True
    while foundleaf:
        foundleaf=False
        for v in G.nodes():
            if G.degree(v)<2:
                foundleaf=recursive
                G.remove_node(v)

def get_main_cc(G):
    ccs = list(nx.connected_components(G))
    maxlen=0
    for cc in ccs:
        if len(cc) > maxlen:
            maxlen = len(cc)
            maxcc=cc
    return G.subgraph(maxcc).copy()


# Construct full Meetup graph
def construct_meetup_graph(groups,topics,members,events,rsvps):
    G = nx.Graph()

    #Create nodes
    for group in groups:
        G.add_node(('group',group['id'])) # Group
    for event in events:
        G.add_node(('event',event)) # Event
    for member in members:
        G.add_node(('member',member)) # Member
    for key in topics:
        G.add_node(('topic',key)) # Topic

    #Create edges:
    for group in groups:
        for key in group['event_keys']:
            G.add_edge(('event',key),('group',group['id']),etype='event of')
        for key in group['member_keys']:
            G.add_edge(('member',key),('group',group['id']),etype='member of')
        for key in group['topic_keys']:
            G.add_edge(('topic',key),('group',group['id']),etype='topic of')
        for key in group['rsvp_keys']:
            rsvp = rsvps[key]
            if rsvp['response']=='yes':
                G.add_edge(('member',rsvp['member']['member_id']),('event',rsvp['event']['id']),etype='rsvp')

    for key in members:
        if 'topics' in members[key]:
            for topic in members[key]['topics']:
                G.add_edge(('member',key),('topic',topic['id']),etype='interested in')
    return G

# General purpose function for slicing a graph
# G : graph to be sliced
# ntypes, etypes : (string or list of strings) node/edge types to slice 
def graph_slice(G,ntypes=[],etypes=[],inplace=False):
    if isinstance(ntypes, basestring):
        ntypes = [ntypes]
    if isinstance(etypes, basestring):
        etypes = [etypes]
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
    return H

# Compute graph slices in matrix form + auxiliary data and pickle it
def compute_slice_matrices(G):
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

    OrderedNodes=[]
    for ntype in ALLNTYPES:
        OrderedNodes = OrderedNodes + node_layer[ntype]
    NodeIndex = {}
    for idx,n in enumerate(OrderedNodes):
        NodeIndex[n]=idx

    #Construct Adjacency Matrices for various slices (single edge type)
    AdjMat = {}
    Degs = {} # Degre
    Invdegs = {}
    for etype in ALLETYPES:
        print '--computing slice for edge type "'+etype+'"'
        H = graph_slice(G,etypes=etype)
        AdjMat[etype] = nx.to_scipy_sparse_matrix(H,OrderedNodes,format='csr')
        Degs[etype] = np.array([[max(1,float(H.degree(n)))] for n in OrderedNodes])
        Invdegs[etype] = np.reciprocal(Degs[etype])

    #Pickle the results to file
    picklefile = 'pickle/cached_slices.pkl'
    #first, make sure the dir exists
    d = os.path.dirname(picklefile)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(picklefile,'w') as f:
        pickle.dump((OrderedNodes,NodeIndex,AdjMat,Invdegs),f)
        print '-slices saved'

def process(v,Invdegs,process=''):
    if process=='deg':
        return v * Invdegs
    elif process=='sqrtdeg':
        return v * np.sqrt(Invdegs)
    else:
        return v

def normalize(v,normalization=''):
    if normalization=='':
        return v
    elif normalization=='gmean': # Geometric mean, excluding zero entries
        return v / math.exp(np.mean(np.log(v[np.nonzero(v)])))
    elif normalization=='l1': # 
        return v / sum(v)
    elif normalization=='l2': 
        return v / np.linalg.norm(v)

def graph_walk(v0, edge_sequence, AdjMat,Invdegs,preprocess='sqrtdeg',postprocess='sqrtdeg',normalization='gmean'):
    if isinstance(preprocess,basestring):
        preprocess = [preprocess]*len(edge_sequence)
    if isinstance(postprocess,basestring):
        postprocess = [postprocess]*len(edge_sequence)

    v = v0
    for prepr, etype, postpr in zip(preprocess, edge_sequence, postprocess):
        v = process(v,Invdegs[etype],prepr)
        v = AdjMat[etype].dot(v)
        v = process(v,Invdegs[etype],postpr)
    return normalize(v,normalization=normalization)

def get_col_from_vector(v,OrderedNodes):
    col={}
    v = v.flatten()
    for idx in range(len(v)):
        col[OrderedNodes[idx][1]] = v[idx]
    return col


#def main():

### Load Meetup Data
print 'Loading meetup data:'
meetupfrom = 'belgian_groups.json'
#meetupfrom = 'groups_be_tech.json'
print '-loading groups'
with open('../crawlers/output/'+meetupfrom, 'r') as f:
    groups = json.load(f)

groupsdict = {}
for group in groups:
    groupsdict[group['id']] = group
print '-loading members'
members = extract_members(groups)


if not os.path.isfile('pickle/cached_slices.pkl'):
    print 'First time running this script!'
    print '-loading topics'
    topics = extract_topics(groups)
    print '-loading events'
    events = extract_events(groups)
    print '-loading rsvps'
    rsvps = extract_rsvps(groups)
    
    print '-constructing full metup graph'
    G = construct_meetup_graph(groups,topics,members,events,rsvps)
    print '-computing matrices and slices...'
    compute_slice_matrices(G)
else:
    print 'skipping slice computations. To recompute, delete pickle folder.'
print "loading matrices and slices"
with open('pickle/cached_slices.pkl','r') as f:
    OrderedNodes,NodeIndex,AdjMat,Invdegs = pickle.load(f)

#Find the BruDataSci meetup
for idx,n in enumerate(OrderedNodes):
    if n[0]=='group':
        if groupsdict[n[1]]['name']=='Brussels Data Science Meetup':
            BDS = groupsdict[n[1]]
            BDSidx = idx
            break

N = len(OrderedNodes)

#Initialize vector on the Brussels Data Science group
v0 = np.zeros((N,1))
v0[BDSidx,0] = 1

# Metrics defined by edge paths 
#   membership = Kris' "connectivity": members of groups which share members with BruDataSci
#   topic_1 = Being interested in the same topics as BruDataSci
#   topic_2 = Being interested in the same topics as members of BruDataSci
#   topic_3 = Being interested in the same topics as attendees of BruDataSci meetups (wheighted by attendance)
edge_seqs = {'membership':['member of','member of','member of'],
             'topic_1':['topic of','interested in'],
             'topic_2':['member of','interested in','interested in'],
             'topic_3':['event of','rsvp','interested in','interested in']
            }

# Create a dataframe to view the output
df = pd.DataFrame({'id':[m for m in members]})
df['name']=df.id.map(lambda x:members[x]['name'])
df['BruDataSci_member'] = df.id.map(lambda x: x in BDS['member_keys'])
print 'computing metrics'
for metric in edge_seqs:
    # Compute weights using the default (square root) pre and post processing
    v = graph_walk(v0,edge_seqs[metric],AdjMat,Invdegs)
    df[metric] = df.id.map(get_col_from_vector(v,OrderedNodes))

    #v = graph_walk(v0,edge_seqs[metric],AdjMat,Invdegs,preprocess='deg',postprocess='deg')
    #df[metric+'_norm^2'] = df.id.map(get_col_from_vector(v,OrderedNodes))

    #v = graph_walk(v0,edge_seqs[metric],AdjMat,Invdegs,preprocess='',postprocess='deg')
    #df[metric+'_proc_in'] = df.id.map(get_col_from_vector(v,OrderedNodes))

    #v = graph_walk(v0,edge_seqs[metric],AdjMat,Invdegs,preprocess='deg',postprocess='')
    #df[metric+'_proc_out'] = df.id.map(get_col_from_vector(v,OrderedNodes))

    # Compute weights without processing nor normalization
    #v = graph_walk(v0,edge_seqs[metric],AdjMat,Invdegs,preprocess='',postprocess='',normalization='')
    #df[metric+'_noproc'] = df.id.map(get_col_from_vector(v,OrderedNodes))

# Compute weights without processing nor normalization
v = graph_walk(v0,edge_seqs['membership'],AdjMat,Invdegs,preprocess='',postprocess='',normalization='')
df['membership_raw'] = df.id.map(get_col_from_vector(v,OrderedNodes))

df.to_csv('output/metrics.csv',encoding='utf-8')
print 'saved output'
    

#if __name__ == "__main__":
#    main()