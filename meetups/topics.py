import json
import logging
from networkx.readwrite import json_graph
import networkx as nx
logging.basicConfig(level=logging.WARN)

def main():
    with open('../crawlers/output/belgian_groups.json', 'r') as f:
        groups = json.load(f)

    topics = {}


    for group in groups:
        group['topic_keys'] = set()
        for topic in group['topics']:
            key = topic['id']
            topics[key]= topic
            group['topic_keys'].add(key)


    graph = nx.Graph()
    for group in groups:
        if len(group['topic_keys']) == 0:
            continue
        category_name = ''
        if 'category' in group:
            category_name = group['category']['name']
        graph.add_node(group['id'], name=group['name'], group=category_name, members=group['members'])
    #for topic in topics:
    #    for group in topic['groups']:

    for left in groups:
        for right in groups:
            #to make sure we add edges only once
            if left['id'] >= right['id']:
                continue
            intersection = left['topic_keys'].intersection(right['topic_keys'])
            if len(intersection) > 1:
                if len(intersection) > len(left['topic_keys'])/10:
                    if len(intersection) > len(right['topic_keys'])/10:
                        intersection_topics = [topics[key] for key in intersection]
                        graph.add_edge(left['id'], right['id'], weight=len(intersection))
                        logging.info('added edge between %s and %s' %(left['id'], right['id']))
    logging.info('graph built')

    d = json_graph.node_link_data(graph) # node-link format to serialize
    json.dump(d, open('html/force.json','w'))
    print('Wrote node-link JSON data to html/force.json')


if __name__ == "__main__":
    main()