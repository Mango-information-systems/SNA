import json
import logging
from networkx.readwrite import json_graph
import networkx as nx
logging.basicConfig(level=logging.WARN)


def add_potential_edge(graph, left, right, bond, ratio=0.125):
    intersection= len(left[bond].intersection(right[bond]))
    union = len(left[bond].union(right[bond]))
    if intersection == 0 or union == 0:
        return
    current_ratio = float(intersection)/float(union)
    if current_ratio > ratio:
        graph.add_edge(left['id'], right['id'], weight=current_ratio*100)
        logging.info('added edge between %s and %s' % (left['id'], right['id']))


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


def main():
    with open('../crawlers/output/belgian_groups.json', 'r') as f:
        groups = json.load(f)

    #only look at groups with at least 10 people
    #groups = [group for group in groups if group['members'] >= 10]
    topics = extract_topics(groups)
    members = extract_members(groups)

    graph = nx.Graph()
    members_graph = nx.Graph()
    for group in groups:
        category_name = ''
        if 'category' in group:
            category_name = group['category']['name']
        members_graph.add_node(group['id'], name=group['name'], group=category_name, members=group['members'])
        if len(group['topic_keys']) > 0:
            graph.add_node(group['id'], name=group['name'], group=category_name, members=group['members'])


    for left in groups:
        for right in groups:
            #to make sure we add edges only once
            if left['id'] >= right['id']:
                continue
            add_potential_edge(graph, left, right, 'topic_keys')
            add_potential_edge(members_graph, left, right, 'member_keys', ratio=0.05)
    logging.info('graph built')

    d = json_graph.node_link_data(graph) # node-link format to serialize
    json.dump(d, open('html/force.json','w'))
    d = json_graph.node_link_data(members_graph) # node-link format to serialize
    json.dump(d, open('html/members_force.json','w'))
    print('Wrote node-link JSON data to html/force.json')


if __name__ == "__main__":
    main()