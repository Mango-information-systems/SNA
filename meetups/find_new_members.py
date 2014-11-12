import json
import logging
import csv
import networkx as nx
import codecs
from enum import Enum
from joblib import Parallel, delayed
logging.basicConfig(level=logging.INFO)

PARALLELISM = 8


class Type(Enum):
    Groups = 1
    Members = 2
    Topics = 3
    Membership = 4
    Interest = 5


def add_member(graph, member, member_key):
    if member_key in graph:
        if 'type' in graph[member_key] and graph[member_key]['type'] != Type.Members:
            raise 'key error: %s exists but is not a member' % member_key
    else:
        graph.add_node(member_key, name=member['name'], type=Type.Members)


def link_members(graph, group, key, members):
    for member_key in group['member_keys']:
        member = members[str(member_key)]
        add_member(graph, member, member_key)
        graph.add_edge(key, member_key, type=Type.Membership)


def add_topic(graph, topic, topic_key):
    if topic_key in graph:
        if 'type' in graph[topic_key] and graph[topic_key]['type'] != Type.Topics:
            raise 'key error: %s exists but is not a topic' % topic_key
    else:
        graph.add_node(topic_key, name=topic['name'], type=Type.Topics)


def link_topics(graph, group, key, topics):
    for topic_key in group['topic_keys']:
        topic = topics[str(topic_key)]
        add_topic(graph, topic, topic_key)
        graph.add_edge(key, topic_key, type=Type.Interest)


def build_meetups_graph(groups, topics, members):
    graph = nx.Graph()
    for group in groups:
        key = group['id']
        graph.add_node(key, name=group['name'], type=Type.Groups, members=group['members'])
        link_members(graph, group, key, members)
        #link_topics(graph, group, key, topics)
    nx.freeze(graph)

    logging.info('graph built')
    return graph


def calculate_connectedness(graph, members, neighbour_member):
    connectedness = 0
    for member in members:
        #for each member, calculate the number of direct connection with our members.
        #direct meaning: being a member of the same meetup group
        paths = list(nx.all_simple_paths(graph, source=member, target=neighbour_member, cutoff=2))
        connectedness += len(paths)
    return connectedness


def build_new_member_list(group, graph, members_dict):
    '''
        collect 3 metrics to identify good candidates:
        1. Activity: should be active in the meetup groups
        2. Connectedness: Should be closely connected to our current members
        3. Interests: should share many of the interests of the group
    '''
    candidates = {}
    key = group['id']
    center = graph[key]

    # get all members of our meetup group
    members = graph.neighbors(key)
    # get people who are members of groups which also contains members of our group
    neighbour_members = [key for key,value in nx.single_source_shortest_path_length(graph,key,cutoff=3).iteritems() if value == 3]
    size = len(neighbour_members)
    count = 0
    for neighbour_member in neighbour_members:
        candidates[neighbour_member] = dict(info=members_dict[str(neighbour_member)], connectedness=0, activity=0, interests=0)
        connectedness = calculate_connectedness(graph, members, neighbour_member)
        count +=1
        candidates[neighbour_member]['connectedness'] = connectedness
        logging.info('added member %d/%d %s', count, size, str(candidates[neighbour_member]))

    return candidates



def get_all_groups(graph, member_keys):
    return [edge[0] for edge in graph.edges_iter(member_keys, data=True) if edge[2]['type']==Type.Membership]


def get_all_members(graph, group_key):
    return get_all_members(graph, [group_key])


def get_all_members(graph, group_keys):
    return [edge[1] for edge in graph.edges_iter(group_keys, data=True) if edge[2]['type']==Type.Membership]


def find_group(groups, urlname):
    for group in groups:
        if group['urlname'] == urlname:
            return group
    return None


def save(new_members, filename):
    member_list = new_members.values()
    sorted_members = sorted(new_members.values(), key=lambda m: m['connectedness'], reverse = True)
    with open(filename, 'w', ) as csvfile:
        w = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['id', 'name', 'connectedness', 'interests', 'activity', 'url', 'thumb', 'photo'])
        for member in sorted_members:
            logging.info('saving %s', str(member))
            thumb = None
            photo = None
            if 'photo' in member['info']:
                thumb = member['info']['photo']['thumb_link']
                photo = member['info']['photo']['photo_link']
            w.writerow([member['info']['id'], member['info']['name'].encode('utf8'),
                        member['connectedness'], member['interests'],member['activity'],
                        member['info']['link'], thumb, photo])


def main():
    topics = json.load(codecs.open('output/topics.json', 'r', encoding='utf-8'))
    members = json.load(codecs.open('output/members.json', 'r', encoding='utf-8'))
    groups = json.load(codecs.open('output/groups.json', 'r', encoding='utf-8'))

    group = find_group(groups, 'Brussels-Data-Science-Community-Meetup')
    graph = build_meetups_graph(groups, topics, members)
    new_members = build_new_member_list(group, graph, members)
    save(new_members, 'output/new_members.csv')

if __name__ == "__main__":
    main()