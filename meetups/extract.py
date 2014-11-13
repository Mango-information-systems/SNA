import json
import logging
logging.basicConfig(level=logging.WARN)

def extract_topics(groups):
    topics = {}
    for group in groups:
        group['topic_keys'] = [topic['id'] for topic in group['topics']]
        for topic in group['topics']:
            key = topic['id']
            topics[key] = topic
    return topics


def extract_members(groups):
    all_members = {}
    for group in groups:
        group['member_keys'] = []
        members = json.load(open('../crawlers/output/members/members_%s.json' % group['urlname'], 'r'))
        for member in members:
            key = member['id']
            if not key in all_members:
                #bio is different per meetup group. Ignoring that.
                all_members[key] = member
            group['member_keys'].append(key)
    return all_members


def main():
    with open('../crawlers/output/belgian_groups.json', 'r') as f:
        groups = json.load(f)

    #only look at groups with at least 10 people
    #groups = [group for group in groups if group['members'] >= 10]
    topics = extract_topics(groups)
    members = extract_members(groups)
    json.dump(topics, open('output/topics.json', 'w'), indent=4, separators=(',', ': '))
    json.dump(members, open('output/members.json', 'w'), indent=4, separators=(',', ': '))
    json.dump(groups, open('output/groups.json', 'w'), indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()