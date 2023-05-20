# Import libraries needed
import json
import numpy


# This function is used when  the agent is unable to answer the asked question
def unknown():
    return json.dumps({'Answers': []})


# Sort the list of passages on score and return the best option
def best_passage(passages):

    passages = passages[(passages['context'] >= 0.50)]

    # Sort passages by score
    passages = passages.sort_values(by=['Score'], ascending=False, ignore_index=True)

    if len(passages) == 0:
        return unknown()

    # Return a predetermined answer is best score is low or the answer with the best score.
    if passages['Score'][0] < 0.5:
        return unknown()

    max_len = 11
    if len(passages) < 11:
        max_len = len(passages)

    for i in range(max_len):
        if passages['structure_score'][i] == 1:
            return json.dumps({'Answers': passages['NL'][:3].to_list()})

    return json.dumps({'Answers': passages['NL'][:3].to_list()})
