from collections import OrderedDict
from pdb import set_trace

def collect_prefix(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:len(prefix)] == prefix:
            name = k[(len(prefix) + 1):] # remove "prefix."
            new_state_dict[name] = v
    return new_state_dict

def rename_prefix(state_dict, prefix_old, prefix_new):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print(f'old key: {k}')
        for idx, prefix in enumerate(prefix_old):
            if k[:len(prefix)] == prefix:
                # print(f'new suffix: {k[(len(prefix) + 1):]}')
                # print(f'new prefix: {prefix_new[idx]}')
                name = prefix_new[idx] + k[(len(prefix) + 1):] # replace old prefix with a new one
                new_state_dict[name] = v
                # print(f'new key: {name}')

    return new_state_dict