from collections import OrderedDict

def create_new_state_dict(checkpoint, keyword='net'):

    new_state_dict = OrderedDict()
    for k, v in checkpoint[keyword].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict