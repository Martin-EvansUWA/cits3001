
#convert observation to current state
def info_to_state(info):
    state = []

    # convert to list
    state[0] = info['world']
    state[1] = info['stage']
    state[2] = info['y_pos']
    state[3] = info['x_pos']
    
    return state
