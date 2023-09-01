
#convert observation to current state
def info_to_state(info):
    state = []

    # convert to list
    state.append(info['world'])
    state.append(info['stage'])
    state.append(info['y_pos'])
    state.append(info['x_pos'])
    
    return state
