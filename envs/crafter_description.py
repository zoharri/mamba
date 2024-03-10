import numpy as np
import crafter

env = crafter.Env(size=(64, 64))
action_space = env.action_space

vitals = ["health","food","drink","energy",]

rot = np.array([[0,-1],[1,0]])
directions = ['front', 'right', 'back', 'left']

id_to_item = [0]*19
import itertools
for name, ind in itertools.chain(env._world._mat_ids.items(), env._sem_view._obj_ids.items()):
    name = str(name)[str(name).find('objects.')+len('objects.'):-2].lower() if 'objects.' in str(name) else str(name)
    id_to_item[ind] = name
player_idx = id_to_item.index('player')
print(id_to_item)

def describe_inventory(info):
    result = ""

    status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info['inventory'][v]) for v in vitals]))
    result += status_str + "\n\n"

    inventory_str = "\n".join(["- {}: {}".format(i, num) for i,num in info['inventory'].items() if i not in vitals and num!=0])
    inventory_str = "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    result += inventory_str #+ "\n\n"

    return result.strip()


REF = np.array([0, 1])

def rotation_matrix(v1, v2):
    dot = np.dot(v1,v2)
    cross = np.cross(v1,v2)
    rotation_matrix = np.array([[dot, -cross],[cross, dot]])
    return rotation_matrix

def describe_loc(ref, P):
    # print(ref,P)
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    return "-".join(desc)


def describe_env(info):
    print("printing info from the descriptor: " + str(info) + "\n")
    assert(info['semantic'][info['player_pos'][0],info['player_pos'][1]] == player_idx)
    # pad semantic so that when agent is at a wall, we can still segment semantic with a 9 by 7 view centered at agent.
    padded_semantic = np.pad(info['semantic'], ((4,4),(3,3)), 'constant') #pad 4 on first axis and 3 on 2nd axis
    padded_player_pos = [info['player_pos'][0]+4, info['player_pos'][1]+3]
    # semantic dim is 9 by 7
    semantic = padded_semantic[
               padded_player_pos[0]-info['view'][0]//2 :
               padded_player_pos[0]+info['view'][0]//2+1,
               padded_player_pos[1]-info['view'][1]//2+1:
               padded_player_pos[1]+info['view'][1]//2]
    center = np.array([info['view'][0]//2,info['view'][1]//2-1])
    # center = [4,3]
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x,y)
    # x1, y1 are all relative coordinates within the 9 by 7 view
    loc = np.stack((y1, x1),axis=-1)
    dist = np.absolute(center-loc).sum(axis=-1)
    obj_info_list = []
    print("id to item: " + str(id_to_item) + "\n")
    facing = info['player_facing']
    target = (center[0] + facing[0], center[1] + facing[1])
    print("target:" + str(target) + "\n");
    print("semantic:" + str(semantic) + "\n");
    target = "wall" if semantic[target] == 0 else id_to_item[semantic[target]]
    obs = "You face {} at your front.".format(target, describe_loc(np.array([0,0]),facing))

    # R = rotation_matrix(info['player_facing'], REF)
    for idx in np.unique(semantic):
        item = "wall" if idx == 0 else id_to_item[idx]
        if idx == player_idx:
            continue
        # if id_to_item[idx] == target:
        #     continue
        # print(id_to_item[idx])
        smallest = np.unravel_index(np.argmin(np.where(semantic==idx, dist, np.inf)), semantic.shape)
        obj_info_list.append((item, dist[smallest], describe_loc(np.array([0,0]), smallest-center)))
        # obj_info_list.append((id_to_item[idx], dist[smallest], describe_loc(np.array([0,0]), R @ (smallest-center))))

    if len(obj_info_list)>0:
        status_str = "You see:\n{}".format("\n".join(["- {} {} steps to your {}".format(name, dist, loc) for name, dist, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += obs.strip() + "\n\n" + status_str

    return result.strip()


def describe_act(info):
    result = ""

    action_str = info['action'].replace('do', 'interact')
    action_str = action_str.replace('move_up', 'move_north')
    action_str = action_str.replace('move_down', 'move_south')
    action_str = action_str.replace('move_left', 'move_west')
    action_str = action_str.replace('move_right', 'move_east')

    # act = "Player took action {}.".format(action_str)
    # result+= act #+ "\n\n"

    return action_str.strip()


def describe_status(info):
    return ""
    # Not implemented yet
    if info['sleeping']:
        return "Player is sleeping.\n\n"
    else:
        return ""

    if info['dead']:
        return "Player died.\n\n"
    else:
        return ""


def describe_frame(info):
    result = ""

    # result+=describe_act(info)
    result+=describe_status(info)
    # result+="\n\n"
    result+=describe_env(info)
    result+="\n\n"
    result+=describe_inventory(info)

    return describe_act(info).strip(), result.strip()

action_list = ["Noop", "Move West", "Move East", "Move North", "Move South", "Do", \
    "Sleep", "Place Stone", "Place Table", "Place Furnace", "Place Plant", \
    "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe", "Make Wood Sword", \
    "Make Stone Sword", "Make Iron Sword"]

def match_act(string):
    for i, act in enumerate(action_list):
        if act.lower() in string.lower():
            return i
    print("LLM failed with output \"{}\", taking action Do...".format(string))
    return action_list.index("Do")