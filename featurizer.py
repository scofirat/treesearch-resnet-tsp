import cityexplorer


def featurize_action(move):
    ans_x=[0]*10
    ans_y=[0]*10
    ans_x[int(move["x"]/512)]=1
    ans_y[int(move["y"]/340)]=0
    return [int(cityexplorer.is_prime(move["id"])),
     move["x"],
     move["y"],
     move["distanceTo"],
            move["dx"],
            move["dy"]]+ans_x+ans_y


def featurize_path_size(path_size):
    cluster=int((path_size)/200000)
    if cluster > 9:
        return [[0.0]*14+[1.0]]
    ans=[0.0]*15
    ans[cluster] = 1.0
    return [ans]