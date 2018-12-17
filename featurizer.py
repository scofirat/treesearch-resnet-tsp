import cityexplorer
def featurizeAction(move):
    ans_x=[0]*10
    ans_y=[0]*10
    ans_x[int(move["x"]/512)]=1
    ans_y[int(move["y"]/340)]=0
    return [int(cityexplorer.isPrime(move["id"])),
     move["x"],
     move["y"],
     move["distanceTo"],
            move["dx"],
            move["dy"]]+ans_x+ans_y

def featurizePathSize(pathSize):
    cluster=int((pathSize)/200000)
    if cluster > 9:
        return [[0.0]*14+[1.0]]
    ans=[0.0]*15
    ans[cluster] = 1.0
    return [ans]