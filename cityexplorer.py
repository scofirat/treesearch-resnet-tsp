import csv
import random as rnd
import featurizer
import math
from sklearn.neighbors import KDTree
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

prime={}
def is_prime(num):
    if num in prime:
        return prime[num]
    for i in range(2,int(num**0.5+1)):
        if(num%i==0):
            prime[num]=False
            return False
    prime[num]=True
    return True
class CityExplorer:
    """
    This class represents the game CityExplorer
    """
    __feat = PolynomialFeatures(degree=2, include_bias=False)

    def __init__(self):
        reader=csv.reader(open('cities.csv','r'))
        self.__actions={}
        self.__cur_state={}
        self.__path=[]
        self.__path_size=0
        self.__all_actions=[]
        next(reader,None)
        for row in reader:
            id=int(row[0])
            x=float(row[1])
            y=float(row[2])
            self.__all_actions.append([x,y])
            self.__actions[int(id)]={"id":id,
                             "x":x,
                             "y":y}

        self.__tree=KDTree(self.__all_actions)
        self.__initCity=self.__actions[0]

        self.make_move(0)
        self.__state=[[[0.0]*8]*10]*10
        for move in self.__actions.values():
            distance = self.distanceTo(move["id"])
            clusterX = int(move["x"] / 512)
            clusterY = int(move["y"] / 340)
            self.__state[clusterX][clusterY][0] += 1
            self.__state[clusterX][clusterY][1] = (self.__state[clusterX][clusterY][1] * (
                    self.__state[clusterX][clusterY][0] - 1) + distance) / self.__state[clusterX][clusterY][0]
            self.__state[clusterX][clusterY][2]+=float(is_prime(move["id"]))
            dx= move["x"]-self.__cur_state["x"]
            dy = move["y"]-self.__cur_state["y"]
            self.__state[clusterX][clusterY][3]=(self.__state[clusterX][clusterY][3] * (
                    self.__state[clusterX][clusterY][0] - 1) + dx) / self.__state[clusterX][clusterY][0]
            self.__state[clusterX][clusterY][4]=(self.__state[clusterX][clusterY][1] * (
                    self.__state[clusterX][clusterY][0] - 1) + dy) / self.__state[clusterX][clusterY][0]
            self.__state[clusterX][clusterY][5]=len(self.__path)%10
            self.__state[clusterX][clusterY][6]=float(len(self.__path)%10>5)
            self.__state[clusterX][clusterY][7]=float(len(self.__path)%10<5)

    def action_batch(self,size_to_search,max_lookup=10000):
        """
        Gets action of maximum size sizeToSearch closest to the current point
        Actions are already featurized
        :param sizeToSearch: Max size returned
        :param maxLookup: maximum size to fetch/lookup
        :return: A square list of floats representing different actions, and moveIds of those actions of the same order
        """
        ans=[]
        ids=[]
        sizeToFetch=min(size_to_search*2,max_lookup)
        if size_to_search>len(self.__actions):
            keys=self.__actions.keys()
            for id in keys:
                ids.append(id)
                move = self.__getMoveRaw(id)
                ans.append(move)
            return self.__feat.fit_transform(ans),ids
        distance,ind = self.__tree.query([[self.__cur_state["x"],self.__cur_state["y"]]],k=sizeToFetch)
        idInd=0
        while True:
            if ind[0][idInd] in self.__actions and len(ans)<size_to_search:
                ids.append(ind[0][idInd])
                move=self.__getMoveRaw(ind[0][idInd])
                ans.append(move)
            idInd+=1
            if len(ans)>=size_to_search:
                return self.__feat.fit_transform(ans),ids
            if sizeToFetch==idInd:
                if max_lookup<=sizeToFetch:
                    if ans== [[]]:
                        toAppend = rnd.choices(list(self.__actions.keys()), k=size_to_search * 3)
                        for id in toAppend:
                            ids.append(id)
                            move = self.__getMoveRaw(id)
                            ans.append(move)
                    return ans, ids
                elif  not len(ans)==0:
                    return self.__feat.fit_transform(ans),ids
                else:
                    sizeToFetch=min(sizeToFetch*2,max_lookup)
                    distance, ind = self.__tree.query([[self.__cur_state["x"], self.__cur_state["y"]]], k=sizeToFetch)

    def distanceTo(self,id):
        """Returns the cost of travelling from the current city to city of id"""
        try:
            action = self.__actions[id]
            pureDistance = math.sqrt((self.__cur_state["x"] - action["x"]) ** 2 + (self.__cur_state["y"] - action["y"]) ** 2)
            if (len(self.__path))%10==0:
                if not is_prime(self.__cur_state["id"]):
                    return pureDistance*1.1
                else:
                    return pureDistance
            return pureDistance
        except KeyError:
            return 0.0

    def make_move(self,id):
        try:
            x = int(self.__actions[id]["x"] / 512)
            y = int(self.__actions[id]["y"] / 340)
            pureDistance = math.sqrt((self.__cur_state["x"] - self.__actions[id]["x"]) ** 2 + (
                        self.__cur_state["y"] - self.__actions[id]["y"]) ** 2)
            self.__state[x][y][0] -= 1
            try:
                self.__state[x][y][1] = (self.__state[x][y][1] * (self.__state[x][y][0] + 1) - pureDistance) / self.__state[x][y][0]
            except ZeroDivisionError:
                self.__state[x][y][1] = -1.0
            self.__state[x][y][2] -= float(is_prime(id))
            dx=self.__actions[id]["x"]-self.__cur_state["x"]
            dy =  self.__actions[id]["y"]-self.__cur_state["y"]
            for x in range(0,10):
                for y in range(0,10):
                    self.__state[x][y][1] = math.sqrt((self.__state[x][y][3]-dx)**2+(self.__state[x][y][4]-dy)**2)
                    self.__state[x][y][3] += dx
                    self.__state[x][y][4] += dy
                    self.__state[x][y][5] = len(self.__path)+1 % 10
                    self.__state[x][y][6] = float(len(self.__path)+1 % 10 > 5)
                    self.__state[x][y][7] = float(len(self.__path)+1 % 10 < 5)

        except KeyError:
            pass
        self.__path_size += self.distanceTo(id)
        self.__cur_state=self.__actions.pop(id)
        self.__path.append(id)
        if len(self.__actions)==0 and not self.__path[-1]==0 and len(self.__path)>1:
            self.__actions[0]=self.__initCity

    def is_completed(self):
        """If the tour has been completed"""
        return len(self.__actions)==0 and self.__path[-1]==0 and len(self.__path)>1

    def get_move(self,id):
        """Gets the move features"""
        return self.__feat.fit_transform([self.__getMoveRaw(id)])

    def __getMoveRaw(self,id):
        move = self.__actions[id]
        try:
            move["distanceTo"]
            move["dx"]
        except KeyError:
            move["distanceTo"] = self.distanceTo(id)
            move["dx"] = move["x"] - self.__cur_state["x"]
            move["dy"] = move["y"] - self.__cur_state["y"]
        return featurizer.featurize_action(move)

    def path(self):
        return self.__path

    def path_size(self):
        return self.__path_size

    def path_size_feature(self):
        return featurizer.featurize_path_size(self.path_size())

    def batch_move(self,moveIds):
        for id in moveIds:
            if len(self.__actions)==0:
                self.__actions[0]=self.__initCity
            self.__path_size+=self.distanceTo(id)
            self.__path.append(id)
            self.__cur_state=self.__actions.pop(id)
        self.__state=[[[0.0]*8]*10]*10
        for move in self.__actions.values():
            distance = self.distanceTo(move["id"])
            clusterX = int(move["x"] / 512)
            clusterY = int(move["y"] / 340)
            self.__state[clusterX][clusterY][0] += 1
            self.__state[clusterX][clusterY][1] = (self.__state[clusterX][clusterY][1] * (
                    self.__state[clusterX][clusterY][0] - 1) + distance) / self.__state[clusterX][clusterY][0]
            self.__state[clusterX][clusterY][2]+=float(is_prime(move["id"]))
            dx= move["x"]-self.__cur_state["x"]
            dy = move["y"]-self.__cur_state["y"]
            self.__state[clusterX][clusterY][3]=(self.__state[clusterX][clusterY][3] * (
                    self.__state[clusterX][clusterY][0] - 1) + dx) / self.__state[clusterX][clusterY][0]
            self.__state[clusterX][clusterY][4]=(self.__state[clusterX][clusterY][1] * (
                    self.__state[clusterX][clusterY][0] - 1) + dy) / self.__state[clusterX][clusterY][0]
            self.__state[clusterX][clusterY][5]=len(self.__path)%10
            self.__state[clusterX][clusterY][6]=float(len(self.__path)%10>5)
            self.__state[clusterX][clusterY][7]=float(len(self.__path)%10<5)

    def get_state(self):
        return self.__state

    def training_features(self,result):
        return featurizer.featurize_path_size(result - self.path_size())
