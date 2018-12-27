from cityexplorer import CityExplorer
import time
import matplotlib.pyplot as plt
import csv
import copy
import featurizer

class treesearcher:
    """
        This class is for searching the game tree
    """
    def __init__(self,nnet):
        self.curGame=CityExplorer()
        self.nnet=nnet
    def search(self,concordeSteps=0,nearest_neighborSteps=0,nnSteps=0):
        time1=time.time()
        if concordeSteps!=0:
            self.__fileTrain("concorde.csv",upSample=concordeSteps)
        if nearest_neighborSteps!=0:
            self.__fileTrain("nearest_neighbor_cities.csv",upSample=nearest_neighborSteps)
        print("Time Elapsed="+str(time.time()-time1))
        for i in range(0,nnSteps):
            time1 = time.time()
            self.__searchGame()
            print("Time Elapsed=" + str(time.time() - time1))

    def eval(self,fileName="eval.csv",plot=True,look_up=50):
        time1=time.time()
        self.__searchGame(lookup=look_up,train=False,plot=plot)
        print(self.curGame.path_size())
        writer=csv.writer(open(fileName,'w+'))
        for move in self.curGame.path():
            writer.writerow([str(move)])
        self.curGame=CityExplorer()
        print("Time Elapsed="+str(time.time()-time1))

    def __searchGame(self,lookup=100,train=True,plot=False):
        if plot:
            xAxis=[]
            yAxis=[]
        while not self.curGame.isCompleted():
            if len(self.curGame.path())%100 == 0 and plot:
                xAxis.append(float(len(self.curGame.path()))/1000)
                yAxis.append(self.curGame.path_size()/1000)
            self.__itrSearch(lookup)
        if (plot):
            plt.plot(xAxis, yAxis)
            plt.autoscale()
            plt.show()
        if train:
            if self.curGame.isCompleted():
                self.__train(self.curGame.path_size(),self.curGame.path())

    def __itrSearch(self,lookupModifier=100):
        stateTensor = [self.curGame.getState()]
        actionBatch, movesId = self.curGame.action_batch_v2(lookupModifier)
        choice = self.nnet.predict(stateTensor, actionBatch)
        self.curGame.makeMove(movesId[int(choice[0])])
        return choice

    def __train(self,result,path,fromMove=1,upSample=1):
        testGame = CityExplorer()
        ans=CityExplorer()
        if len(path)!=1:
            for moveNum in range(1, len(path)):
                if(moveNum>=fromMove):
                    self.nnet.train([testGame.getState()], testGame.getMove(int(path[moveNum]))*upSample, testGame.training_features(result))
                if fromMove>1 and fromMove==moveNum:
                    ans=copy.deepcopy(testGame)
                testGame.makeMove(path[moveNum])
            if fromMove>1:
                self.curGame=CityExplorer()
            else:
                self.curGame=ans
        self.nnet.save()

    def __fileTrain(self,filePath,upSample=10):
        reader=csv.reader(open(filePath,'r'))
        game=CityExplorer()
        next(reader,None)
        next(reader,None)
        path=[]
        for line in reader:
            path.append(int(line[0]))
        game.batchMove(path)
        print("Done!")
        print(game.path_size())
        self.__train(game.path_size(),game.path(),upSample=upSample)
