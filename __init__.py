from searcher import treesearcher
import nnet

def main():
    nn=nnet.Estimator("./model")
    searcher=treesearcher(nn)
    #searcher.search(concordeSteps=10)
    #searcher.search(nnSteps=25)
    #searcher.search(nearest_neighborSteps=5)
    #searcher.search(nnSteps=30)
    searcher.eval("eval_25_10.csv",plot=False,look_up=25)
    #searcher.eval("eval_50_10.csv",plot=False,look_up=50)
    #searcher.eval("eval_100_10.csv",plot=False,look_up=100)
    #searcher.eval("eval_10_10.csv",plot=False,look_up=10)

main()