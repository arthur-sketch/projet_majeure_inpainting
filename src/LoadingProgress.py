import math
from colorama import Fore
from yaml import load



def loadingProgress(progress, total):


    lengthBar = 60
    pourcentage = progress/total 
    barreProgression = ""
    loaded = "="*int(pourcentage*lengthBar)
    toDo = "-"*(int((1-pourcentage)*lengthBar))

    barreProgression = "[" + loaded + toDo + "]"
    
    if progress!=total:
        print( Fore.YELLOW + f"\r{barreProgression}" + Fore.RESET + f"   {pourcentage*100:.1f} %", end="")

    else:
        print( Fore.GREEN + f"\r{barreProgression}" + Fore.RESET + f"   {pourcentage*100:.1f} %")
        Fore.RESET






def test():

    L = [x for x in range(10000, 12000)]

    for k,x in enumerate(L):
        math.factorial(x)
        loadingProgress(k , len(L)-1)



if __name__ == '__main__':
    test()
