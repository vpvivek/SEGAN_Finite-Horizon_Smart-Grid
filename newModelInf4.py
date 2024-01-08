import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
from copy import deepcopy
avgCount = 5
resultsData = np.zeros((avgCount , 4))
for avg in range(avgCount):
    np.random.seed(avg)
    print("seed ", avg)
    startTime =  time.time()
    renewalEnergySwitch = 0
    #Note: For switching on and off renewal energy, just change in getNextState function.
    maxIterations = 2000000
    horizons = 10

    maximum_demand = 4
    demand_array = [1, 2, 3, 4] # 4 possible demands valued 1,2,3,4

    maximum_renewal = 4
    renewal_array = [0.8, 1.2, 1.6, 2] # 4 possible renewal energy valued 1,2,3,4

    maximum_price = 4
    price_array = [1, 2, 3, 4] # 4 possible prices valued 1,2,3,4

    maximum_battery = 4  #capacity of the battery
    maximum_MainGrid = 4  # maximum possible amount to buy from main grid

    #Probability matrices
    demandTPM = np.array([[0.3, 0.4, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2], [0.2, 0.4, 0.3, 0.1], [0.1, 0.2, 0.5, 0.2]])
    priceTPM = np.array([[0.3, 0.3, 0.3, 0.1], [0.1, 0.2, 0.5, 0.2], [0.25, 0.35, 0.1, 0.3], [0.3, 0.4, 0.2, 0.1]])
    renewalProb = np.array([0.3, 0.3, 0.2, 0.2])

    print("H,D,B,P", horizons, maximum_demand, maximum_battery, maximum_price)
    print("Renewal Energy", renewal_array)
    if(renewalEnergySwitch == 1):
        print("Renewal Energy ON")
    else:
        print("Renewal Energy OFF")

    #All possible states
    States = [(i, j, k) for i in range(1, maximum_demand+1) for j in range(maximum_battery+1) for k in range(1, maximum_price+1)]
    No_of_states = len(States)



    def getNextDemand(k):
        return int(np.random.choice(demand_array, 1, p=demandTPM[k-1, :]))


    def getNextPrice(k):
        return int(np.random.choice(price_array, 1, p=priceTPM[k-1, :]))


    def getNextRenewal():
        return int(np.random.choice(renewal_array, 1, p=renewalProb))


    def getFeasibleActionsFromState(d, b, p):
        acts = [(uM, uB) for uM in range(maximum_MainGrid + 1) for uB in range(b + 1)]
        return acts
    #print(getFeasibleActionsFromState(2,3,2))
    #Maximal set of actions
    MaximalActionSet = getFeasibleActionsFromState(maximum_demand, maximum_battery, maximum_price)
    No_of_actions = len(MaximalActionSet)

    def chooseHorizon(horizons):
        return random.randint(0, horizons) #This function returns a an integer from the range including both start and stop

    def getCost(h,d,b,p,u1,u2):
        c = 1
        if(h<horizons/2):
            return c*(d-u2) + h*p*u1
        else:
            return c * (d - u2) + h*p*u1

    def getAction(h,d,b,p):
        u1 = random.randint(0, maximum_MainGrid) #how much to buy from main grid
        u2 = random.randint(0, b)  #how much to spent from battery
        return u1, u2


    def getNextState(h,d,b,p,u1,u2):
        # battery update
        bb = b + u1
        bb = bb - u2

        # renewal energy update
        if (renewalEnergySwitch == 1):
            r = getNextRenewal()
        else:
            r = 0

        #r = getNextRenewal() * renewalEnergySwitch

        #r = 0


        bb = bb + r
        bb = min(maximum_battery, bb)
        bb = max(0, bb)
        # demand update
        dd = getNextDemand(d)
        # price update
        pp = getNextPrice(p)
        return dd, bb, pp

    def a_n(n):
        return 1/math.ceil((n+1)/10)
        #return 0.2

    def getStateNumber(d, b, p):
        return (d-1)*(maximum_battery+1)*maximum_price + b*maximum_price + p-1

    def getActionNumber(d,b,p,u1, u2):

        return u1*(b+1) + u2

    #*************************************************************************************************
    #Q Tables, note that for every h,s all actions are not valid , hence not all hxsxa  elements are not used
    hitCount = np.zeros([horizons+1, No_of_states])
    QMatrix_current = np.zeros([horizons+1, No_of_states, No_of_actions])
    QMatrix_prev = np.ones([horizons+1, No_of_states, No_of_actions])
    kr1 = np.sum(QMatrix_prev)
    #print("kr1",kr1)
    #Never used entries set to 0 in the above table
    summ=0
    for hh in range(horizons+1):
        for ss in range(No_of_states):
            d1, b1, p1 = States[ss]
            A = getFeasibleActionsFromState(d1, b1, p1)
            ll = len(A)
            summ = summ + No_of_actions-ll
            for aa in range(ll, No_of_actions):
                QMatrix_prev[hh][ss][aa] = 0

    kr2 = np.sum(QMatrix_prev)
    #print("Test = {} {}".format(kr1-summ, kr2))
    #print(np.linalg.norm(QMatrix_current-QMatrix_prev))

    f = open("output.txt", "a")
    f.truncate(0)
    f.seek(0)
    #f.write("d b p r uM uB  cost \n" )

    #Q-learning starts
    current_State = (1, 2, 2) #initial state s0
    m=0 #recursion index
    #while(np.linalg.norm(QMatrix_current-QMatrix_prev)>5*math.pow(10,-3)):
    while(m<maxIterations):
        m = m + 1
        QMatrix_prev = deepcopy(QMatrix_current)

        h = chooseHorizon(horizons)
        #print("h={}".format(h))
        d, b, p = current_State
        s = getStateNumber(d, b, p)


        # action taken, cost calculated
        u1, u2 = getAction(h, d, b, p)
        current_cost = getCost(h, d, b, p, u1, u2)
        a = getActionNumber(d, b, p, u1, u2)
        #f.write("{} {} {} {} {}  {}   {} \n".format(d, b, p, r, u1, u2, current_cost))
        #a = getActionNumber(u1, u2)
        hitCount[h, s] = hitCount[h, s] + 1
        nextState = getNextState(h, d, b, p, u1, u2)
        d, b, p = nextState
        A = getFeasibleActionsFromState(d, b, p)

        snext = getStateNumber(*nextState)
        ct = hitCount[h, s]
        #print("current cost = {}".format(current_cost))
        if h == horizons:
            QMatrix_current[h, s, a] = (1 - a_n(ct)) * QMatrix_current[h, s, a] + a_n(ct) * (current_cost)
        else:
            QMatrix_current[h, s, a] = (1 - a_n(ct)) * QMatrix_current[h, s, a] + a_n(ct) * (current_cost + 0.9*np.amin(QMatrix_prev[h + 1, snext, 0:len(A)]))
        #print("Qvalue is {}".format(QMatrix_current[h,s,a]))
        current_State = nextState
        #print("m={} Error={}".format(m, np.linalg.norm(QMatrix_current - QMatrix_prev)))
        #print("Sum of QCurr = {}".format(np.sum(QMatrix_current)))
        #print("ggg{}".format(np.linalg.norm(QMatrix_current - QMatrix_prev)))

        if(m==maxIterations):
            break


    policyMatrixLearned = np.zeros((horizons+1, No_of_states))
    valueMatrixLearned = np.zeros((horizons+1, No_of_states))


    for h in range(horizons+1):
        for s in range(No_of_states):
            d, b, p = States[s]
            A = getFeasibleActionsFromState(d,b,p)
            policyMatrixLearned[h, s] = np.argmin((QMatrix_current[h, s, 0 : len(A)])) # I think error here, have to take non zero minimum ,
            # demand and price is never zero
            #print(policyMatrixLearned[h, s])
            valueMatrixLearned[h, s] = np.amin(QMatrix_current[h, s, 0 : len(A)])
    #print(np.sum(QMatrix_current))
    #print(np.sum())

    def getOptimalAction(h,d,b,p): #Finite Horizon Q-learning
        s = getStateNumber(d, b, p)
        a = policyMatrixLearned[h, s]
        ac = getFeasibleActionsFromState(d,b,p)
        return ac[int(a)]
    #*************************************************


    #************************************************************************************************
    #*************************************************************************************************
    #InfQlearning ***************
    #Q Tables, note that for every h,s all actions are not valid , hence not all hxsxa  elements are not used
    hitCountInf = np.zeros([horizons+1, No_of_states])
    QMatrix_currentInf = np.zeros([horizons+1, No_of_states, No_of_actions])
    QMatrix_prevInf = np.ones([horizons+1, No_of_states, No_of_actions])
    kr1Inf = np.sum(QMatrix_prevInf)
    #print("kr1Inf",kr1Inf)
    #Never used entries set to 0 in the above table
    summInf=0
    for hh in range(horizons+1):
        for ss in range(No_of_states):
            d1, b1, p1 = States[ss]
            A = getFeasibleActionsFromState(d1, b1, p1)
            ll = len(A)
            summInf = summInf + No_of_actions-ll
            for aa in range(ll, No_of_actions):
                QMatrix_prevInf[hh][ss][aa] = 0

    kr2Inf = np.sum(QMatrix_prevInf)
    #print("Test = {} {}".format(kr1Inf-summInf, kr2Inf))
    #print(np.linalg.norm(QMatrix_currentInf-QMatrix_prevInf))

    f = open("outputInf.txt", "a")
    f.truncate(0)
    f.seek(0)
    #f.write("d b p r uM uB  cost \n" )

    #Q-learning INF starts
    current_State = (1, 2, 2) #initial state s0
    m=0 #recursion index
    #while(np.linalg.norm(QMatrix_current-QMatrix_prev)>5*math.pow(10,-3)):
    while(m<maxIterations):
        m = m + 1
        QMatrix_prevInf = deepcopy(QMatrix_currentInf)

        h = chooseHorizon(horizons)
        #h=1
        #print("h={}".format(h))
        d, b, p = current_State
        s = getStateNumber(d, b, p)


        # action taken, cost calculated
        u1, u2 = getAction(1, d, b, p)
        current_cost = getCost(1, d, b, p, u1, u2)
        a = getActionNumber(d, b, p, u1, u2)
        #f.write("{} {} {} {} {}  {}   {} \n".format(d, b, p, r, u1, u2, current_cost))
        #a = getActionNumber(u1, u2)
        hitCountInf[1, s] = hitCountInf[1, s] + 1
        nextState = getNextState(1, d, b, p, u1, u2)
        d, b, p = nextState
        A = getFeasibleActionsFromState(d, b, p)

        snext = getStateNumber(*nextState)
        ct = hitCountInf[1, s]
        #print("current cost = {}".format(current_cost))
        # if h == horizons:
        #     QMatrix_currentInf[1, s, a] = (1 - a_n(ct)) * QMatrix_currentInf[1, s, a] + a_n(ct) * (current_cost)
        # else:
        QMatrix_currentInf[1, s, a] = (1 - a_n(ct)) * QMatrix_currentInf[1, s, a] + a_n(ct) * (current_cost + 0.9*np.amin(QMatrix_prevInf[1, snext, 0:len(A)]))
        #print("Qvalue is {}".format(QMatrix_current[h,s,a]))
        current_State = nextState
        #print("m={} Error={}".format(m, np.linalg.norm(QMatrix_current - QMatrix_prev)))
        #print("Sum of QCurr = {}".format(np.sum(QMatrix_current)))
        #print("ggg{}".format(np.linalg.norm(QMatrix_current - QMatrix_prev)))

        if(m==maxIterations):
            break


    policyMatrixInf = np.zeros((horizons+1, No_of_states))
    valueMatrixInf= np.zeros((horizons+1, No_of_states))


    for h in range(horizons+1):
        for s in range(No_of_states):
            d, b, p = States[s]
            A = getFeasibleActionsFromState(d,b,p)
            policyMatrixInf[h, s] = np.argmin((QMatrix_currentInf[h, s, 0 : len(A)])) # I think error here, have to take non zero minimum ,
            # demand and price is never zero
            #print(policyMatrixLearned[h, s])
            valueMatrixInf[h, s] = np.amin(QMatrix_currentInf[h, s, 0 : len(A)])
    #print(np.sum(QMatrix_currentInf))
    #print(np.sum())
    def getOptimalActionInf(h,d,b,p): #infinite Horizon Q-learning
        s = getStateNumber(d, b, p)
        a = policyMatrixInf[1, s]
        ac = getFeasibleActionsFromState(d,b,p)
        return ac[int(a)]
    #*************************************************

    # def getNonOptimalAction1(h, d, b, p):
    #     u1 = maximum_MainGrid
    #     u2 = min(d, b)
    #     return u1, u2

    def getNonOptimalAction2(h, d, b, p):
        if(d>b):
            u1 = d-b
        else:
            u1 = 0
        u2 = min(d, b)
        return u1, u2

    def getNonOptimalAction3(h, d, b, p):
        u1 = d + (maximum_battery-b)
        u2 = min(d, b)
        return u1, u2

    u1, u2 = getOptimalAction(0,3,1,1) #(h,s)
    #print(u1, u2)

    #print(valueMatrixLearned[0,getStateNumber(3,1,2)])
    #print("c={}".format(c))
    # print(m)
    # print(getFeasibleActionsFromState(2,0,1))
    # print(getFeasibleActionsFromState(1,0,2))
    # print(States[getStateNumber(2,0,1)])

    #for s in States:
        #print(getFeasibleActionsFromState(*s))
        #print("\n")
    #print("zeros")
    count =0
    f.write("h   d b p   uM uB \n")
    for h in range(horizons+1):
        for s in range(No_of_states):

            count = count + 1
            d,b,p = States[s]
            a = policyMatrixLearned[h, s]
            ac = getFeasibleActionsFromState(d, b, p)
            uM, uB = ac[int(a)]

            f.write("{}   {} {} {}   {}   {} \n".format(h, d, b, p, uM, uB))
            #print(h, "  ", d, " ", b, " ",p, "  ", uM, " ", uB)
            #print("a=", uM, uB)
            #print(valueMatrixLearned[h,getStateNumber(d,b,p)])

    f.close()
    #print(count)
    #print((horizons+1)*No_of_states)


    #Testing 1

    current_State = (1, 2, 2) #initial state s0
    m=0 #recursion index
    #while(np.linalg.norm(QMatrix_current-QMatrix_prev)>5*math.pow(10,-3)):
    learnedCost = 0
    while(m<maxIterations):
        m = m + 1

        h = chooseHorizon(horizons)
        #print("h={}".format(h))
        d, b, p = current_State
        s = getStateNumber(d, b, p)

        # action taken, cost calculated
        u1, u2 = getOptimalAction(h, d, b, p)
        current_cost = getCost(h, d, b, p, u1, u2)
        learnedCost = learnedCost + current_cost
        a = getActionNumber(d, b, p, u1, u2)

        nextState = getNextState(h, d, b, p, u1, u2)
        #d, b, p = nextState
        current_State = nextState

        if(m==maxIterations):
            break
    print("Finite Horizon Qlearning : ", learnedCost/maxIterations)
    resultsData[avg,0] = learnedCost/maxIterations

    # #Testing 2
    #
    # current_State = (1, 2, 2) #initial state s0
    # m=0 #recursion index
    # #while(np.linalg.norm(QMatrix_current-QMatrix_prev)>5*math.pow(10,-3)):
    # learnedCost = 0
    # while(m<maxIterations):
    #     m = m + 1
    #
    #     h = chooseHorizon(horizons)
    #     #print("h={}".format(h))
    #     d, b, p = current_State
    #     s = getStateNumber(d, b, p)
    #
    #     # action taken, cost calculated
    #     u1, u2 = getNonOptimalAction1(h, d, b, p)
    #     current_cost = getCost(h, d, b, p, u1, u2)
    #     learnedCost = learnedCost + current_cost
    #     a = getActionNumber(d, b, p, u1, u2)
    #
    #     nextState = getNextState(h, d, b, p, u1, u2)
    #     #d, b, p = nextState
    #     current_State = nextState
    #
    #     if(m==maxIterations):
    #         break
    # print(learnedCost/maxIterations)

    #Testing 3

    current_State = (1, 2, 2) #initial state s0
    m=0 #recursion index
    #while(np.linalg.norm(QMatrix_current-QMatrix_prev)>5*math.pow(10,-3)):
    learnedCost = 0
    while(m<maxIterations):
        m = m + 1

        h = chooseHorizon(horizons)

        d, b, p = current_State
        s = getStateNumber(d, b, p)

        # action taken, cost calculated
        u1, u2 = getNonOptimalAction2(h, d, b, p)
        current_cost = getCost(h, d, b, p, u1, u2)
        learnedCost = learnedCost + current_cost
        a = getActionNumber(d, b, p, u1, u2)

        nextState = getNextState(h, d, b, p, u1, u2)
        #d, b, p = nextState
        current_State = nextState

        if(m==maxIterations):
            break
    print("Meet Entire Demand :", learnedCost/maxIterations)
    resultsData[avg, 1] = learnedCost / maxIterations

    #Testing 4

    current_State = (1, 2, 2) #initial state s0
    m=0 #recursion index

    learnedCost = 0
    while(m<maxIterations):
        m = m + 1

        h = chooseHorizon(horizons)

        d, b, p = current_State

        s = getStateNumber(d, b, p)

        # action taken, cost calculated
        u1, u2 = getNonOptimalAction3(h, d, b, p)
        current_cost = getCost(h, d, b, p, u1, u2)
        learnedCost = learnedCost + current_cost
        a = getActionNumber(d, b, p, u1, u2)

        nextState = getNextState(h, d, b, p, u1, u2)

        current_State = nextState

        if(m==maxIterations):
            break
    print("MEDFC :", learnedCost/maxIterations)
    resultsData[avg, 2] = learnedCost / maxIterations

    #Testing 5

    current_State = (1, 2, 2) #initial state s0
    m=0 #recursion index

    learnedCost = 0
    while(m<maxIterations):
        m = m + 1


        h = chooseHorizon(horizons)

        d, b, p = current_State

        s = getStateNumber(d, b, p)

        # action taken, cost calculated
        u1, u2 = getOptimalActionInf(h, d, b, p)
        current_cost = getCost(h, d, b, p, u1, u2)
        learnedCost = learnedCost + current_cost
        a = getActionNumber(d, b, p, u1, u2)

        nextState = getNextState(h, d, b, p, u1, u2)

        current_State = nextState

        if(m==maxIterations):
            break
    print("Infinite Horizon Q Learning : ", learnedCost/maxIterations)
    resultsData[avg, 3] = learnedCost / maxIterations
    np.save('results_{}_{}_{}_{}_{}'.format(horizons, maximum_demand, maximum_battery, maximum_price, renewalEnergySwitch), resultsData)
    endTime = time.time()
    print("Time taken is {} minutes.".format((endTime-startTime)/60))
    print('*****************************************************************************')


print(resultsData)
results_mean = np.mean(resultsData, axis=0)
results_std_dev = np.std(resultsData, axis=0)
print("mean")
print(results_mean)
print("standard deviation")
print(results_std_dev)


