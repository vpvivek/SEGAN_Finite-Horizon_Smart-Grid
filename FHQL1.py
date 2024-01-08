import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "Times New Roman"
from copy import deepcopy
#HParray = np.array([[10,5,5],[20, 50, 10],[20,50,5]])
HParray = np.array([[10,5,5]])
#HParray = [(20,20,10),(20,50,10),(10,5,5),(10,20,10),(100,20,20)]

for conf in range(np.size(HParray, 0)):
    horizons = HParray[conf, 0]
    No_of_states = HParray[conf, 1]
    No_of_actions = HParray[conf, 2]

    max_iterations = 2000000

    states = np.arange(No_of_states)
    actions = np.arange(No_of_actions)


    costVector = np.random.rand(horizons+1,No_of_states,No_of_actions)*10

    Probbase = np.random.rand(horizons+1,No_of_states,No_of_actions,No_of_states)
    ProbMatrix = np.apply_along_axis(lambda x: x/np.sum(x), 3, Probbase)

    terminalCost = np.arange(No_of_states)
# terminalCost[0]=1
# terminalCost[1]=2
# terminalCost[2]=3
# terminalCost[3]=4
# terminalCost[4]=5
# terminalCost[5]=4
# terminalCost[6]=3
# terminalCost[7]=2
# terminalCost[8]=1
# terminalCost[9]=1

    QMatrix_current = np.zeros([horizons+1,No_of_states,No_of_actions])
    QMatrix_prev = np.ones([horizons+1,No_of_states,No_of_actions])
    QMatrix_DP = np.zeros([horizons+1,No_of_states,No_of_actions])
    #QMatrix_DP_prev = np.ones([horizons+1,No_of_states,No_of_actions])
    tot_count = np.zeros((horizons + 1, No_of_states, No_of_actions, No_of_states))

    for s in range(No_of_states):
        for a in range(No_of_actions):
            QMatrix_current[horizons,s,a]= terminalCost[s]
            QMatrix_prev[horizons,s,a] = terminalCost[s]
            QMatrix_DP[horizons,s,a] = terminalCost[s]
            # QMatrix_DP_current[horizons, s, a] = terminalCost[s]
            # QMatrix_DP_prev[horizons, s, a] = terminalCost[s]

    def sampleState(h, s, a):
        return np.random.choice(states, 1, p=ProbMatrix[h,s,a,:])

    def stepSize(n):
        return 1/math.ceil((n+1)/10)
    #print(np.sum(QMatrix_DP_prev))
    #print(np.sum(QMatrix_DP_current))
    print("Value Iteration")

#Dynamic Programming

    for h in reversed(range(horizons)):
        for s in range(No_of_states):
            for a in range(No_of_actions):
                for snext in range(No_of_states):
                    QMatrix_DP[h, s, a] += ProbMatrix[h, s, a, snext]*(costVector[h, s, a]+ np.amin(QMatrix_DP[h+1, snext, :]))
                    #print(np.sum(QMatrix_DP_current), np.sum(QMatrix_DP_prev))
                    #print(np.linalg.norm(QMatrix_DP_current - QMatrix_DP_prev))

        # if np.linalg.norm(QMatrix_DP_current-QMatrix_DP_prev) < 0.0001:
        #     break
    QlearningError = np.zeros(max_iterations)


    print(np.sum(QMatrix_current))
    print(np.sum(QMatrix_prev))
    print(np.linalg.norm(QMatrix_current-QMatrix_prev))

#Q-learning
    h = 0
    state = np.random.randint(0, No_of_states)
    for m in range(max_iterations):
        if m % 10000 == 0:
            print("Iteration ", m, flush=True)

        # print(n)
        # be careful with h=horizon case
        if (h >= horizons):
            h = np.random.randint(0, horizons)
            state = np.random.randint(0, No_of_states)

        a = np.random.randint(0, No_of_actions)  # numpy has same function , don't confuse
        s_new = int(np.random.choice(np.arange(No_of_states), 1, p=ProbMatrix[h, state, a, :]))


        #r = R[h][state][act1][act2]
        r = costVector[h,state,a]

        # print(Q[s_new,:,:])

        tot_count[h][state][a][s_new] += 1

        next_state_value = np.amin(QMatrix_current[h + 1, s_new, :])

            # Q update
            # print(np.sum(tot_count[h][state][act1][act2]))
        step = stepSize(np.sum(tot_count[h][state][a])) #This works and gives sum over all snew when for [h][state][a]
        QMatrix_current[h, state, a] = (1 - step) * QMatrix_current[h, state, a] + step * (r + next_state_value)
        # print("hi")
        # print(m, np.linalg.norm((QMatrix_current-QMatrix_prev)))
        # print(m, np.linalg.norm((QMatrix_current-QMatrix_DP)))
        #
        # QMatrix_prev = deepcopy(QMatrix_current)
        QlearningError[m] = np.linalg.norm(QMatrix_DP-QMatrix_current)/(math.sqrt((horizons+1)*No_of_states*No_of_actions))

        # if(QlearningError[m]<1.2491):
        #     print(m)
        #     quit()

        if(np.linalg.norm(QMatrix_DP-QMatrix_current)/np.linalg.norm(QMatrix_DP)<.1):
            print(np.linalg.norm(QMatrix_DP - QMatrix_current))
            print(QlearningError[m])
            print(m)
            quit()

            # print("hihi Q")
            # print(np.sum(Q))

        h = h + 1
        state = s_new

    np.save('FHQLError_{}_{}_{}.npy'.format(horizons,No_of_states,No_of_actions),QlearningError)
    policyMatrixDP = np.zeros([(horizons+1), No_of_states])
    valueMatrixDP = np.zeros([horizons+1,No_of_states])
    policyMatrixLearned = np.zeros([horizons+1,No_of_states])
    valueMatrixLearned = np.zeros([horizons+1,No_of_states])

    for h in range(horizons+1):
        for s in range(No_of_states):
            policyMatrixDP[h, s] = np.argmin(QMatrix_DP[h, s, :])
            valueMatrixDP[h, s] = np.amin(QMatrix_DP[h, s, :])
            policyMatrixLearned[h, s] = np.argmin(QMatrix_current[h, s, :])
            valueMatrixLearned[h, s] = np.amin(QMatrix_current[h, s, :])

#Plot error


    midHorizon = math.ceil(horizons/2)
#Plotting value Function
    plot1 = plt.figure(conf + 1)
    plt.xticks(np.arange(0, stop = No_of_states+1 , step = 1))
    #plt.xticks(rotation=90)
    plt.plot(np.arange(No_of_states), valueMatrixLearned[0, :], '-',label='Learned ')
    plt.plot(np.arange(No_of_states), valueMatrixDP[0, :],'--',label='DP')
#plt.plot([],[],label='horizons = {}, states = {}, actions = {}'.format(horizons,No_of_states,No_of_actions))


    plt.title('Optimal Value Function $J_0$: N = {}, |S| = {}, |A| = {}'.format(horizons, No_of_states, No_of_actions))
    plt.xlabel('States')
    plt.ylabel('Optimal Value ')
    lg = plt.figlegend()
    lg.set_draggable(state=True)
    #plt.savefig('value_{}_{}_{}_async'.format(horizons,No_of_states,No_of_actions))
    plt.show()

    plot2 = plt.figure(conf + 2)
    plt.plot(np.arange(No_of_states), policyMatrixLearned[0, :], '-',label='Learned ')
    plt.plot(np.arange(No_of_states), policyMatrixDP[0, :],'--',label='DP')
    plt.xticks(np.arange(No_of_states))
    plt.yticks(np.arange(No_of_actions))
    plt.title('Optimal Policy $\pi_0$ : N = {}, |S| = {}, |A| = {}'.format(horizons, No_of_states, No_of_actions))
    plt.xlabel('States')
    plt.ylabel('Optimal Action ')
    lg = plt.figlegend()
    lg.set_draggable(state=True)
    #plt.savefig('policy_{}_{}_{}_async'.format(horizons,No_of_states,No_of_actions))
    plt.show()
    plot3 = plt.figure(conf + 3)
#plt.axis([0, min(5000,max_iterations), 0, 0.50])
    x = np.arange(max_iterations)
    print(x.shape)
    print(QlearningError.shape)

    #plt.axis([0, m, 0, 50])
    plt.plot(QlearningError, label= 'Finite horizon Q-learning')
    plt.yscale("log")
#plt.xticks(list(range(1,max(x)+1)), [str(i) for i in range(1,max(x)+1)])
    plt.title("Error for N = {}, |S| = {}, |A| = {} ".format(horizons, No_of_states, No_of_actions))
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    lg = plt.figlegend()
    lg.set_draggable(state=True)
    plt.savefig('error_{}_{}_{}_async'.format(horizons,No_of_states,No_of_actions))
    plt.show()
    print(m)
    print(np.linalg.norm(QMatrix_current-QMatrix_DP))
    print(np.linalg.norm(QMatrix_current[0]-QMatrix_DP[0]))
    print(np.sum(QMatrix_DP))
    print(np.sum(QMatrix_current))
