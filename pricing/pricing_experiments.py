import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
from tqdm import tqdm
from queue import PriorityQueue
import pandas as pd
import time
import argparse
import os
from generate_problems import generate_problems

def generate_histories(n_samples, Q, T):
    histories = []
    for i in range(n_samples):
        tau = set()
        for t in range(T):
            tau.add(np.random.randint(Q))
        histories.append(tau)
    return histories

def get_q_by_history(tau): # remove redundant q's if given history is not set
    '''
    input: tau - history of length T
    output: q - output of the history
    '''
    s = set()
    for q in tau:
        s.add(q)
    return s

def compute_profit(histories, X, V, mu = None, return_q_dist = False):
    '''
    input:
        histories - list of trajectories, with q in index form
        X - price
        V - list of value vectors, one for each trajectory
    output:
        profit - profit for the seller
    '''
    profit = 0.0
    q_dist = np.zeros(V.shape[1])
    if mu is None:
        mu = np.array([1.0/V.shape[0] for _ in range(V.shape[0])])
    
    pt = 1/len(histories)

    for tau in histories:
        for v, p in zip(V, mu):
            surplus = v - X
            # find the q that maximizes surplus, given that q is in tau
            q = max(tau, key=lambda q: surplus[q])
            profit += X[q] * p * pt
            q_dist[q] += p * pt
    if return_q_dist:
        return profit, q_dist
    else:
        return profit

class PricingScheme():

    def __init__(self) -> None:
        pass

    def compute_pricing(self, histories, V, mu = None):
        return np.zeros(V.shape[1])
    
class MyersonPricing(PricingScheme):

    def compute_pricing(self, histories, V, mu = None):
        '''
        input:
            histories - list of trajectories, with q in index form  
            V - Theta x Q matrix of valuations for each buyer type and quality
            mu - probability distribution over buyer types. None if uniform
        output:
            X -  optimal pricing vector per quality
        '''
        Q = V.shape[1]
        Th = V.shape[0]
        X = np.zeros(Q)

        if mu is None:
            mu = np.array([1.0/Th for _ in range(Th)])
        
        for q in range(Q):
            Vq = V[:, q]
            # sort mu in descending order based on Vq
            muq = mu[np.argsort(Vq)[::-1]]
            # sort Vq in descending order
            Vq = np.sort(Vq)[::-1]
            # compute cumulative sum of muq
            muq = np.cumsum(muq)
            # find argmax of muq * Vq
            best_th = np.argmax(muq * Vq)
            # set X[q] to Vq[best_th]
            X[q] = Vq[best_th]

        prof = compute_profit(histories, X, V)
        return X, prof
            

class LinearMyersonPricing(PricingScheme):

    def __init__(self, M = None) -> None:
        super().__init__()
        self.M = None

    def compute_pricing(self, histories, V, mu = None):
        '''
        input:
            histories - list of trajectories, with q in index form  
            V - Theta x Q matrix of valuations for each buyer type and quality
            mu - probability distribution over buyer types. None if uniform
            M - number of pricing schemes to test
        output:
            X -  optimal pricing vector per quality
        '''
        Q = V.shape[1]
        Th = V.shape[0]
        M = self.M
        if mu is None:
            mu = np.array([1.0/Th for _ in range(Th)])
        if M is None:
            M = Q
        Xs = []
        ths = []
        shift = -M//2
        for i in range(M):
            X = np.zeros(Q)
            th = np.zeros(Q)
            for q in range(Q):
                Vq = V[:, q]
                # sort mu in descending order based on Vq
                muq = mu[np.argsort(Vq)[::-1]]
                # sort Vq in descending order
                Vq = np.sort(Vq)[::-1]
                # compute cumulative sum of muq
                muq = np.cumsum(muq)
                # find argmax of muq * Vq
                best_th = np.argmax(muq * Vq)
                # shift best_th by shift, but within range
                best_th = min(max(best_th + shift, 0), Th - 1)
                # set X[q] to Vq[best_th]
                X[q] = Vq[best_th]
                th[q] = best_th
            Xs.append(X)
            ths.append(th)
            shift += 1

        # find the best X that maximizes profit
        profits = []
        for X in Xs:
            profits.append(compute_profit(histories, X, V))
        best_X = Xs[np.argmax(profits)]
        best_prof = np.max(profits)
        best_theta = ths[np.argmax(profits)]
        return best_X, best_prof, best_theta
    
class BalancePricing(PricingScheme):

    def __init__(self, M = None) -> None:
        super().__init__()
        self.M = M

    def compute_pricing(self, histories, V, mu = None, starting_th = None):
        '''
        input:
            histories - list of trajectories, with q in index form  
            V - Theta x Q matrix of valuations for each buyer type and quality
            mu - probability distribution over buyer types. None if uniform
            M - number of pricing schemes to test
        output:
            X -  optimal pricing vector per quality
        '''
        Q = V.shape[1]
        Th = V.shape[0]
        M = self.M
        if M is None:
            M = Th * Q
        if mu is None:
            mu = np.array([1.0/Th for _ in range(Th)])
        Xs = []
        profits = []
        Vqs = [] # list of sorted value vectors
        muqs = [] # list of sorted mu vectors by Vq
        ths = np.zeros(Q, dtype=int) # list of best theta for each q
        X = np.zeros(Q)
        for q in range(Q):
            Vq = V[:, q]
            # sort mu in descending order based on Vq
            muq = mu[np.argsort(Vq)[::-1]]
            # sort Vq in descending order
            Vq = np.sort(Vq)[::-1]
            Vqs.append(Vq)
            muqs.append(muq)
            # compute cumulative sum of muq
            cmuq = np.cumsum(muq)
            # find argmax of muq * Vq
            best_th = np.argmax(cmuq * Vq)
            # set X[q] to Vq[best_th]
            X[q] = Vq[best_th] # start with myerson pricing
            ths[q] = best_th

        # print(Vqs)
        if starting_th is not None:
            for q in range(Q):
                ths[q] = starting_th[q]
                X[q] = Vqs[q][ths[q]]
        prof, q_dist = compute_profit(histories, X, V, mu, return_q_dist=True)
        Xs.append(X)
        profits.append(prof)
        leaf_nodes = PriorityQueue() # maintain a priority queue of (X, q_dist, profit), sorted by profit
        # print(profits[0])
        leaf_nodes.put((-profits[0], np.random.rand(), ths, q_dist, X))

        for i in range(M):
            # pop the best leaf node
            _, _, ths, q_dist, X = leaf_nodes.get()

            # if all ths are at Th - 1, skip this node
            if not all([ths[q] >= Th - 1 for q in range(1,Q)]):
                # make copy of ths and X
                ths_up = ths.copy()
                Xup = X.copy()
                # find q with lowest mass out of qs with ths[q] < Th - 1
                # q = np.argmin([q_dist[q] for q in range(1,Q) if ths[q] < Th - 1])
                if np.sum([1-q_dist[q] for q in range(1,Q) if ths[q] < Th - 1]) != 0:
                    q = np.random.choice([q for q in range(1,Q) if ths[q] < Th - 1], p = [1-q_dist[q] for q in range(1,Q) if ths[q] < Th - 1]/sum([1-q_dist[q] for q in range(1,Q) if ths[q] < Th - 1]))
                    # print('up q', q)
                    # increment ths[q]
                    ths_up[q] = min(ths[q] + 1, Th - 1)
                    # print('ths_up', ths_up)
                    # set Xup[q] to Vqs[q][ths_up[q]]
                    Xup[q] = Vqs[q][ths_up[q]]
                    # print('up price change', Vqs[q][ths_up[q]] - Vqs[q][ths[q]])
                    # print('up price', Vqs[q][ths_up[q]])
                    # find profit if we push up
                    upprof, up_q_dist = compute_profit(histories, Xup, V, mu, True)
                    # print('up', up_q_dist)
                    # push up node
                    leaf_nodes.put((-upprof, np.random.rand(), ths_up, up_q_dist, Xup))
                    # append to Xs and profits
                    Xs.append(Xup)
                    profits.append(upprof)
                    # print('upprof', upprof)

            # if all ths are at 0, skip this node
            if not all([ths[q] == 0 for q in range(1,Q)]):
                # make copy of ths and X
                ths_down = ths.copy()
                Xdown = X.copy()
                if np.sum([q_dist[q] for q in range(1,Q) if ths[q] > 0]) != 0:
                    # find q with highest mass out of qs with ths[q] > 0
                    q = np.random.choice([q for q in range(1,Q) if ths[q] > 0], p = [q_dist[q] for q in range(1,Q) if ths[q] > 0]/ sum([q_dist[q] for q in range(1,Q) if ths[q] > 0]))
                    # print('down q', q)
                    # decrement ths[q]
                    ths_down[q] = max(ths[q] - 1, 0)
                    # print('ths_down', ths_down)
                    # set Xdown[q] to Vqs[q][ths_down[q]]
                    Xdown[q] = Vqs[q][ths_down[q]]
                    # print('down price change', Vqs[q][ths_down[q]] - Vqs[q][ths[q]] )
                    # find profit if we push down
                    downprof, down_q_dist = compute_profit(histories, Xdown, V, mu, True)
                    # print('down', down_q_dist)
                    # push down node
                    leaf_nodes.put((-downprof, np.random.rand(), ths_down, down_q_dist, Xdown))
                    # append to Xs and profits
                    Xs.append(Xdown)
                    profits.append(downprof)
                    # print('downprof', downprof)

        # find the best X that maximizes profit
        best_X = Xs[np.argmax(profits)]
        best_prof = np.max(profits)
        return best_X, best_prof

def milp(mu, histories, n_samples, n_types, Q, u):
    '''
    input: 
        mu - probability distribution over types
        histories - list of histories of length n_samples sampled according to the transition probabilities P
        n_samples - number of samples
        n_types - number of types
        Q - number of actions
        T - number of time steps
        P - transition probabilities
        u - utility for each outcome
    output:
        payment - optimal payment with respect to strategy sigma for the buyer
    '''
    # Build the LP model
    model = gp.Model('milp')
    model.setParam('OutputFlag', 0)

    # Add variables
    x = []
    for i in range(Q):
        x.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='x'+str(i)))
    model.addConstr(x[0] == 0.0)
    y = []
    z = []
    for i in range(n_types):
        y_theta = []
        z_theta = []
        for j in range(n_samples):
            y_theta_j = []
            z_theta_j = []
            q_tau = histories[j]
            for q in range(Q):
                y_theta_j.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='y'+str(i)+str(j)+str(q)))
                z_theta_j.append(model.addVar(vtype=GRB.BINARY, name='z'+str(i)+str(j)+str(q)))
                if q not in q_tau:
                    model.addConstr(z_theta_j[q] == 0)
            model.addConstr(gp.quicksum(z_theta_j) == 1.0)
            y_theta.append(y_theta_j)
            z_theta.append(z_theta_j)
        y.append(y_theta)
        z.append(z_theta)
    a = []
    for i in range(n_types):
        a_i = []
        for j in range(n_samples):
            a_i.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='a'+str(i)+str(j)))
        a.append(a_i)
    
    # Set objective
    obj = 0.0
    for i in range(n_types):
        for j in range(n_samples):
            q_tau = histories[j]
            for q in q_tau:
            #for q in range(Q):
                obj += mu[i]*y[i][j][q]*(1.0 / float(n_samples))
    
    M_c = float(2**10)
    for i in range(n_types):
        for j in range(n_samples):
            q_tau = histories[j]
            for q in q_tau:
                buyer_utility = u[i][q] - x[q]
                model.addConstr(a[i][j] - buyer_utility >= 0, name='lb'+str(i)+str(j)+str(q))
                model.addConstr(a[i][j] - buyer_utility <= M_c*(1.0 - z[i][j][q]), name='ub'+str(i)+str(j)+str(q))

    for i in range(n_types):
        for j in range(n_samples):
            q_tau = histories[j]
            for q in q_tau:
                model.addConstr(y[i][j][q] <= x[q], name='feasible_x'+str(i)+str(j)+str(q))
                model.addConstr(y[i][j][q] <= M_c*z[i][j][q] , name='feasible_z'+str(i)+str(j)+str(q))
                model.addConstr(y[i][j][q] >= x[q] - M_c*(1.0 - z[i][j][q]), name='feasible_xz'+str(i)+str(j)+str(q))

    model.setObjective(obj, GRB.MAXIMIZE)
    model.write('specific_strategy_program.lp')
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print(model.printStats())
        return model.objVal, [x[i].x for i in range(Q)], 'optimal'
    elif model.status == GRB.Status.UNBOUNDED:
        return np.PINF, None, 'unbounded'
    else:
        return np.NINF, None, 'infeasible'


def load_history(foldername):
    '''
    input: foldername, folder of .txt files. each file is a trajectory, with each line being a state
    output: list of histories
    '''
    histories = []
    for filename in os.listdir(foldername):
        # only open .txt files
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(foldername, filename), 'r') as f:
            history = []
            for line in f:
                history.append(float(line.strip()))
            histories.append(history)
    return histories

def discretize_history(histories, buckets):
    '''
    input: 
        history - list of states
        buckets - list of buckets
    output: 
        discretized_history - list of discretized states
    '''
    discretized_histories = []
    for history in histories:
        discretized_history = [0]
        for state in history:
            for i in range(len(buckets)):
                if state < buckets[i]:
                    discretized_history.append(i+1)
                    break
        discretized_histories.append(list(set(discretized_history)))
    return discretized_histories

def create_buckets(histories, Q):
    '''
    input: 
        histories - list of histories
        Q - number of buckets
    output:
        buckets - list of buckets, such that each bucket contains equal number of states
    '''
    states = []
    for history in histories:
        states.extend(history)
    states = sorted(states)
    buckets = []
    for i in range(Q):
        buckets.append(states[int(i*len(states)/Q)])
    return buckets
    
def sample_histories(histories, n_types):
    '''
    input: 
        histories - list of histories
        n_types - number of types
    output: 
        sampled_histories - list of sampled histories
    '''
    sampled_histories = []
    for i in range(n_types):
        j = np.random.randint(len(histories))
        sampled_histories.append(histories[j])
    return sampled_histories

def main():
    # set seed
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_types', type=int, default=10)
    parser.add_argument('--Q', type=int, default=10)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--n_problems', type=int, default=100)
    parser.add_argument('--dataname', type=str, default='school_trajectories')
    args = parser.parse_args()
    
    problem_dir = 'problems'
    if not os.path.exists(f'{problem_dir}/problems_{args.n_types}_{args.T}_{args.Q}_{args.n_problems}.pkl'):
        generate_problems(args.n_types, args.T, args.Q, args.n_problems)
    with open(f'{problem_dir}/problems_{args.n_types}_{args.T}_{args.Q}_{args.n_problems}.pkl', 'rb') as f:
        problems = pickle.load(f)
    
    logger_dir = 'logs'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)

    pricing_dir = 'pricing'
    if not os.path.exists(pricing_dir):
        os.makedirs(pricing_dir)
    # logger = open(f'{logger_dir}/log_type_{args.n_types}_T_{args.T}_Q_{args.Q}_np_{args.n_problems}_n_samples_{args.n_samples}.txt', 'w')
    #logger = open(f'{logger_dir}/log_{args.n_types}_{args.T}_{args.Q}.txt', 'w')
    columns=['problem_id', 'welfare_is', 'opt_is_profit', 'milp_os_profit', 'opt_solver_time', 'myerson_is_profit', 'myerson_os_profit', 'myerson_solver_time', 'linear_is_profit', 'linear_os_profit', 'linear_solver_time', 'balanced_is_profit', 'balanced_os_profit', 'balanced_solver_time']
    # columns=['problem_id', 'opt_is_profit', 'opt_os_profit', 'opt_solver_time']
    # create np.array to store results, with columns as above and args.n_problems rows
    results = np.zeros((args.n_problems, len(columns)))

    # create np.array to store pricing for each pricing method
    milp_pricing_results = np.zeros((args.n_problems, args.Q))
    myerson_pricing_results = np.zeros((args.n_problems, args.Q))
    linear_pricing_results = np.zeros((args.n_problems, args.Q))
    balanced_pricing_results = np.zeros((args.n_problems, args.Q))

    # load trajectories
    history_dir = 'trajectories/'
    history_dataname = args.dataname
    raw_histories = load_history(history_dir + history_dataname + '/')
    buckets = create_buckets(raw_histories, args.Q-1)
    # print('buckets', buckets)
    histories = discretize_history(raw_histories, buckets)

    # print('histories', histories)
    # histories = problems[0]['h']

    # find in distribution pricing and profit
    for i in tqdm(range(args.n_problems)):
        v_is = problems[i]['v']
        v_os = problems[(i+1 )% args.n_problems]['v']
        mu = np.ones(args.n_types) / float(args.n_types)

        # histories_is should contain a single trajectory of all states
        histories_is = [list(range(args.Q))]
        # histories_is = sample_histories(histories, args.n_samples)
        # histories_is = problems[i]['h']
        # histories_is = generate_histories(args.n_samples, args.T, args.Q)
        # print('his', histories_is)

        histories_os = sample_histories(histories, args.n_samples)
        # histories_os = problems[(i+1 )% args.n_problems]['h']
        # histories_os = generate_histories(args.n_samples, args.T, args.Q)

        # in distribution welfare is just sum of last column of v_is times mu
        welfare_is = np.sum(v_is[:,-1] * mu)

        # optimal pricing
        # print('optimal pricing')
        start_t = time.time()
        opt_is_profit, opt_sol, marker = milp(mu, histories_is, len(histories_is), args.n_types, args.Q, v_is)
        solver_time = time.time() - start_t
        # print('optimal pricing done')
        opt_os_profit = compute_profit(histories_os, opt_sol, v_os, mu)
        # save opt_sol to pricing_results
        milp_pricing_results[i] = opt_sol

        # myerson pricing
        start_t = time.time()
        myerson_pricer = MyersonPricing()
        myerson_sol, myerson_is_profit = myerson_pricer.compute_pricing(histories_is, v_is, mu)
        myerson_solver_time = time.time() - start_t
        myerson_os_profit = compute_profit(histories_os, myerson_sol, v_os, mu)
        myerson_pricing_results[i] = myerson_sol

        # linear pricing
        start_t = time.time()
        linear_pricer = LinearMyersonPricing(args.Q)
        linear_sol, linear_is_profit, linear_th  = linear_pricer.compute_pricing(histories_is, v_is, mu)
        linear_solver_time = time.time() - start_t
        linear_os_profit = compute_profit(histories_os, linear_sol, v_os, mu)
        linear_pricing_results[i] = linear_sol

        # balanced pricing
        start_t = time.time()
        balanced_pricer = BalancePricing(args.Q * args.n_types)
        balanced_sol, balanced_is_profit = balanced_pricer.compute_pricing(histories_is, v_is, mu, linear_th)
        balanced_solver_time = time.time() - start_t
        balanced_os_profit = compute_profit(histories_os, balanced_sol, v_os, mu)
        balanced_pricing_results[i] = balanced_sol

        # add to results
        results[i, 0] = i
        results[i, 1] = welfare_is
        results[i, 2] = opt_is_profit
        results[i, 3] = opt_os_profit
        results[i, 4] = solver_time
        results[i, 5] = myerson_is_profit
        results[i, 6] = myerson_os_profit
        results[i, 7] = myerson_solver_time
        results[i, 8] = linear_is_profit
        results[i, 9] = linear_os_profit
        results[i, 10] = linear_solver_time
        results[i, 11] = balanced_is_profit
        results[i, 12] = balanced_os_profit
        results[i, 13] = balanced_solver_time

        # logger.write('problem_id: {}, opt_obj: {}, solver_time: {}, program_status: {}\n'.format(i, opt_obj, solver_time, marker))

    # logger.close()
    df = pd.DataFrame(results, columns=columns)

    # add welfare_os, which is just welfare_is shifted by 1, wrapping around
    df['welfare_os'] = df['welfare_is'].shift(-1)
    df['welfare_os'].iloc[-1] = df['welfare_is'].iloc[0]

    # add opt_os_profit, which is just opt_is_profit shifted by 1, wrapping around
    df['opt_os_profit'] = df['opt_is_profit'].shift(-1)
    df['opt_os_profit'].iloc[-1] = df['opt_is_profit'].iloc[0]

    df.to_csv(f'{logger_dir}/log_th_{args.n_types}_ns_{args.n_samples}_Q_{args.Q}_T_{args.T}_np_{args.n_problems}_dn_{args.dataname}.csv', index=False)

    # save pricing results
    dfMILP = pd.DataFrame(milp_pricing_results, columns=[0] + buckets)
    dfMILP['Pricing Algorithm'] = 'MILP'
    dfMyerson = pd.DataFrame(myerson_pricing_results, columns=[0] + buckets)
    dfMyerson['Pricing Algorithm'] = 'Myerson'
    dfLinear = pd.DataFrame(linear_pricing_results, columns=[0] + buckets)
    dfLinear['Pricing Algorithm'] = 'Linear'
    dfBalanced = pd.DataFrame(balanced_pricing_results, columns=[0] + buckets)
    dfBalanced['Pricing Algorithm'] = 'Balanced'

    dfPricing = pd.concat([dfMILP, dfMyerson, dfLinear, dfBalanced])
    dfPricing.to_csv(f'{pricing_dir}/pricing_results_th_{args.n_types}_ns_{args.n_samples}_Q_{args.Q}_T_{args.T}_np_{args.n_problems}_dn_{args.dataname}.csv', index=False)

    # # save pricing results
    # df = pd.DataFrame(milp_pricing_results, columns=[0] + buckets)
    # df.to_csv(f'{pricing_dir}/pricing_results_milp_th_{args.n_types}_ns_{args.n_samples}_Q_{args.Q}_T_{args.T}_np_{args.n_problems}_dn_{args.dataname}.csv', index=False)

    # df = pd.DataFrame(myerson_pricing_results, columns=[0] + buckets)
    # df.to_csv(f'{pricing_dir}/pricing_results_myerson_th_{args.n_types}_ns_{args.n_samples}_Q_{args.Q}_T_{args.T}_np_{args.n_problems}_dn_{args.dataname}.csv', index=False)

    # df = pd.DataFrame(linear_pricing_results, columns=[0] + buckets)
    # df.to_csv(f'{pricing_dir}/pricing_results_linear_th_{args.n_types}_ns_{args.n_samples}_Q_{args.Q}_T_{args.T}_np_{args.n_problems}_dn_{args.dataname}.csv', index=False)

    # df = pd.DataFrame(balanced_pricing_results, columns=[0] + buckets)
    # df.to_csv(f'{pricing_dir}/pricing_results_balanced_th_{args.n_types}_ns_{args.n_samples}_Q_{args.Q}_T_{args.T}_np_{args.n_problems}_dn_{args.dataname}.csv', index=False)

if __name__ == '__main__':
    main()

'''
Instructions for running:

'''