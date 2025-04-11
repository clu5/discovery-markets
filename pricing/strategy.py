import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
from tqdm import tqdm
from queue import PriorityQueue
import pandas as pd

class StrategicSearcher():

    def __init__(self, Q, T):
        self.Q = Q
        self.T = T
        self.current_strategy = np.random.randint(0,2, size=(Q,T))

    def next_strategy(self, prev_value):
        ''' input value of current strategy. returns the next strategy to be used '''
        pass

class RandomSearcher(StrategicSearcher):

    def __init__(self, Q, T):
        super().__init__(Q, T)

    def next_strategy(self, prev_value):
        self.current_strategy = np.random.randint(0,2, size=(self.Q,self.T))
        return self.current_strategy

class SequentialSearcher(StrategicSearcher):

    def __init__(self, Q, T):
        super().__init__(Q, T)
        self.current_strategy = np.ones((Q,T))
        self.change_index = (0,T-1)

    def next_strategy(self, prev_value):
        self.current_strategy[self.change_index] = 0
        self.change_index = (self.change_index[0] + 1, self.change_index[1])
        if self.change_index[0] >= self.Q:
            self.change_index = (0, self.change_index[1] - 1)
        return self.current_strategy

class SmartSequentialSearcher(StrategicSearcher):

    def __init__(self, Q, T):
        super().__init__(Q, T)
        self.proposed_changes = PriorityQueue()
        self.to_test = [(0, T-1)]
        self.current_strategy = np.ones((Q,T))
        self.test_index = None

    def next_strategy(self, prev_value):
        # place tested strategy in queue
        if self.test_index is not None:
            self.proposed_changes.put((-prev_value, self.test_index))
        
        if not self.to_test:
            # add more strategies to test
            change_index = self.proposed_changes.get()[1]
            self.current_strategy[change_index] = 0

            # check potential new change position 1 to the right
            if change_index[0] == self.Q-1:
                pass
            elif change_index[1] == self.T-1:
                self.to_test.append((change_index[0] + 1, change_index[1]))
            elif self.current_strategy[change_index[0] + 1, change_index[1]+1] == 0:
                self.to_test.append((change_index[0] + 1, change_index[1]))
            
            # check potential new change position 1 to the left
            if change_index[1] == 0:
                pass
            elif change_index[0] == 0:
                self.to_test.append((change_index[0], change_index[1]-1))
            elif self.current_strategy[change_index[0]-1, change_index[1]-1] == 0:
                self.to_test.append((change_index[0], change_index[1]-1))

            # put proposed changes back in test queue
            while not self.proposed_changes.empty():
                self.to_test.append(self.proposed_changes.get()[1])

        if not self.to_test:
            return None

        out = self.current_strategy.copy()
        self.test_index = self.to_test.pop()
        out[self.test_index] = 0
        return out
    
class QSearcher(StrategicSearcher):

    def __init__(self, Q, T):
        super().__init__(Q, T)
        self.Q = Q - 1 # ignore last Q
        self.q_indices = np.zeros(self.Q, dtype=int) #ignore last Q
        self.current_strategy = np.concatenate((np.ones((Q,1)), np.zeros((Q,T-1))), axis=1)
        self.current_index = -2
        self.values = np.zeros(self.Q)
    

    def next_strategy(self, prev_value):
        ''' input value of current strategy. returns the next strategy to be used '''
        if self.current_index == -2:
            self.current_index += 1
            return np.zeros((self.Q+1, self.T))

        # store value of previous strategy
        self.values[self.current_index] = prev_value

        if self.current_index == self.Q-1:
            # change current strategy
            max_index = np.argmax(self.values)
            self.q_indices[max_index] += 1
            if self.q_indices[max_index] > self.T-1:
                return None
            self.current_strategy[max_index, self.q_indices[max_index]] = 1
            self.values = np.zeros(self.Q)

        # increment current index
        self.current_index += 1
        self.current_index = self.current_index % self.Q

        # j = 0
        # skip over maxed out indices
        while self.q_indices[self.current_index] == self.T-1:
            if self.current_index == self.Q-1:
                # change current strategy
                max_index = np.argmax(self.values)
                self.q_indices[max_index] += 1
                if self.q_indices[max_index] > self.T-1:
                    return None
                self.current_strategy[max_index, self.q_indices[max_index]] = 1
                self.values = np.zeros(self.Q)

            self.current_index += 1
            self.current_index = self.current_index % self.Q
            # j += 1
            # if j > self.Q:
            #     return None

        out = self.current_strategy.copy()
        out[self.current_index, self.q_indices[self.current_index]+1] = 1
        return out

def generate_unique_strategies(Q, T, A = [0,1], startq = 0):
    ''' generates all unique strategies

        in:
            Q - cardinality of output space
            T - cardinality of time 
            A - set of actions
        
        out: 
            Sigma - list of strategies 
    
    '''
    # generate all possible strategies for the T-2 timesteps in which buyer can make choices
    Sigma = generate_arrays((Q, T), top = True, bottom = True)

    # add strategy where buyer quits at start
    Sigma.append(np.zeros((Q, T)))

    #print(Sigma)

    # remove redundant strategies
    Sigma = [sigma for sigma in Sigma if not check_redundancy(sigma)]

    # print(Sigma)
    return Sigma

def find_first_zero_index(arr, n):
    """
    Finds the first index of value 0 in a NumPy array after index n.

    Parameters:
    arr (numpy.ndarray): Input NumPy array
    n (int): Index after which to search for the first zero index

    Returns:
    int: The first index of value 0 after index n, or -1 if not found
    """
    # Check if n is out of bounds
    if n >= len(arr):
        return -1

    # Slice the array from index n and search for the first zero index
    sub_arr = arr[n:]
    zero_index = np.where(sub_arr == 0)[0]

    # Check if zero index is found
    if len(zero_index) > 0:
        return zero_index[0] + n
    else:
        return -1
    
def is_first_one_index(arr, n):
    """
    Finds the first index of value 1 in a NumPy array after index n.

    Parameters:
    arr (numpy.ndarray): Input NumPy array
    n (int): Index after which to search for the first zero index

    Returns:
    boolean: True if the first index of value 1 after index n is found, False otherwise
    """
    # Check if n is out of bounds
    if n >= len(arr):
        return False

    # Slice the array from index n and search for the first one index
    sub_arr = arr[n:]
    one_index = np.where(sub_arr == 1)[0]

    # Check if one index is found
    if len(one_index) > 0:
        return True
    else:
        return False
    
def generate_arrays(shape, top = True, bottom = True):
    """
    Generates all possible numpy arrays of shape `shape` filled with either 0s or 1s.

    if top = True, then the top row of the array is filled with 1s
    if bottom = True, then the bottom row of the array is filled with 0s
    """

    newT = shape[1] - int(top) - int(bottom)
    if newT <= 0:
        return [np.hstack((np.ones(shape[0]), np.zeros(shape[0])))]
    arrays = []
    size = (shape[0]-1) * newT # remember that the bottom row is all zeros
    for i in range(2**(size)):
        array = np.array(list(bin(i)[2:].zfill(size)), dtype=np.byte).reshape(shape[0]-1, newT)
        array = np.concatenate((array, np.zeros((1,newT))), axis = 0)
        if top:
            array = np.concatenate((np.ones((shape[0],1)), array), axis = 1)
        if bottom:
            array = np.concatenate((array, np.zeros((shape[0],1))), axis = 1)
        arrays.append(array)
    return arrays

def check_redundancy(sigma):
    ''' checks if a strategy is redundant
        in:
            sigma - strategy 
    
        out: 
            redundant - boolean
    
    '''
    # zero_indices = np.array(sigma.shape[0])
    cur_zero_index = 0
    for q in range(sigma.shape[0]):
        cur_zero_index = find_first_zero_index(sigma[q,:], cur_zero_index)
        if cur_zero_index == -1:
            return False
        if is_first_one_index(sigma[q,:], cur_zero_index):
            return True
    return False

def generate_decoder(Q, T, A = [0,1]):
    def decoder(sigma, q, t):
        out = sigma[q,t]
        if out == -1:
            raise ValueError('invalid action')
        else:
            return out
    return decoder

def calculate_payoff(sigma, Q, T, decoder, V, P, C, startq = 0):
    ''' calculates payoff of a strategy

        in:
            sigma - strategy 
            Q - cardinality of state space
            T - cardinality of time 
            decoder: sigma x Q x T -> A - decodes the strategy into an action
            V: Q  - utility for each outcome
            P: Q x Q x T -> [0,1] - transition probabilities 
            C: T - cost of time
            startq - starting state
        
        out: 
            payoff - payoff of strategy
    
    '''

    # idea: just need to fill in utility for each action using dynamic programming
    #payoffs = np.zeros((Q, T))
    payoffs = {}
    for q in range(Q):
        for t in range(T):
            payoffs[(q,t)] = None

    # fill in payoffs for last time step
    for q in range(Q):
        payoffs[(q, T-1)] = V[q] - C[T-1]

    # fill in payoffs for all other time steps
    for t in range(T-2, -1, -1):
        for q in range(Q):
            if decoder(sigma, q, t) == 1:
                temp = 0
                for q_prime in range(Q):
                    temp += P[q_prime, q, t] * payoffs[(q_prime, t+1)] 
                payoffs[(q, t)] = temp
            else:
                payoffs[(q, t)] = V[q] - C[t]
        
    return payoffs[(startq, 0)]

def calculate_principal_payoff(sigma, Q, T, decoder, x, P, C, startq = 0):
    ''' calculates principal's payoff of a strategy for single buyer

        in:
            sigma - strategy
            Q - cardinality of output space
            T - cardinality of time 
            decoder: sigma x Q x T -> A - decodes the strategy into an action
            x: Q  - payment for each outcome
            P: Q x Q x T -> [0,1] - transition probabilities 
            C: T - cost of time
        
        out: 
            payoff - principal's payoff of strategy
    
    '''
    # compute expected q 
    expected_q = calculate_expected_q(sigma, Q, T, decoder, P, startq)
    return np.sum(expected_q * x)


def calculate_expected_q(sigma, Q, T, decoder, P, startq = 0):
    ''' calculates expected q of a strategy for single buyer

        in:
            sigma - strategy
            Q - cardinality of output space
            T - cardinality of time 
            decoder: sigma x Q x T -> A - decodes the strategy into an action
            P: Q x Q x T -> [0,1] - transition probabilities 
    
        out: 
            payoff - principal's payoff of strategy
    
    '''
    # probabilities at current time step
    cur_p_q = np.zeros(Q)
    cur_p_q[startq] = 1

    # compute expected q 
    out_p_q = np.zeros(Q)
    for t in range(T):
        out_p_q += cur_p_q * (1-sigma[:, t])
        if np.all(cur_p_q == 0):
            break
        cur_p_q = P[:, :, t] @ (cur_p_q * sigma[:, t])
    return out_p_q

def check_valid_P(P):
    ''' checks if P is a valid transition matrix
        in:
            P: Q x Q x T -> [0,1] - transition probabilities 
        out: 
            valid - boolean
    '''
    for t in range(P.shape[2]):
        if not np.allclose(np.sum(P[:, :, t], axis=0), 1):
            return False
    return True

def smart_binomial_search():
    pass

def solve_for_strategy(sigma, Sigma, decoder, Q, T, u, P, C, marker, startq = 0):
    ''' solves for optimal payment given strategy for the follower
        in:
            sigma - strategy in dictionary form
            Sigma - list of strategies in dictionary form, takes in tuple (t, q) as key
            Q - cardinality of output space
            T - cardinality of time 
            A - set of actions
            V: Q  - utility for each outcome
            P: Q x Q x T -> [0,1] - transition probabilities 
            C: T - cost of time
        out: 
            payment - optimal payment with respect to strategy sigma for the buyer
    '''

    # Build the LP model
    model = gp.Model('specific_strategy_program')
    model.setParam('OutputFlag', 0)

    # Add variables
    x = []
    for i in range(Q):
        x.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name='x'+str(i)))

    # Set objective:
    obj = calculate_principal_payoff(sigma, Q, T, decoder, x, P, C, startq)
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Calculate V and buyer payoff
    V = [u[i] - x[i] for i in range(Q)]
    buyer_payoff = calculate_payoff(sigma, Q, T, decoder, V, P, C, startq)
    # Add IR constraint
    model.addConstr(buyer_payoff >= 0.0)
    # Add IC constraints
    for sigma_prime in Sigma:
        if (sigma_prime == sigma).all():
            continue
        buyer_deceptive_payoff = calculate_payoff(sigma_prime, Q, T, decoder, V, P, C, startq)
        model.addConstr(buyer_payoff >= buyer_deceptive_payoff)
    # Feasibility
    model.addConstr(x[0] == 0.0)


    #model.write('specific_strategy_program_{}.lp'.format(marker))
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        return model.objVal, [x[i].x for i in range(Q)], 'optimal'
    else:
        return np.NINF, None, 'infeasible'


def brute_force_search(Q, T, P, C, u, startq = 0):

    decoder = generate_decoder(Q, T)
    Sigma = generate_unique_strategies(Q, T)
    
    objectives, status, solutions =  [np.NINF]*len(Sigma), [None]*len(Sigma), [None]*len(Sigma)
    
    for i, sigma in enumerate(Sigma):
        obj, sol, sta = solve_for_strategy(sigma, Sigma, decoder, Q, T, u, P, C, i, startq)
        objectives[i], solutions[i], status[i] = obj, sol, sta
        # print(sigma, obj)

    idx = np.argmax(objectives)
    print(Sigma[idx], objectives[idx])
    return objectives[idx], solutions[idx]

def smart_search(Q, T, P, C, u, startq = 0):
    '''
    Note that searching over all strategies using smart search is O(Q x Q x T)
    '''
    decoder = generate_decoder(Q, T)
    smart_searcher = SmartSequentialSearcher(Q, T)
    Sigma = generate_unique_strategies(Q, T)
    objectives, status, solutions =  [],[],[]
    obj = None
    sigma = smart_searcher.next_strategy(obj)
    while sigma is not None:
        obj, sol, sta = solve_for_strategy(sigma, Sigma, decoder, Q, T, u, P, C, 0, startq)
        objectives.append(obj)
        solutions.append(sol)
        status.append(sta)
        sigma = smart_searcher.next_strategy(obj)
        print(sigma, obj)
    idx = np.argmax(objectives)
    return objectives[idx], solutions[idx]

def q_search(Q, T, P, C, u, startq = 0):
    '''
    Note that searching over all strategies using smart search is O(Q x Q x T)
    '''
    decoder = generate_decoder(Q, T)
    smart_searcher = QSearcher(Q, T)
    Sigma = generate_unique_strategies(Q, T)
    objectives, status, solutions =  [],[],[]
    obj = None
    sigma = smart_searcher.next_strategy(obj)
    while sigma is not None:
        obj, sol, sta = solve_for_strategy(sigma, Sigma, decoder, Q, T, u, P, C, 0, startq)
        objectives.append(obj)
        solutions.append(sol)
        status.append(sta)
        sigma = smart_searcher.next_strategy(obj)
        # print(sigma, obj)
    idx = np.argmax(objectives)
    return objectives[idx], solutions[idx]

def sequential_search(Q, T, P, C, u, startq = 0):
    
    decoder = generate_decoder(Q, T)
    smart_searcher = SequentialSearcher(Q, T)
    Sigma = generate_unique_strategies(Q, T)
    search_space = int((Q*T)) 
    objectives, status, solutions =  [np.NINF]*search_space, [None]*search_space, [None]*search_space
    obj = None
    for i in range(search_space):
        sigma = smart_searcher.next_strategy(obj)
        if sigma is None:
            break
        obj, sol, sta = solve_for_strategy(sigma, Sigma, decoder, Q, T, u, P, C, i, startq)
        objectives[i], solutions[i], status[i] = obj, sol, sta
        # print(sigma, obj)
    idx = np.argmax(objectives)
    return objectives[idx], solutions[idx]


def main():
    # Input parameters
    #Q, T, P, C, u  = 2, 2, np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]), np.array([0, 1]), np.array([1, 3])

    # slightly bigger example
    problems = None
    ntypes = 1
    T = 5
    Q = 4
    filename = 'problems_{}_{}_{}.pkl'.format(ntypes, T, Q)
    
    with open(filename, 'rb') as f:
        problems = pickle.load(f)
    
    n_problems = 100

    results_filename = '2results_{}_{}_{}_{}.csv'.format(ntypes, T, Q, n_problems)
    # len(problems)
    logger = open('log.txt', 'w')
    columns=['problem_id', 'brute_force_u', 'sequential_u']
    df = pd.DataFrame(columns=columns)
    for i in tqdm(range(n_problems)):
        n_types, Q, T, P, C, u = problems[i]['ntypes'], problems[i]['Q'], problems[i]['T'], problems[i]['P'], problems[i]['c'], problems[i]['v'][0]
        # print('Problem {}'.format(i))
        # print('Q: {}, T: {}'.format(Q, T))
        # print('P: {}'.format(P))
        # print('C: {}'.format(C))
        # print('u: {}'.format(u))
        # print('Brute force search')
        # obj, sol = brute_force_search(Q, T, P, C, u)
        # print('obj: {}, sol: {}'.format(obj, sol))
        # print('Smart search')
        # obj, sol = smart_search(Q, T, P, C, u)
        # print('obj: {}, sol: {}'.format(obj, sol))
        # print('-----------------------')
    
    
    #Q, T, P, C, u  = 3, 3, np.array([[[0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.8, 0.2, 0]], [[0, 0.5, 0.5], [0, 0.6, 0.4], [0, 0.7, 0.3]], [[0,0,1], [0,0,1], [0,0,1]]]), np.array([0, 2, 3]), np.array([0, 4, 5])
    

    # Q, T, P, C, u  = 2, 2, np.array([[[0.5, 0.5], [0.6, 0.4]], [[0.0, 1.0], [0.0, 1.0]]]), np.array([0, 0.5]), np.array([0, 1.5])

        #P = np.transpose(P, (2, 1, 0))
        #print(P)

        if check_valid_P(P):
            print('P is valid')
        else:
            continue
            #raise ValueError('P is not valid')
        print(u.shape, C.shape, P.shape)
        # currently P is in T x Q0 x Q1 format, need to transpose it to Q1 x Q0 x T
        opt_obj, opt_sol = brute_force_search(Q, T, P, C, u)
        # opt_obj, opt_sol = 0,0
        smart_obj, smart_sol = q_search(Q, T, P, C, u)
        # smart_obj, smart_sol = smart_search(Q, T, P, C, u)
        logger.write("opt_obj: {}, opt_sol: {}, smart_obj: {}, smart_sol: {}\n".format(opt_obj, opt_sol, smart_obj, smart_sol))
        df = pd.concat([df, pd.DataFrame([[i, opt_obj, smart_obj]], columns=columns)])
    logger.close()
    df.to_csv(results_filename)
    #return opt_obj, opt_sol, smart_obj, smart_sol

if __name__ == '__main__':
    main()
    # opt_obj, opt_sol, smart_obj, smart_sol = main()
    # print(opt_obj, opt_sol, smart_obj, smart_sol)
