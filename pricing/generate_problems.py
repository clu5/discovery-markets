import pickle
import numpy as np
import os
import argparse

# Define a function that checks for first-order stochastic dominance
def fosd_dominates(p1, p2):
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    if np.all(cdf1 <= cdf2) and np.any(cdf1 < cdf2):
        return True
    else:
        return False

def generate_problem(ntypes, T, Q, nhistories = None, a = 0.25, b = 0.75):
    '''
    Generates a random problem instance
    Input: ntypes, T, Q, a, b
    ntypes is the number of agent types
    T is the number of time periods
    Q is the number of actions
    nhistories is the number of histories 
    a is the maximum cost
    b is the trajectory quality probability of inclusion
    Output: problem dictionary
    P is a Q x Q x T matrix of transition probabilities, where P[q', q, t] is the probability of transitioning from q to q' at time t
    '''
    problem = {}
    problem['ntypes'] = ntypes
    problem['T'] = T
    problem['Q'] = Q
    if nhistories is None:
        nhistories = ntypes
    problem['nhistories'] = nhistories
   
    # Generate random parameters
    problem['v'] = np.sort(np.random.rand(ntypes, Q-1)) # Theta x Q matrix of values
    # prepend 0 to each row
    problem['v'] = np.concatenate((np.zeros((ntypes, 1)), problem['v']), axis=1)
    problem['c'] = a*np.linspace(0, 1, T) # vector of costs
    P = np.zeros((Q, T, Q)) # Q x T x Q matrix of transition probabilities

    # Generate random transition probabilities
    for q in range(Q):
        dists = []
        for t in range(T):
            dists += [np.random.dirichlet(np.ones(Q-q))]
            # P[q, t, q:] = np.random.dirichlet(np.ones(Q-q))
        # Sort the list by first-order stochastic dominance
        sorted_dists = sorted(dists, key=lambda x: np.any([fosd_dominates(x, y) for y in dists if not np.array_equal(x, y)]), reverse=True)
        sorted_dists = sorted(sorted_dists, key=lambda x: x[0])
        P[q,:,q:] = np.array(sorted_dists)

    P = np.transpose(P, (2, 0, 1))
    problem['P'] = P

    # Generate random histories
    problem['h'] = []
    for i in range(nhistories):
        # generate subset of Q, H, where each index in Q has probablity b of being included. 0 should always be included
        # print('Q: ', Q)
        H = np.random.choice([False, True], size=Q, p=[1-b, b])
        H[0] = True
        # now select indices with true from [0,...,Q-1]
        QQ = np.arange(Q)
        problem['h'].append(QQ[H])
    return problem

def generate_problems(ntypes, T, Q, nproblems):
    # set seed
    np.random.seed(808)

    problem_dir = 'problems'
    # if problem_dir does not exist, create it
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    file_name = '{}/problems_{}_{}_{}_{}.pkl'.format(problem_dir, ntypes, T, Q, nproblems)

    # Generate problems
    problems = []
    for i in range(nproblems):
        # Add problem to list
        problems.append(generate_problem(ntypes, T, Q))
    
    # Save problems
    with open(file_name, 'wb') as f:
        pickle.dump(problems, f)

def main():
    # create input parameters T, Q, ntypes, nproblems using command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--Q', type=int, default=10)
    parser.add_argument('--ntypes', type=int, default=10)
    parser.add_argument('--nproblems', type=int, default=100)
    args = parser.parse_args()
    T = args.T
    Q = args.Q
    ntypes = args.ntypes
    nproblems = args.nproblems
    generate_problems(ntypes, T, Q, nproblems)

if __name__ == '__main__':
    main()


    
    #ntypes, T, Q = 10, 10, 10
    # Generate problems
    # for n_types in range(3, 6):
    #     for T in range(2, 11):
    #         problems = []
    #         file_name = '{}/problems_{}_{}_{}.pkl'.format(problem_dir, n_types, T, 100)
    #         for i in range(nproblems):
    #             # Add problem to list
    #             problems.append(generate_problem(n_types, T, 100))
            
    #         # Save problems
    #         with open(file_name, 'wb') as f:
    #             pickle.dump(problems, f)

    # #ntypes, T, Q = 10, 10, 10
    # # Generate problems
    # problems = []
    # for i in range(nproblems):
    #     # Add problem to list
    #     problems.append(generate_problem(ntypes, T, Q))
    
    # # Save problems
    # with open(file_name, 'wb') as f:
    #     pickle.dump(problems, f)

'''
Example usage:
with open('problems_10_10_10.pkl', 'rb') as f:
    problems = pickle.load(f)
print(problems[0])
'''