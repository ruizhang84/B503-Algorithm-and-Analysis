from numpy import zeros, argmax, ones
import numpy.testing as npt
import random
import itertools
random.seed(0)

# function to parse the paraeters from a file

def parse_parameters(parameterFilePath) :
    initial, transition, emission = dict(), dict(), dict()
    
    ##read-in code
    try:
        # state num and alphabet size, set up
        state_num = 0
        alphabet_size = 0
        alphabet = set([])
        
        # open files
        f = open(parameterFilePath, 'r')
        
        # initial set up n, m
        # 2 integers n and m signifying number of states and alphabet size
        for line in f:
            a, b = line.strip().split()
            state_num = int (a)
            alphabet_size = int (b)
            # assert num is positive integer
            assert(state_num > 0 )
            assert(alphabet_size > 0)
            break
    
        # calculate number of line to process probabilities
        # n x n + (n)
        transition_line_num = state_num * state_num + state_num
        # n x m (+ n x n + 1)
        emission_line_num = state_num * alphabet_size + transition_line_num
    
        # process next probabilities
        lineRead = 0     # counter the number of line
        for line in f:
            # n lines of initial probabilities
            if lineRead < state_num:
                # read in parameters
                state, prob_str = line.strip().split()
                prob = float(prob_str)
                # assert positive integer
                assert(prob <= 1 and prob >= 0)
                
                # set up initial probability dict
                assert (state not in initial)
                initial[state] = prob
                lineRead += 1
            
            # n x n lines of transition probabilities
            elif lineRead < transition_line_num:
                # read in parameters
                state_start, state_end, prob_str = line.strip().split()
                prob = float(prob_str)
                # assert positive/zero integer
                assert(prob <= 1 and prob >= 0)
                
                # set up transition probability dic
                #print state_start, state_end
                assert ((state_start, state_end) not in transition)
                transition[(state_start, state_end)] = prob
                lineRead += 1
                #print transition
            
            # n x m lines of emission probabilities
            elif lineRead < emission_line_num:
                # read in parameters
                state, emitted, prob_str = line.strip().split()
                alphabet.add(emitted)
                prob = float(prob_str)
                # assert integer
                assert(prob <= 1 and prob >= 0)
                
                # set up emission probabilities dic
                assert((state, emitted) not in emission)
                emission[(state, emitted)] = prob
                lineRead += 1
            
            # additional input
            else:
                if (len(line.strip()) != 0):    # only white space is allowed
                    lineRead += 1

        # input line should stop at correct number of line
        assert (lineRead == emission_line_num)
        
        #print transition
        #print initial
        #print emission
        
        f.close()
    except:
        #print ("input error", sys.exc_info()[0])
        raise
    
    # assert integrity of transition and emission prob dict
    try:
        #initial prob
        check_initial = 0
        for state in initial:
            check_initial += initial[state]
        assert(abs(check_initial - 1)< 0.0000001)
        
        #transition prob
        for state_start in initial:
            check_sum = 0.0
            for state_end in initial:
                check_sum += transition[(state_start, state_end)]
            # assert transit sum = 1
            assert(abs(check_sum-1)<0.000001)
        #alphabet
        #print alphabet
        assert(len(alphabet) == alphabet_size)

        #emmission prob
        for state in initial:
            check_sum = 0
            for emitted in alphabet:
                check_sum += emission[(state, emitted)]
            # asserr emissmion sum equal 1
            #print check_sum, alphabet
            assert(abs(check_sum-1)<0.000001)
    except:
        raise


    return initial, transition, emission

# viterbi will return the most likely hiden state sequence.

def viterbi(initial, transition, emission, observed) :
    # initial str for hidden state
    hidden = ''

    # infer hidden states
    states = initial.keys()
    
    # initialize matrices
    score = zeros([len(states), len(observed)])             # score, n x m matrices
    trace = zeros([len(states), len(observed)], dtype=int)  # trace[current_state][at observed] = prev_state
    
    # forward pass
    for i, obs in enumerate (observed) :                    # num, emitted alphabet,
        for j, st in enumerate (states) :                   # num, state
            ### Fill the score and trace matrices
            if i == 0 :
                # score for all state in start observed alphabet [j->state][i->0]
                score[j][i] = initial[st] * emission[(st, obs)]
            else :
                # find the maxium score for each state at i observed
                # record the trace at [state, Oi] for traceback where it comes from
                max_pass = 0                                                                            # max probably pass through from i-1
                for k, prev_st in enumerate (states):                                                   # for loop on all states
                    if max_pass <= score[k][i-1] * transition[(prev_st, st)]:
                        max_pass = score[k][i-1] * transition[(prev_st, st)]                            # update max prob
                        trace[j][i] = k                                                                 # record i-1 state k
                
                for k, prev_st in enumerate (states):
                    if score[j][i] <= score[k][i-1] * transition[(prev_st, st)] * emission[(st, obs)]:  # score calc from i-1 node
                        score[j][i] = score[k][i-1] * transition[(prev_st, st)] * emission[(st, obs)];  # max score at current state with observed

    # trace back
    z = argmax(score[:,-1])
    hidden = states[z]
    

    # reverse prob
    for i in range(1,len(observed))[::-1] :
        z = trace[z,i]
        hidden = states[z] + hidden                             # back tracking string one by one
    
    return hidden


# naive brute force algorithm
def naive(initial, transition, emission, observed) :
    # initial str for hidden state
    hidden = ''
    
    # infer hidden states
    states = initial.keys()
    #see python document itertools.product
    comb = itertools.product(states, repeat = len(observed))
    
    max = 0
    temp_hidden = []
    temp_prob = 0
    # calculate max prob
    for seq in comb:
        for i in range(len(observed)):
            if i == 0:
                temp_prob = initial[seq[i]] * emission[(seq[i], observed[i])]
            else:
                temp_prob *= transition[(seq[i-1], seq[i])]* emission[(seq[i], observed[i])]
        
        # record max
        if temp_prob >= max:
            max = temp_prob
            temp_hidden = seq
    
    hidden = "".join(temp_hidden)
    
    return hidden


#  random sequence generator
def gen_sequence(length) :
    protein = ["R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "G", "P", "A", "V", "I", "L", "M", "F", "Y", "W"]
    seq_string = ''
    
    for i in range(length):
        seq_string += random.choice(protein)

    return seq_string

# viterbi will slight modiftion to skip unessary calculation return the most likely hiden state sequence.
def viterbi_improve(initial, transition, emission, score, trace, observed, pre_observed) :
    # initial str for hidden state
    hidden = ''
    
    # infer hidden states
    states = initial.keys()
    
    # initialize matrices
    #score = zeros([len(states), len(observed)])             # score, n x m matrices
    #trace = zeros([len(states), len(observed)], dtype=int)  # trace[current_state][at observed] = prev_state
    
    #difference aftwards
    diff = 0
    cut = len(pre_observed)
    
    # forward pass
    for i, obs in enumerate (observed) :                    # num, emitted alphabet
        if i < cut and pre_observed[i] == observed[i] and diff == 0:
            continue
        else:
            diff = 1
        
        for j, st in enumerate (states) :                   # num, state
            ### Fill the score and trace matrices
            if i == 0 :
                # score for all state in start observed alphabet [j->state][i->0]
                score[j][i] = initial[st] * emission[(st, obs)]
            else :
                # find the maxium score for each state at i observed
                # record the trace at [state, Oi] for traceback where it comes from
                max_pass = 0                                                                            # max probably pass through from i-1
                for k, prev_st in enumerate (states):                                                   # for loop on all states
                    if max_pass <= score[k][i-1] * transition[(prev_st, st)]:
                        max_pass = score[k][i-1] * transition[(prev_st, st)]                            # update max prob
                        trace[j][i] = k                                                                 # record i-1 state k
                
                for k, prev_st in enumerate (states):
                    if score[j][i] <= score[k][i-1] * transition[(prev_st, st)] * emission[(st, obs)]:  # score calc from i-1 node
                        score[j][i] = score[k][i-1] * transition[(prev_st, st)] * emission[(st, obs)];  # max score at current state with observed

    # trace back
    z = argmax(score[:,-1])
    hidden = states[z]
    
    
    # reverse prob
    for i in range(1,len(observed))[::-1] :
        z = trace[z,i]
        hidden = states[z] + hidden                             # back tracking string one by one
    
    return (hidden, score, trace)



# compare running time
if __name__ == '__main__' :
    sequence = gen_sequence(100)
    initial, transition, emission = parse_parameters("sample_parameters.dat")
    
    #print (viterbi (initial, transition, emission, sequence))
    #print (naive (initial, transition, emission, sequence))

    score = zeros([len(initial.keys()), len(sequence)])
    trace = zeros([len(initial.keys()), len(sequence)], dtype=int)
    hidden, score, trace = viterbi_improve (initial, transition, emission, score, trace, sequence, " ")
    #print (hidden)
    hidden, score, trace = viterbi_improve (initial, transition, emission, score, trace, sequence, sequence)
    #print (hidden)

