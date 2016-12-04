from hiddenMarkov import *
import time

# compare running time
if __name__ == '__main__' :
    initial, transition, emission = parse_parameters("sample_parameters.dat") 
    time_s = []
    size_s = []
    score = zeros([len(initial.keys()), 30])
    trace = zeros([len(initial.keys()), 30], dtype=int)
    old_sequence = " "
    sequence = gen_sequence(10)
    
    #max size
    for i in range(0, 20):
        new_sequence = sequence + gen_sequence(i)
        size_s.append(10+i)

        #running time
        start_time = time.time()
        hidden, score, trace = viterbi_improve (initial, transition, emission, score, trace, new_sequence, old_sequence)
        time_now = time.time() - start_time
        time_s.append(time_now)

        old_sequence = new_sequence

    #write out time vs size
    f = open("best_improve.txt", "w")
    for i in range(len(size_s)):
        f.write("%d\t%.10f\n" %(size_s[i], time_s[i]))
