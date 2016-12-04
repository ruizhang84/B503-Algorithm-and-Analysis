from hiddenMarkov import *
import time

# compare running time
if __name__ == '__main__' :
    initial, transition, emission = parse_parameters("sample_parameters_plus.dat") 
    time_s = []
    size_s = []
    
    #max size
    for i in range(1, 10):
        sequence = gen_sequence(i)
        size_s.append(i)

        #running time
        start_time = time.time()
        naive (initial, transition, emission, sequence)
        time_now = time.time() - start_time
        time_s.append(time_now)

    #write out time vs size
    f = open("time_state_4_naive.txt", "w")
    for i in range(len(size_s)):
        f.write("%d\t%.10f\n" %(size_s[i], time_s[i]))
