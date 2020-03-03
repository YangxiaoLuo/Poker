import sys
import time
#from poker_env.ToyPoker.data.eval_potential import sample_data
from poker_env.Texas.data.eval_potential import sample_data

if __name__ == "__main__":

    t0 = time.time()
    # sample_flop_data(1)
    sample_num = sys.argv[1]
    subprocess_num = sys.argv[2]
    run_function = sys.argv[3]
    if not (sample_num.isdecimal() and subprocess_num.isdecimal()):
        raise TypeError('argv must be integers.')
    sample_data(run_function, int(sample_num), int(subprocess_num))
    print("Time cost:{}".format(time.time() - t0))
