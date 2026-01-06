import os
import pickle

traj_root = os.getenv('NAVSIM_TRAJPDM_ROOT')

if __name__ == '__main__':
    out_dir = 'vocab_score_full_16384_navtrain_v2'
    ins = [f'navtrain_sub{i}_v2.pkl' for i in range(1, 33)]
    out = 'navtrain.pkl'

    result = {}
    for in_pkl in ins:
        curr_pickle = pickle.load(open(f'{traj_root}/{out_dir}/{in_pkl}', 'rb'))
        print(f'{traj_root}/{out_dir}/{in_pkl}', len(curr_pickle))
        for k, v in curr_pickle.items():
            result[k] = v
    print(f'Length: {len(result)}')
    pickle.dump(result, open(f'{traj_root}/{out_dir}/{out}', 'wb'))