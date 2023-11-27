import importlib
import numpy as np
import argparse

import dcs
# for any reason dcs.py is not updated after editing
importlib.reload(dcs)
from dcs import discrete_cosine
from lsh_with_cosine_similarity import calc_cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description='argument to run experiment')
    parser.add_argument('-d', type=str, default='/',
                        help="Data file path")
    parser.add_argument('-s', type=int, default=2023,
                        help='Random seed')
    parser.add_argument('-m', type=str, default='js',
                        help='Similarity measure (js / cs / dcs)')
    args = parser.parse_args()
    return args
#[d,user_movie_rating.npy]

def main():
    global result
    a = 'started'
    print(a)
    args = parse_args()
    a = 'parameter scan successful'
    print(a)
    print(args.m)   
    data = np.load(args.d)
    print(data)
    
    if args.m == 'dcs':
         result = discrete_cosine(data, 0.73, args.s,args.d)
#         result = data
#    elif args.m == 'js':
#        result = jaccard(0.5)
    elif args.m == 'cs':
        result = calc_cosine_similarity(datafile = args.d, bands = 11, signature = 110, threshold = 0.73, seed = args.s)



if __name__ == "__main__":
    main()  
    '''
    @ Usage

    #eg: python main.py -d D:/user_movie_rating.npy  -s 2020 -m dcs
    
    @ Prerequisites
    
    !pip install numpy
    
    '''