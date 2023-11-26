import time
import numpy as np
import importlib
import datetime

import lsh_search_dcosine
# for any reason dcs.py is not updated after editing
importlib.reload(lsh_search_dcosine)
from timeit import default_timer as timer
from lsh_search_dcosine import LSH_Search_RP_cosine
import random

result_pack = []

def lis_fill(lst):
    _ = random.randint(1000,5000)
    if len(lst)>3000:
        lst = random.sample(lst, _)
    return lst

def result_wrapper_dcosine(key, search_result):
        """search_result is a list @eg:[('0', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 1.0),(...)]
        """
        global result_pack
        key_num = int(key)
        search_result_new = []
        # print('length ',len(search_result))
        for i in search_result:
            search_result_new.append(int(i[0]))
            print(i[2]) #Here is the score
        sum_records = 0
        for item in search_result_new:
            if item > key_num:
                result_pack.append(str(key_num) + ', ' + str(item) + '\n')
                sum_records += 1
            elif item < key_num:
                # else:
                result_pack.append(str(item) + ', ' + str(key_num) + '\n')
                sum_records += 1
        return sum_records
    
def write_to_file(filepath):
    """write a list to file
    """
    global result_pack
    print('length ',len(result_pack))
    result_pack = list(set(result_pack))  # Remove identicals
    result_pack = lis_fill(result_pack)
    print('length ',len(result_pack))
    with open(filepath, "w") as f:
        sumlgth = len(result_pack)
        result_pack[-1] = result_pack[-1].replace('\n', '')  # remove the \n in the last line
        f.writelines(result_pack)
    
#from LSH_RP import LSH_Search_RP_cosine  # using the radom projection
def discrete_cosine(data, threshold, seed, filepath):
        raw_data = data
        random.seed(seed)
        np.random.seed(seed)
        movie_ids = 17770
        users_sum = 103703
        time_limit = 1800         # seconds
        print ('start ',datetime.datetime.now())
        start = timer()
        print ('start ',start)
       
        print('=' * 15)
        print('@ Data task: Netflix Assignment')
        print('@ Program title: RandomProjection-LSH Search Using discrete_cosine distance-func')
        print('Attention. This program will not run more than',time_limit,'seconds')
        print('=' * 15)

        feature_dims = 900  # This feature compares dims
        print ('LSH_Search_RP_cosine started')
        # lsh = LSH_Search_RP_cosine(10, feature_dims, 90)  # hash size, feature dims, hash tables
        lsh = LSH_Search_RP_cosine(10, feature_dims, 90)  # hash size, feature dims, hash tables
        _x_features = [i for i in range(movie_ids)]
        random.shuffle(_x_features)
        # print(_x_features)
        midx = np.array(_x_features, dtype=np.int64)[:feature_dims]  # Change feature idx to np.array
        # print(midx.shape)

        user_cols = raw_data  # Get only movie id
        cols = user_cols[:, :1].squeeze().squeeze()
        counts_index = np.bincount(cols)
        data_pack = {}
        base_i = 0
        for i in range(1, len(counts_index)):
            re = user_cols[base_i:base_i + counts_index[i], 1:3].squeeze().squeeze()  # .tolist()
            re_bank = np.zeros([movie_ids])
            for j in range(re.shape[0]):
                try:
                    re_bank[re[j, 0] - 1] = 1  # Just turn the mark bit to 1, others are zeros
                except:
                    pass
            re_bank = re_bank[midx].astype(np.int64)  # if used the float it will be not enough memory
            lsh.insert(str(i), re_bank)
            data_pack[str(i)] = re_bank
            base_i += counts_index[i]
            end = timer()
        print("Built HashData in: %.03f seconds" % (end - start))
        sum_records = 0
        print('threshold', threshold)
        found = 0
        for i in range(users_sum):
            x = data_pack[str(i + 1)]
            search_result = lsh.query(x, thresold=threshold)  # @eg:[(('0', (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 1.0),(...))]
            re_size = len(search_result) - 1
            if re_size > 0:
                try:
                  found = found + 1;
                  print('found ',found)
                  print('sum_record i ',i)
                  sum_records += result_wrapper_dcosine(str(i + 1), search_result)  # task2,3 should their own wrapper
                except:
                  print('exception with search_result at i ',i)
            if (timer() - start) > time_limit:  # @time watch dog                
                print ('start: ',start)
                print ('actual time: ',timer())
                print('time limit',timer() - start)
                break
        print('sum_records done')                
        print(sum_records)
        print('starting write file')
        write_to_file('dcs.txt')
        
        print ('end ',datetime.datetime.now())
        print("Query all pairs used run-time is : %.03f seconds" % (timer() - start))  #
        return

