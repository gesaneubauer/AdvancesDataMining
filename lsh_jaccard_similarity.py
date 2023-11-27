import numpy as np
from scipy.sparse import csr_matrix
import time
import random


def transform_data(datafile):
    data = np.load(datafile)
    user_id, movie_id, ratings = data[:, 0], data[:, 1], data[:, 2]

    user_indices = user_id - 1
    movie_indices = movie_id - 1

    ratings_sparse = csr_matrix((ratings, (user_indices, movie_indices)))

    return ratings_sparse

def generate_hash_functions(num_hashes, max_value, seed):
    np.random.seed(seed)
    return [(np.random.randint(1, max_value, dtype=np.uint64), np.random.randint(1, max_value, dtype=np.uint64)) for _ in range(num_hashes)]

def minhash_signature(user_ratings, max_value, hash_functions):
    signature = np.inf * np.ones(len(hash_functions), dtype=np.uint64)
    for i, hash_function in enumerate(hash_functions):
        h = (hash_function[0] * user_ratings + hash_function[1]) % max_value
        signature[i] = np.min(h)
    return signature

def minhash_band(users, bands, rows):
    buckets = [{} for _ in range(bands)]
    for band in range(bands):
        start = band * rows
        end = (band + 1) * rows
        band_signatures = np.vstack([user_signature[start:end] for user_signature in users.values()])
        band_hashes = np.apply_along_axis(lambda x: hash(tuple(x)), 1, band_signatures)
        for i, h in enumerate(band_hashes):
            if h not in buckets[band]:
                buckets[band][h] = []
            buckets[band][h].append(i)
    return buckets

def get_keys(hash_tables):
  keys_with_multiple_values = []
  for i in range(len(hash_tables)):
    hash_table = hash_tables[i]
    for _, value in hash_table.items():
      if len(value) > 1:
        keys_with_multiple_values.append((value))

  return keys_with_multiple_values

def calc_jaccard_similarity(set1, set2):
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size

def jaccard_similarity(datafile, bands=20, signature=160, threshold=0.5, seed=2023):
    print(f"Starting LSH with Jaccard similarity with signatures of size {signature} segmented into {bands} bands")
    start_time = time.time()

    open("js.txt", "w")
    ratings = transform_data(datafile)
    max_value = ratings.shape[1]
    hash_functions = generate_hash_functions(signature, max_value, seed)

    users = {}
    print("GETTTING SIGNATURES")
    for user_index, user_ratings in enumerate(ratings):
        rated_movies = user_ratings.indices
        users[user_index] = minhash_signature(rated_movies, max_value, hash_functions)

    buckets = minhash_band(users, bands, signature // bands)
    hash_indices = get_keys(buckets)
    print("number of buckets to be reviewed", len(hash_indices))

    similar_pairs = set()
    k = 0
    for indices in hash_indices:
        if (time.time() - start_time) >= 1770: #abort run when about to exceed time limit
            print("task aborted, about to exceed time limit")
            break
        if k % 1000 == 0:
            print(k, "buckets compared,", "%s seconds passed" % (round(time.time() - start_time)))
        if len(indices) >= 10000:
            continue
        k += 1
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                user1 = indices[i]
                user2 = indices[j]
                similarity = calc_jaccard_similarity(set(ratings[user1].indices), set(ratings[user2].indices))
                if similarity >= threshold:
                    user_pair = (user1 + 1, user2 + 1)
                    if user_pair not in similar_pairs:
                        similar_pairs.add(user_pair)
                        # Write similar user pairs to a text file
                        with open("js.txt", "a") as output_file:
                            for user_pair in similar_pairs:
                                output_file.write(f"{user_pair[0]}, {user_pair[1]}\n")

    print("Program ran for %s seconds" % (round(time.time() - start_time)))
    print(len(similar_pairs), "similar users found")

