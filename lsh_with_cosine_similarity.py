import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time

#transform data in that has columns: user, movie, rating into a compressed sparse row matrix. Each row is a user
def transform_data(datafile):
  data = np.load(datafile)
  user_id, movie_id, ratings = data[:, 0], data[:, 1], data[:, 2]

  user_indices = user_id -1
  movie_indices = movie_id -1

  ratings_sparse = csr_matrix((ratings, (user_indices, movie_indices)))

  return ratings_sparse

#create random projections and form hash code based on if a point is below or above a random projection
def random_project(hyperplanes, data):
  dim =  data.shape[1]
  planes = np.random.rand(dim, hyperplanes) - 0.5
  x = data.dot(planes)
  x = [[(-1) if j<0 else 1 for j in i] for i in x]

  return np.array(x)

#segment the data into bands and create a hash table for each band, add the hash segments to the hash tables return the tables
def buckets(hash_arrays, b):
  hash_tables = [{} for _ in range(b)]

  for j in range(len(hash_arrays)):
    hash_array = hash_arrays[j]
    r = np.round(len(hash_array) / b)
    r = r.astype(int)
    for i in range(b):
      start = int(i * r)
      end = int(start + r if i < b - 1 else len(hash_array))

      segment = tuple(hash_array[start:end])
      if i < len(hash_tables):
        if segment not in hash_tables[i]:
            hash_tables[i][segment] = []
        hash_tables[i][segment].append(j)

  return hash_tables

#get the indices of the original data of hash strings that occur in the same buckets return a list of lists of indices that occur in the same bucket
def get_keys(hash_tables):
  keys_with_multiple_values = []
  for i in range(len(hash_tables)):
    hash_table = hash_tables[i]
    for key, value in hash_table.items():
      if len(value) > 1:
        keys_with_multiple_values.append((value))

  return keys_with_multiple_values

#Calculate the cosine similarity of each user of which a part of its hash string occured in the same bucket
def calc_cosine_similarity(datafile, bands = 10, signature = 100, threshold=0.73, seed=2023):
  print("Starting locality sensitive hashing with cosine similarity")
  random.seed(seed)
  start_time = time.time()

  ratings = transform_data(datafile)
  hash = random_project(signature, ratings)
  hash_tables = buckets(hash, bands)
  hash_indices = get_keys(hash_tables)

  print("number of buckets to be compared:", len(hash_indices))

  open("cs.txt", "w")
  similar_pairs = set()
  k = 0
  for indices in hash_indices:
    if k % 1000 == 0:
      print(k, "buckets compared")
    if len(indices) >= 10000:
      continue
    k += 1
    sparse_vectors = ratings[indices, :]

    similarity_matrix = cosine_similarity(sparse_vectors) #calculate cosine similarity
    angles = np.arccos(similarity_matrix) * (180 / np.pi) #calculate angle and convert 180 to radians
    degree_similarity = 1 - (angles / 180) #calculate cosine similarity

    #set lower triangle including main diagonal of matrix to 0 to filter out symmetric pairs
    similarity_matrix = np.triu(degree_similarity, k = 1)
    #Finding indices where similarity exceeds the threshold
    indices_above_threshold = np.argwhere(similarity_matrix > threshold)

    # Adding pairs to the set
    for i, j in indices_above_threshold:
      user_pair = (indices[i] + 1, indices[j] + 1)
      if user_pair not in similar_pairs:  # Check if pair is not already written
        similar_pairs.add(user_pair)
        with open("cs.txt", "a") as output_file:  # Open file in append mode
          output_file.write(f"{user_pair}\n")  # Write the pair to the file

  print("program ran for %s seconds" % (time.time() - start_time))
  print(len(similar_pairs), "similar users found")
