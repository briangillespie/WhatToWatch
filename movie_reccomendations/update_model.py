import sys, os
import math
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel



new_user_ratings = [
    (0, 260, 4),  # Star Wars (1977)
    (0, 1, 3),  # Toy Story (1995)
    (0, 16, 3),  # Casino (1995)
    (0, 25, 4),  # Leaving Las Vegas (1995)
    (0, 32, 4),  # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
    (0, 335, 1),  # Flintstones, The (1994)
    (0, 379, 1),  # Timecop (1994)
    (0, 296, 3),  # Pulp Fiction (1994)
    (0, 858, 5),  # Godfather, The (1972)
    (0, 50, 4)  # Usual Suspects, The (1995)
]


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)

model = MatrixFactorizationModel.load(sc, "..\models\movielens_als_model_mini")

