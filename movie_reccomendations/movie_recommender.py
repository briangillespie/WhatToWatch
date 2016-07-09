import sys, os
import math
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)


def parseRatingLine(line, delimiter='\t'):
    line = line.split(delimiter)
    return line[0], line[1], line[2]


def parseMovieLine(line, delimiter='\t'):
    line = line.split(delimiter)
    return line[0], line[1]

# Load the data
ratings = sc.textFile("../ml-100k/u.data").map(parseRatingLine).cache()
movies = sc.textFile("../ml-100k/u.item").map(parseMovieLine).cache()

# Split into training, validation, and testing
training, validation, test = ratings.randomSplit([6, 2, 2], seed=0L)
valid_no_rating = validation.map(lambda x: (x[0], x[1]))
test_no_rating = test.map(lambda x: (x[0], x[1]))

# Remove ratings from validation and test
# validation = validation.map(lambda x: (x[0]. x[1]))
# test = test.map(lambda x: (x[0]. x[1]))

ranks = [2, 4, 8, 10, 12]

least_error = float('inf')
best_rank = 0

for rank in ranks:

    cf_model = ALS.train(training, rank, iterations=10, lambda_=0.1, seed=5L)
    predictions = cf_model.predictAll(valid_no_rating).map(lambda r: ((r.user, r.product), r.rating))
    ratings_and_preds = validation.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)

    rmse = math.sqrt(ratings_and_preds.map(lambda x: (x[1][0] - x[1][1])**2).mean())

    if rmse < least_error:
        least_error = rmse
        best_rank = rank

    print("RMSE for rank " + str(rank) + ": " + str(rmse))

print("Best performance for rank " + str(best_rank) + " with RMSE = " + str(least_error))

cf_model = ALS.train(training, best_rank, iterations=10, lambda_=0.1, seed=5L)

predictions = cf_model.predictAll(test_no_rating).map(lambda r: ((r.user, r.product), r.rating))
ratings_and_preds = test.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)

cf_model.save(sc, os.path.join('..', 'models', 'movielens_als_model_mini'))

error = math.sqrt(ratings_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print("For the final model, the best rmse was: " + str(error))






