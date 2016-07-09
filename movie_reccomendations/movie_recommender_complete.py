import sys, os
import math
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel


def parseRatingLine(line):
    line = line.split(',')
    return int(line[0]), int(line[1]), float(line[2])


def parseMovieLine(line):
    line = line.split(',')
    return int(line[0]), line[1]


def run():

    conf = SparkConf().setMaster("local").setAppName("mr")
    sc = SparkContext(conf=conf)


    # Load the data
    ratings_complete_raw = sc.textFile("../ratings_mini.csv")
    ratings_complete_header = ratings_complete_raw.take(1)[0]
    ratings_complete = ratings_complete_raw.filter(lambda line: line != ratings_complete_header) \
                                           .map(parseRatingLine).cache()
    num_ratings = ratings_complete.count()

    print("We are processing " + str(num_ratings) + " ratings!")

    training, test = ratings_complete.randomSplit([7, 3], seed=0L)
    test_no_ratings = test.map(lambda x: (x[0], x[1]))

    rank = 2
    model = ALS.train(training, rank, iterations=10, lambda_=0.1, seed=5L)

    predictions = model.predictAll(test_no_ratings).map(lambda r: ((r.user, r.product), r.rating))

    ratings_and_preds = test.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)

    error = math.sqrt(ratings_and_preds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
    print("RMSE on test data: " + str(error))

    model.save(sc, os.path.join('..', 'models', 'movielens_als_model'))
    print("ALS model saved to models/movielens_als_model")



    movies_complete_raw = sc.textFile("../movies_mini.csv")
    movies_complete_header = movies_complete_raw.take(1)[0]
    movies_complete = movies_complete_raw.filter(lambda line: line != movies_complete_header) \
                                         .map(parseMovieLine).cache()
    num_movies = movies_complete.count()

    # movies_complete_titles = movies_complete.map(lambda x: (int(x[0]), x[1]))

    def getCountAndAvgRating(id_and_ratings):
        num_ratings = len(id_and_ratings)
        return id_and_ratings[0], (num_ratings, sum(id_and_ratings[1]))

    ratings_for_movie = ratings_complete.map(lambda x: (x[1], x[2])).groupByKey()
    avg_rating_and_count_for_movie = ratings_for_movie.map(getCountAndAvgRating)
    rating_counts_for_movie = avg_rating_and_count_for_movie.map(lambda x: (x[0], int(x[1][0])))

    new_user_id = 0
    new_user_ratings = [
        (0, 10, 4),
        (0, 1, 3),
        (0, 16, 3),
        (0, 25, 4),
        (0, 32, 4)
    ]

    new_data_rdd = sc.parallelize(new_user_ratings)
    updated_data_rdd = ratings_complete.union(new_data_rdd)
    new_user_rated_movies = new_data_rdd.map(lambda x: x[1]).collect()
    new_user_rated_titles = new_data_rdd.map(lambda x: (x[1], x[2])).join(movies_complete).map(lambda x: (x[1][1], (str(x[1][0]) + '*')))

    print("So you rated the following movies...\n%s") % new_user_rated_titles.take(5)

    new_model = ALS.train(updated_data_rdd, rank, iterations=10, lambda_=0.1, seed=5L)

    new_user_unrated_movies = movies_complete.filter(lambda x: x[0] not in new_user_rated_movies) \
                                             .map(lambda y: (new_user_id, y[0]))

    new_user_predictions = new_model.predictAll(new_user_unrated_movies).map(lambda x: (x.product, x.rating))

    new_user_recs_rating_title_count = new_user_predictions.join(movies_complete).join(rating_counts_for_movie)
    new_user_recs_rating_title_count = new_user_recs_rating_title_count.map(lambda x: (x[1][0][1], x[1][0][0], x[1][1]))

    recommendations = new_user_recs_rating_title_count.filter(lambda x: x[2] >= 2).takeOrdered(25, lambda x: -x[1])

    print("Our recommendations for you...\n" + '\n'.join(map(str, recommendations)))


if __name__ == '__main__':
    run()













