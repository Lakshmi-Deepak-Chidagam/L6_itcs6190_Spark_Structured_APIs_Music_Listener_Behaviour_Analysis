# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Create Spark Session
spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# --------------------------------
# Load datasets
# --------------------------------

listening_df = spark.read.csv(
    "listening_logs.csv",
    header=True,
    inferSchema=True
)

songs_df = spark.read.csv(
    "Songs_metadata.csv",
    header=True,
    inferSchema=True
)

# Convert timestamp column to timestamp type
listening_df = listening_df.withColumn(
    "timestamp",
    to_timestamp("timestamp")
)

# Join both datasets
music_df = listening_df.join(songs_df, "song_id", "inner")

# --------------------------------
# Task 1: User Favorite Genres
# (Most listened genre per user)
# --------------------------------

genre_count = music_df.groupBy("user_id", "genre") \
    .count()

window_spec = Window.partitionBy("user_id").orderBy(desc("count"))

favorite_genre = genre_count.withColumn(
    "rank",
    rank().over(window_spec)
).filter(col("rank") == 1)

print("===== Task 1: User Favorite Genres =====")
favorite_genre.select("user_id", "genre", "count").show()


# --------------------------------
# Task 2: Average Listen Time per User
# --------------------------------

avg_listen_time = music_df.groupBy("user_id") \
    .agg(avg("duration_sec").alias("avg_duration"))

print("===== Task 2: Average Listen Time =====")
avg_listen_time.show()


# --------------------------------
# Task 3: Genre Loyalty Score
# (Total listens per genre per user)
# Rank and show Top 10
# --------------------------------

genre_loyalty = music_df.groupBy("user_id", "genre") \
    .count() \
    .withColumnRenamed("count", "loyalty_score")

loyalty_window = Window.orderBy(desc("loyalty_score"))

ranked_loyalty = genre_loyalty.withColumn(
    "rank",
    dense_rank().over(loyalty_window)
)

top10_loyalty = ranked_loyalty.filter(col("rank") <= 10)

print("===== Task 3: Top 10 Genre Loyalty Scores =====")
top10_loyalty.show()


# --------------------------------
# Task 4: Users Listening Between 12 AM and 5 AM
# --------------------------------

music_df = music_df.withColumn(
    "hour",
    hour("timestamp")
)

late_night_users = music_df.filter(
    (col("hour") >= 0) & (col("hour") < 5)
).select("user_id").distinct()

print("===== Task 4: Late Night Users (12 AM - 5 AM) =====")
late_night_users.show()


# Stop Spark Session
spark.stop()