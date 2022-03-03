from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler



if __name__ == "__main__":

    spark = SparkSession.builder.master("local[*]").appName("appname").getOrCreate()


    # Load data
    df_raw = spark.read.option("delimiter", "\t").option("header", True).csv("data/data-final.csv")
    # 1015341 rows

    df_clearzero = df_raw\
        .withColumn("MINS", least("EXT1", "EXT2", "EXT3", "EXT4", "EXT5", "EXT6", "EXT7", "EXT8", "EXT9", "EXT10",
                                  "EST1", "EST2", "EST3", "EST4", "EST5", "EST6", "EST7", "EST8", "EST9", "EST10",
                                  "AGR1", "AGR2", "AGR3", "AGR4", "AGR5", "AGR6", "AGR7", "AGR8", "AGR9", "AGR10",
                                  "CSN1", "CSN2", "CSN3", "CSN4", "CSN5", "CSN6", "CSN7", "CSN8", "CSN9", "CSN10",
                                  "OPN1", "OPN2", "OPN3", "OPN4", "OPN5", "OPN6", "OPN7", "OPN8", "OPN9", "OPN10"))\
        .where(col("MINS") > 0)

    # Prepare data
    df_rates = df_clearzero\
        .withColumn("EXT", (col("EXT1") + (6-col("EXT2")) + col("EXT3") + (6-col("EXT4")) + col("EXT5") +
                            (6-col("EXT6")) + col("EXT7") + (6-col("EXT8")) + col("EXT9") + (6-col("EXT10")))/10)\
        .withColumn("EST", ((6-col("EST1")) + col("EST2") + (6-col("EST3")) + col("EST4") + (6-col("EST5")) +
                            (6-col("EST6")) + (6-col("EST7")) + (6-col("EST8")) + (6-col("EST9")) + (6-col("EST10")))/10)\
        .withColumn("AGR", ((6-col("AGR1")) + col("AGR2") + (6-col("AGR3")) + col("AGR4") + (6-col("AGR5")) +
                            col("AGR6") + (6-col("AGR7")) + col("AGR8") + col("AGR9") + col("AGR10"))/10)\
        .withColumn("CSN", (col("CSN1") + (6-col("CSN2")) + col("CSN3") + (6-col("CSN4")) + col("CSN5") +
                            (6-col("CSN6")) + col("CSN7") + (6-col("CSN8")) + col("CSN9") + col("CSN10"))/10)\
        .withColumn("OPN", (col("OPN1") + (6-col("OPN2")) + col("OPN3") + (6-col("OPN4")) + col("OPN5") +
                            (6-col("OPN6")) + col("OPN7") + col("OPN8") + col("OPN9") + col("OPN10"))/10)\
        .select(col("EXT"), col("EST"), col("AGR"), col("CSN"), col("OPN"))

    # Clean data WE NEED TO CLEAN 0 VALUES
    df_clean = df_rates.na.drop()

    assembler = VectorAssembler(
        inputCols=["EXT", "EST", "AGR", "CSN", "OPN"],
        outputCol="features")

    df = assembler.transform(df_clean)

    # Standarize features
    scale = StandardScaler(inputCol='features', outputCol='standardized')
    data_scale = scale.fit(df)
    df_scale = data_scale.transform(df)
    df_scale.show(truncate=0)

    # Training
    kmeans = KMeans(featuresCol="standardized").setK(4).setSeed(1)
    model = kmeans.fit(df_scale)

    # Make predictions
    predictions = model.transform(df_scale)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                    metricName='silhouette', distanceMeasure='squaredEuclidean')

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)


    predictions.coalesce(1).write.format("parquet").mode("overwrite").save("output/Clustering-KMeans5.parquet")
    predictions.show(truncate=0)


    # df.printSchema()

    print("END")
