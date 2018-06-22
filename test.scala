package UsingSpark.StandardLightLDAModel

import java.io.File

import UsingSpark.Testing.TrySparkLDA.search_dir
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
  * spark lda online，单机下运行时间54s
  * light lda，单机下运行时间118s
  */
object test {
  def get_df(spark: SparkSession): DataFrame = {
    val dir = new File(search_dir)
    var res_arr = Array[DataFrame]()
    if (dir.isDirectory) {
      val fl = dir.listFiles().take(10)
      for (file <- fl) {
        //        println(file)
        val df = spark.read
          .options(Map("header" -> "false", "delimiter" -> "\t"))
          .csv(file.getAbsolutePath)
          .toDF("file_name", "sentence")
        //        println(df.count())
        res_arr = res_arr :+ df
      }
    }
    val df = res_arr.reduce((ele1, ele2) => ele1.union(ele2))
    df
  }

  def test_trans_df(df: DataFrame, spark: SparkSession): Unit = {
    //--------------------------------------
    val d = new Documents
    d.trans_df_2_rdd(df, "sentence", 20)
    //    d.save_word_index()
    val model = new LightLDAModel(spark)
    model.init(d)
    model.fit()
  }

  def compare_spark_lda(input_df: DataFrame): Unit = {

    val split_udf = org.apache.spark.sql.functions.udf {
      sentence: String =>
        val chn_words = sentence.split(" ")
          .filter(ele => ele.length > 1) // && ele.length != ele.getBytes().length)
        chn_words
    }
    var df = input_df.withColumn("words", split_udf(input_df.col("sentence")))
    val cvModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(100000) //最大词的数量
      .setMinDF(5) //最少在多少片文本中出现
      .setMinTF(2) //单篇文本中出现次数
      .fit(df)
    df = cvModel.transform(df)
    df = df.repartition(20).persist(StorageLevel.MEMORY_AND_DISK)
    println("开始训练")
    val tmr_start = System.currentTimeMillis()
    val lda = new UsingSpark.testingLDA.ssLDA()
      .setK(10)
      //      .setTopicConcentration(3)
      //      .setDocConcentration(3)
      .setFeaturesCol("features")
      .setOptimizer("online")
      .setMaxIter(50)
    val model = lda.fit(df)
    val tmr_end = System.currentTimeMillis()
    println(s"训练模型，耗时${(tmr_end - tmr_start) / 1000.0}s")
  }

  def main(args: Array[String]): Unit = {
    val spconf = new SparkConf()
      .setMaster("local")
      .setAppName("an_app")
      .set("spark.port.maxRetries", "100")
    val spark = SparkSession.builder().config(spconf).getOrCreate()

    //    val d = new Documents
    //    d.docs = spark.sparkContext.parallelize(Seq((1, Array(2, 3, 4)), (0, Array(0, 4, 3, 2, 1))))
    //    d.word_index += ("死了" -> 1)
    //    d.word_index += ("丢你" -> 2)
    //    d.word_index += ("好烦" -> 3)
    //    d.word_index += ("啊" -> 4)
    //    d.word_index += ("摸了" -> 0)
    //    d.index_word = d.word_index.values.zip(d.word_index.keys).toMap
    //
    //    val model = new LightLDAModel(spark)
    //    model.init(d)
    //测试alias
    val as = new AliasSampler
    as.build(Array(2, 1, 1, 0))
    (for (i <- 0 until 10000) yield as.sample()).groupBy(line => line)
      .map(line => (line._1, line._2.size)).foreach(println)
    //测试模型的迭代
    //    model.fit()
    //测试从df进行转换
    test_trans_df(get_df(spark), spark)

    //    compare_spark_lda(get_df(spark))
  }
}
