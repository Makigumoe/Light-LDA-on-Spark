package UsingSpark.StandardLightLDAModel

import java.io._

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.linalg.{SparseVector => SPSV}
import org.apache.spark.mllib.linalg.{Vector => SPV, Vectors => SPVs}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel

/**
  * 用来存所有文档的数据结构
  */
class Documents {
  var docs: RDD[(Int, Array[Int])] = null
  var word_index = Map[String, Int]()
  var index_word = Map[Int, String]()

  def save_word_index(): Unit = {
    val path = "D:\\ProjectResources\\resources_new\\UsingSpark\\index_words.txt"
    val fw = new FileWriter(new File(path))
    for (line <- index_word)
      fw.write(line._1.toString + "\t" + line._2 + "\r\n")
    fw.close()
  }

  /**
    * 将dataframe转换为rdd的便捷函数。这个dataframe必须包含一列String类型，该列代表这已经分词并使用空格隔开的词
    *
    * @param input_df
    * @param sentence_col
    * @param partitions
    */
  def trans_df_2_rdd(input_df: DataFrame, sentence_col: String, partitions: Int): Unit = {
    try {
      val split_udf = org.apache.spark.sql.functions.udf {
        sentence: String =>
          val chn_words = sentence.split(" ")
            .filter(ele => ele.length > 1) // && ele.length != ele.getBytes().length)
          chn_words
      }
      var df = input_df.withColumn("words", split_udf(input_df.col(sentence_col)))
      val cvModel = new CountVectorizer()
        .setInputCol("words")
        .setOutputCol("features")
        .setVocabSize(100000) //最大词的数量
        .setMinDF(5) //最少在多少片文本中出现
        .setMinTF(2) //单篇文本中出现次数
        .fit(df)
      val vocab = cvModel.vocabulary
      val index2word = (for (i <- 0 until vocab.length) yield i).zip(vocab).toMap
      index2word.take(5).foreach(println)
      df = cvModel.transform(df)
      df.show(5)
      val ident_words_udf = org.apache.spark.sql.functions.udf {
        (words: SPSV) =>
          words.toArray.map(_.toInt)
            .zipWithIndex.filter(_._1 != 0).flatMap(line => for (i <- 0 until line._1) yield line._2).toSeq //让前面是词的序号，后面是词的数量
      }
      df = df.withColumn("ident_words", ident_words_udf(df.col("features")))
      df.printSchema()
      df.show(5)
      docs = df.rdd.map(_.getAs[Seq[Int]]("ident_words")).zipWithIndex()
        .map(line => (line._2.toInt, line._1.toArray))
        .repartition(partitions)
      docs.persist(StorageLevel.MEMORY_AND_DISK)
      docs.take(5).foreach {
        line =>
          println(line._1)
          println(line._2.map(_.toString()).reduce(_ + " " + _))
      }
      index_word = index2word
      word_index = index2word.values.zip(index2word.keys).toMap
      word_index.take(5).foreach(println)
      println(index_word(1588))
      println(word_index.size)
      println("success")
    }
    catch {
      case e: Exception =>
        println("failed")
    }
  }
}
