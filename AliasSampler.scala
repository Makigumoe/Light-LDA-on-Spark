package UsingSpark.StandardLightLDAModel

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * 构造alias采样表。提供了两个方案：dense和sparse
  * 在这里，由于实数精度所导致的问题有很多，不过也许都一一修复了
  */
class AliasSampler extends Serializable {

  private var alias = ArrayBuffer[(Int, Int, Double)]()

  def build(posibilities: Array[Double]): AliasSampler = {
    build_sparse(posibilities, posibilities.zipWithIndex.map(_._2))
    this
  }

  def build_sparse(posibilities: Array[Double], indexs: Array[Int]): AliasSampler = {
    //    println("posibilities: " + posibilities.map(_.toString).reduce(_ + " " + _))
    //    println(posibilities.length)

    val sum = posibilities.sum
    val per_max = sum / posibilities.length
    var arr_big = ArrayBuffer[(Int, Double)]()
    var arr_small = ArrayBuffer[(Int, Double)]()
    for ((ele, ind) <- posibilities.zip(indexs))
      if (ele < per_max)
        arr_small += ((ind, ele))
      else arr_big += ((ind, ele))
    var alias = ArrayBuffer[(Int, Int, Double)]()
    while ((arr_small.nonEmpty || arr_big.nonEmpty) && alias.length < posibilities.length) {
      if (arr_small.nonEmpty)
        if (alias.length == posibilities.length - 1 && arr_big.isEmpty) {
          val small = arr_small.head
          arr_small -= small
          alias += ((small._1, small._1, small._2))
        }
        else {
          val small = arr_small.head
          arr_small -= small
          val big = arr_big.head
          arr_big -= big
          alias += ((small._1, big._1, small._2 / per_max))
          val remain = big._2 + small._2 - per_max
          if (remain > 0)
            if (remain < per_max)
              arr_small += ((big._1, remain))
            else
              arr_big += ((big._1, remain))
        }
      else {
        val big = arr_big.head
        arr_big -= big
        alias += ((big._1, big._1, 1))
        val remain = big._2 - per_max
        if (remain > 0)
          if (remain < per_max)
            arr_small += ((big._1, remain))
          else
            arr_big += ((big._1, remain))
      }
    }
    this.alias = alias
    this
  }

  def sample(): Int = {
    val ran = new Random()
    val k = ran.nextInt(alias.length)
    val res = if (ran.nextDouble() > alias(k)._3) alias(k)._2 else alias(k)._1
    res
  }

  def sample_test(): String = {
    val arr = for (i <- 0 until 1000) yield sample()
    val count_arr = arr.groupBy(ele => ele).map(line => (line._1, line._2.size))
      .toArray.sortWith((ele1, ele2) => ele1._1 < ele2._1)
    val show_str = count_arr.map(_._2.toString).reduce(_ + " " + _)
    show_str
  }
}
