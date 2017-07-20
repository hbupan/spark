/*
use the textFileStream as input streaming data
 */

package com.microsoft.nao.sparkapps


import com.microsoft.CNTK._
import com.microsoft.ml.spark.CNTKModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import org.apache.spark.streaming.dstream.ConstantInputDStream


class Validation(cntkModel: CNTKModel, vocabulary: mutable.HashMap[String, Int]) extends Serializable {

  val DSSMEpsilon = 0.00000001f

  @transient
  private var model: Function = null

  def initModel(): Unit = {
    model = CNTKModel.loadModelFromBytes(cntkModel.getModel)
  }

  def cosineDistance(query: String, doc: String): Float = {
    val arguments = model.getArguments
    var queryInput = arguments.get(0)
    var documentInput = arguments.get(0)
    for (i <- 0 until arguments.size()) {
      if (arguments.get(i).getName == "query") queryInput = arguments.get(i)
      if (arguments.get(i).getName == "positive_document") documentInput = arguments.get(i)
    }

    val queryInputShape = queryInput.getShape
    val documentInputShape = documentInput.getShape

    val cosDistance: com.microsoft.CNTK.Function = model.findByName("query_positive_document_cosine_distance")
    val queryRecurrenceOutput = cosDistance.getArguments.get(0)
    val docRecurrenceOutput = cosDistance.getArguments.get(1)
    val queryDocCosineDistance = cosDistance.getOutput

    val queryInputValue = createSequenceInput(queryInputShape, query)
    val docInputValue = createSequenceInput(documentInputShape, doc)

    val inputs = new UnorderedMapVariableValuePtr()
    inputs.add(queryInput, queryInputValue)
    inputs.add(documentInput, docInputValue)

    val outputs = new UnorderedMapVariableValuePtr()
    outputs.add(queryRecurrenceOutput, null)
    outputs.add(docRecurrenceOutput, null)
    outputs.add(queryDocCosineDistance, null)

    model.evaluate(inputs, outputs)

    val queryEmbeddingOutput = new FloatVectorVector()
    val docEmbeddingOutput = new FloatVectorVector()
    outputs.getitem(queryRecurrenceOutput).copyVariableValueToFloat(queryRecurrenceOutput, queryEmbeddingOutput)
    outputs.getitem(docRecurrenceOutput).copyVariableValueToFloat(docRecurrenceOutput, docEmbeddingOutput)

    cosine(queryEmbeddingOutput.get(0), docEmbeddingOutput.get(0))
  }

  private def cosine(src: FloatVector, target: FloatVector): Float = {
    var s = 0.0f
    var t = 0.0f
    var mt = 0.0f
    for (i <- 0 until src.size().toInt) {
      s += src.get(i) * src.get(i)
      t += target.get(i) * target.get(i)
      mt += src.get(i) * target.get(i)
    }

    mt / math.sqrt(s * t).toFloat + DSSMEpsilon
  }

  private def createSequenceInput(shape: NDShape, s: String): Value = {
    val rg1 = "[^0-9a-z]".r
    val rg2 = "\\s+".r

    val ss = s.toLowerCase()
    rg1.replaceAllIn(ss, " ")
    rg2.replaceAllIn(ss, " ")
    val words = ss.trim.split(" ")

    val floatVector = new FloatVector()

    for (w <- words) {
      val vector = new Array[Float](49293)
      val encode = encodeWord2Letter3Gram(w)
      for ((key, value) <- encode) {
        vector(key) = value.toFloat
      }
      for (v <- vector)
        floatVector.add(v)
    }

    Value.createSequenceFloat(shape, floatVector, true, DeviceDescriptor.useDefaultDevice())
  }

  private def encodeWord2Letter3Gram(word: String): mutable.HashMap[Int, Int] = {
    val letterGram = new mutable.HashMap[String, Int]()
    val src = s"#$word#"
    for (i <- 0 to src.length - 3) {
      val l3g = src.substring(i, i + 3)
      if (letterGram.contains(l3g)) {
        letterGram(l3g) += 1
      } else letterGram.put(l3g, 1)
    }

    val encodeByVocab = new mutable.HashMap[Int, Int]()

    for (g <- letterGram) {
      if (vocabulary.get(g._1).isDefined) encodeByVocab.put(vocabulary(g._1), g._2)
    }

    encodeByVocab
  }
}

object MmlSparkStreamingTest {

  def main(args : Array[String]): Unit = {
    //capture the parameter
    if (args.length < 4) {
      println("Usage: com.microsoft.nao.sparkapps.NewsInsightBatch <input samples> <model file> <vocabulary file> <output File>")
      println("BN2's input samples path: /data/cntk/selected-samples.txt")
      println("BN2's model file path: /data/cntk/model1.dnn which is actually \\\\stchost-50\\ml\\cntk\\sample\\NewsInsight\\trained_model\\cntk_2_0_6_layer_hinge_loss_batch_1024_2016-01-01_2017-05-31_2017_06_23_03_37_50_model_batch_1400000_90850437.dnn")
      println("BN2's vocabulary file: /data/cntk/src.l3g.txt")
      System.exit(1)
    }

    val inputSamples = args(0)
    val modelFile = args(1)
    val vocabularyFile = args(2)
    val outputFile = args(3)

    val sparkConf = new SparkConf().setAppName("MmlSpark Streaming Test")
    val ssc = new StreamingContext(sparkConf, Seconds(2))

    val spark = SparkSession.builder.config(sparkConf).getOrCreate()
    val cntkModel = new CNTKModel().setModel(spark, modelFile)


    val vocabularyData = spark.sparkContext.textFile(vocabularyFile).collect()
    val vocabulary = new mutable.HashMap[String, Int]()
    for ((v, i) <- vocabularyData.zipWithIndex) {
      vocabulary.put(v, i)
    }
    val rawData = spark.sparkContext.textFile(inputSamples)

    //applied ConstantInputDStream to obtain the fake stream
    var rawDataStream = new ConstantInputDStream(ssc, rawData)
    val validation = new Validation(cntkModel, vocabulary)

    //use accum as a count for the input stream
    val accum = spark.sparkContext.longAccumulator("Accumulator")

    rawDataStream.foreachRDD { (rdd:RDD[String]) =>
      accum.add(1)
      rdd.map(s => {
        val info = s.split("\t")
        (info(0), info(1), info(2))
      }).mapPartitions { iter =>
        // To avoid not serializable problem, init CNTK model in executor side
        validation.initModel()
        iter.map {
          case (query, positive, negative) =>
            val posCos = validation.cosineDistance(query, positive)
            val negCos = validation.cosineDistance(query, negative)
            s"$posCos\t$query\t$positive\n$negCos\t$query\t$negative"
        }
        //avoid overwrite the reault file
      }.saveAsTextFile(outputFile + accum.value)
    }
    ssc.start()
    ssc.awaitTermination()
  }

  private def createSequenceInput(shape: NDShape, s: String, vocabulary: mutable.HashMap[String, Int]): Value = {
    val rg1 = "[^0-9a-z]".r
    val rg2 = "\\s+".r

    val ss = s.toLowerCase()
    rg1.replaceAllIn(ss, " ")
    rg2.replaceAllIn(ss, " ")
    val words = ss.trim.split(" ")

    val floatVector = new FloatVector()

    for (w <- words) {
      val vector = new Array[Float](49293)
      val encode = encodeWord2Letter3Gram(w, vocabulary)
      for ((key, value) <- encode) {
        vector(key) = value.toFloat
      }
      for (v <- vector)
        floatVector.add(v)
    }

    Value.createSequenceFloat(shape, floatVector, true, DeviceDescriptor.useDefaultDevice())
  }

  private def encodeWord2Letter3Gram(word: String, vocabulary: mutable.HashMap[String, Int]): mutable.HashMap[Int, Int] = {
    val letterGram = new mutable.HashMap[String, Int]()
    val src = s"#$word#"
    for (i <- 0 to src.length - 3) {
      val l3g = src.substring(i, i + 3)
      if (letterGram.contains(l3g)) {
        letterGram(l3g) += 1
      } else letterGram.put(l3g, 1)
    }

    val encodeByVocab = new mutable.HashMap[Int, Int]()

    for (g <- letterGram) {
      if (vocabulary.get(g._1).isDefined) encodeByVocab.put(vocabulary(g._1), g._2)
    }

    encodeByVocab
  }

}