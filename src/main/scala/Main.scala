import java.io._
import java.nio.file.{Files, Paths}

import breeze.linalg.{DenseMatrix, DenseVector}
import segmentation._

import scala.util.Try

object Main {

  def main(args: Array[String]) {

    require(args.length == 2, "\n" +
      "SDLHC launcher takes exactly 2 arguments: \n" +
      "- the path of the dataset (.csv) \n" +
      "- the path of the output directory \n")

    val pathDataInput = args(0)
    val pathOutput = args(1)

    require(Files.exists(Paths.get(pathDataInput)), "Error: mandatory first argument data path ".concat(pathDataInput).concat(" doesn't point to a file"))
    require(Files.exists(Paths.get(pathOutput)), "Error: mandatory second argument output directory ".concat(pathOutput).concat(" doesn't exists"))

    val pathDirOutputSegmentation = pathOutput.concat("/segmentation/")
    val data = InputOutput.readTS(pathDataInput)

    run(data, pathDirOutputSegmentation, verbose= true)

  }

  def run(data: List[DenseMatrix[Double]],
          pathDirOutput: String = "None",
          optimModelStrategy: String = "addSegmentOrAugmentPolyDegree",
          evaluationCriterion: String = "BIC",
          polyDegree: Int = 3,
          verbose: Boolean = false):Unit={

    val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")

    println("Segmentation begins: ".concat(data.length.toString).concat(" time series to process"))

    val batchSize: Double = 96
    val nBatch = (data.length / batchSize.toDouble).ceil.toInt
    var resultsAllBatch: List[EMFuncSegState] = List[EMFuncSegState]()

    for (idxBatch <- 0 until nBatch) {

      println("Begins batch ".concat((1 + idxBatch).toString).concat("/").concat(nBatch.toString))
      val idxRange = data.indices.filter(_ >= ((idxBatch: Double) * batchSize)).filter(_ < ((1 + idxBatch: Double) * batchSize))
      val bestModels: List[EMFuncSegState] = Tools.time {
        idxRange.par.map(idx => {
          println(">>>".concat(idx.toString))
          EM.getBestModel(data(idx), functionBasis, optimModelStrategy, evaluationCriterion = evaluationCriterion)
        }).toList
      }

      resultsAllBatch = resultsAllBatch ++ bestModels

      val out = new ObjectOutputStream(new FileOutputStream(pathDirOutput.concat("tempSegment.obj")))
      out.writeObject(resultsAllBatch)
      out.close()

      val pw = new PrintWriter(new File(pathDirOutput.concat(".progress")))
      pw.write(idxBatch.toString.concat("/").concat(nBatch.toString))
      pw.close

      segmentation.InputOutput.writeCoefsCsv(pathDirOutput.concat("coefficients.csv"), resultsAllBatch.toVector, data)
      segmentation.InputOutput.writeRegimesLimitsCsv(pathDirOutput.concat("segmentLimits.csv"), resultsAllBatch, data)

    }
  }
}

