package segmentation

import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.{DenseMatrix, DenseVector}
import com.opencsv.CSVWriter

import scala.collection.JavaConverters._
import scala.io.Source
import scala.util.{Failure, Try}

object InputOutput {

  def readTS(path: String): List[DenseMatrix[Double]] = {
    val lines = Source.fromFile(path).getLines.toList.drop(1)
    lines.indices.map(seg => {
      lines(seg).drop(1).dropRight(1).split(";").toList.map(string => DenseMatrix(string.split(":").map(_.toDouble))).reduce((a, b) => DenseMatrix.vertcat(a, b))
    }).toList
  }

  def outputToCSv(fileName: String,
                  header: List[String],
                  content: List[List[String]]
                 ): Try[Unit] =
    Try(new CSVWriter(new BufferedWriter(new FileWriter(fileName)))).flatMap((csvWriter: CSVWriter) =>
      Try {
        csvWriter.writeAll(
          (header +: content).map(_.toArray).asJava
        )
        csvWriter.close()
      } match {
        case f@Failure(_) =>
          Try(csvWriter.close()).recoverWith {
            case _ => f
          }
        case success =>
          success
      }
    )

  def writeCoefsCsv(fileName: String, bestModelsToRead: Vector[EMFuncSegState], dataSample: List[DenseMatrix[Double]]): Try[Unit] = {

    val res: List[List[String]] = bestModelsToRead.indices.map(idx => {
      val res1: List[List[String]] = PostProcessing.getCoefWithInvarianceFromModel(bestModelsToRead(idx), dataSample(idx)(::, 0), dataSample(idx)(::, 1)
      ).map(coefs => List(idx.toString) ++ coefs.map(coef => coef.toString).toArray.toList)
      res1
    }).reduce(_ ++ _)

    val header: List[String] =
      List("id") ++ List("coefId") ++ res.head.indices.drop(1).map(idx => "coef".concat(idx.toString))

    def addPrefix(lls: List[List[String]]): List[List[String]] =
      lls.foldLeft((1, List.empty[List[String]])) {
        case ((serial: Int, acc: List[List[String]]), value: List[String]) =>
          (serial + 1, (serial.toString +: value) +: acc)
      }._2.reverse

    outputToCSv(fileName, header, addPrefix(res))
  }

  def writeRegimesLimitsCsv(fileName: String,
                            bestModelsToRead: List[EMFuncSegState],
                            dataSample: List[DenseMatrix[Double]]): Try[Unit] = {

    val res: List[List[String]] = bestModelsToRead.indices.map(idx => {
      List(List(EMFuncSegState.extractRegimesLimitsProbabilisticThreshold(bestModelsToRead(idx).membershipProbabilities).toArray
        .map(p => p.toString).reduce((a, b) => a.concat(";").concat(b))))
    }).reduce(_ ++ _)

    val header: List[String] =
      List("id") ++ List("limits")

    def addPrefix(lls: List[List[String]]): List[List[String]] =
      lls.foldLeft((1, List.empty[List[String]])) {
        case ((serial: Int, acc: List[List[String]]), value: List[String]) =>
          (serial + 1, (serial.toString +: value) +: acc)
      }._2.reverse

    outputToCSv(fileName, header, addPrefix(res))
  }

}
