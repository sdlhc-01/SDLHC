package segmentation

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, diag, inv, max, min, sum}
import breeze.numerics.sqrt

import scala.annotation.tailrec
import scala.util.Try

object Tools {

  def polyRegression(nCoef: Int = 0, X: DenseMatrix[Double], Y: DenseVector[Double]): DenseVector[Double] = {

    require(X.rows == Y.length, "In PolyRegression_, X.rows != Y.length")

    if (nCoef > 0) {
      require(X.rows == Y.length, "In PolyRegression_, X columns number is lesser than the number of coefs asked")
      val X_restricted = X(::, 0 until nCoef)
      inv(X_restricted.t * X_restricted) * X_restricted.t * Y

    } else {
      inv(X.t * X) * X.t * Y
    }
  }

  def weightedLeastSquareResolution(X: DenseMatrix[Double], Y: DenseVector[Double], weigths: DenseVector[Double]): DenseVector[Double] = {
    val W = diag(weigths); // n x n
    inv(X.t * W * X) * X.t * W * Y
  }

  def weightedLeastSquareResolutionForWeightsHigherThanPrecision(X: DenseMatrix[Double],
                                                                 Y: DenseVector[Double],
                                                                 weights: DenseVector[Double]): DenseVector[Double] = {
    val sliceRow: Seq[Int] = (0 until weights.length).filter(i => weights(i) > EM.precision)
    if (sliceRow.length <= 4) {
      weightedLeastSquareResolution(X, Y, weights)
    } else {
      val XFiltered: DenseMatrix[Double] = X(sliceRow, ::).toDenseMatrix
      val YFiltered = Y(sliceRow).toDenseVector
      val weightsFiltered: DenseVector[Double] = weights(sliceRow).toDenseVector
      Try({
        weightedLeastSquareResolution(XFiltered, YFiltered, weightsFiltered)
      }) getOrElse weightedLeastSquareResolution(X, Y, weights)
    }
  }

  def weightedVariance(X: DenseVector[Double],
                       mode: DenseVector[Double],
                       weights: DenseVector[Double]): Double = {
    val deltas = X - mode
    val squareDeltas = deltas *:* deltas
    val weightedSquareDeltas = weights *:* squareDeltas
    sum(weightedSquareDeltas) / sum(weights)
  }

  def variance(X: DenseVector[Double]): Double = {
    covariance(X, X)
  }

  def covariance(X: DenseVector[Double], Y: DenseVector[Double]): Double = {
    sum((X - mean(X)) *:* (Y - mean(Y))) / Y.length
  }

  def mean(X: DenseVector[Double]): Double = {
    sum(X) / X.length
  }

  def meanList(X: List[Double]): Double = {
    X.sum / X.length
  }

  def correlation(X: DenseVector[Double], Y: DenseVector[Double]): Double = {
    covariance(X, Y) / (sqrt(variance(X)) * sqrt(variance(Y)))
  }

  def whileLoop(cond: => Boolean)(block: => Unit): Unit =
    if (cond) {
      block
      whileLoop(cond)(block)
    }

  def coordinateDescentAlgo(step: Double = 1e-1, X: DenseMatrix[Double], Y: DenseVector[Double]): DenseVector[Double] = {

    val n = Y.length

    @tailrec
    def go(coefs: DenseVector[Double], X: DenseMatrix[Double], residuals: DenseVector[Double]): DenseVector[Double] = {

      if (coefs.toArray.count(_ > 0D) < coefs.length) {

        var nextResiduals = residuals
        var nextCoefs = coefs

        var correlations = DenseVector((0 until X.cols).map(p => correlation(X(::, p), nextResiduals)).toArray)
        val bestCoefIdx = argmax(correlations)
        var nextBestCoefIdx = argmax(correlations)

        whileLoop(nextBestCoefIdx == bestCoefIdx) {
          val delta = step * Math.signum(correlations(bestCoefIdx))
          nextResiduals = nextResiduals - delta * X(::, bestCoefIdx)
          nextCoefs(bestCoefIdx) = nextCoefs(bestCoefIdx) + delta

          correlations = DenseVector((0 until X.cols).map(p => correlation(X(::, p), nextResiduals)).toArray)
          nextBestCoefIdx = argmax(correlations)
        }

        go(coefs: DenseVector[Double], X: DenseMatrix[Double], nextResiduals: DenseVector[Double])
      } else {
        coefs
      }

    }

    val coefsInit = DenseVector.ones[Double](Y.length)
    go(coefsInit, X, Y)

  }

  def blockReplace(rangeRows: Vector[Int], rangeCols: Vector[Int], X: DenseMatrix[Double], block: DenseMatrix[Double]): DenseMatrix[Double] = {

    require(rangeRows.max <= X.rows, "rangeRow maximum index greater than target rows number")
    require(rangeCols.max <= X.cols, "rangeCols maximum index greater than target rows number")
    require(rangeRows.length <= X.rows, "rangeRow length greater than target rows number")
    require(rangeCols.length <= X.cols, "rangeCols length greater than target rows number")
    require(rangeRows.length == block.rows, "block rows number != rangeRow length")
    require(rangeCols.length == block.cols, "block cols number != rangeCols length")

    val Y = X
    for (i <- rangeRows.indices) {
      for (j <- rangeCols.indices) {
        Y(rangeRows(i), rangeCols(j)) = block(i, j)
      }
    }

    Y
  }

  def factorial(n: Int): Int = n match {
    case 0 => 1
    case _ => n * factorial(n - 1)
  }

  def insertColumn(X: DenseMatrix[Double], x: DenseVector[Double], index: Int): DenseMatrix[Double] = {

    require(x.length == X.rows, "Column to be inserted has not the same length than target matrix row number")
    require((0 until X.cols + 1).contains(index), "Index of insertion is not in the range (0 .. targetMatrix.cols+1)")

    val XWithInsertion = DenseMatrix.zeros[Double](X.rows, X.cols + 1)
    if (index == 0) {
      XWithInsertion(::, 0) := x
      XWithInsertion(::, 1 until XWithInsertion.cols) := X
      XWithInsertion
    } else if (index == XWithInsertion.cols - 1) {
      XWithInsertion(::, XWithInsertion.cols - 1) := x
      XWithInsertion(::, 0 until X.cols) := X
      XWithInsertion
    } else {
      XWithInsertion(::, 0 until index) := X(::, 0 until index)
      XWithInsertion(::, index) := x
      XWithInsertion(::, index + 1 until XWithInsertion.cols) := X(::, index until X.cols)
      XWithInsertion
    }
  }

  def normalizeMatrix(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    X /:/ (sum(X(*, ::)) * DenseVector.ones[Double](X.cols).t)
  }

  def normalizeVector(X: DenseVector[Double]): DenseVector[Double] = {
    (X - min(X)) / (max(X) - min(X))
  }

  def standardizeVector(X: DenseVector[Double]): DenseVector[Double] = {
    (X - mean(X)) / sqrt(variance(X))
  }

  def weightedMean(x: DenseVector[Double], weights: DenseVector[Double]): Double = {
    sum(x *:* weights) / sum(weights)
  }

  def indexWeightedMedian(weights: DenseVector[Double]): Int = {
    val normalizedWeights = (weights / sum(weights)).toArray
    val cumSumWeights: Array[Double] = normalizedWeights.map {
      var s = 0D; d => {
        s += d; s
      }
    }
    cumSumWeights.indexWhere(_ >= 0.5)
  }

  def insert[T](list: List[T], i: Int, value: T) = {
    val (front, back) = list.splitAt(i)
    front ++ List(value) ++ back
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1E9 + " s")
    result
  }

  def timeSec[R](block: => R) = {
    val t0: Double = System.nanoTime()
    val result = block // call-by-name
    val t1: Double = System.nanoTime()

    Vector(result, (t1 - t0) / 1E9)

  }

}
