package test.scala.segmentation_test

import breeze.linalg.{DenseMatrix, DenseVector, all}
import breeze.numerics.abs
import org.scalatest.FlatSpec
import segmentation.{EM, EMFuncSegState, FunctionBasis}

class ParamsSpecs extends FlatSpec {

  val validFunctionBasis: Array[Double => Double] = FunctionBasis.PolynomialBasis(4, "Canonical")
  var validTimes: DenseVector[Double] = DenseVector.tabulate(100) { i => i: Double }
  val validSequence: DenseVector[Double] = DenseVector.tabulate(100) { i => abs(i - 50): Double }

  "Param construction " should "throw an error if params polyDegree and nSegments are not positive integer" in {

    assertThrows[IllegalArgumentException] {
      EM.EMAlgo(1, Array(), 2, validTimes, validSequence)
    }
    assertThrows[IllegalArgumentException] {
      EM.EMAlgo(1, validFunctionBasis, 0, validTimes, validSequence)
    }
  }

  "Param construction " should "throw an error if sequenceLength <= 10 x nSegments" in {

    assertThrows[IllegalArgumentException] {
      EM.EMAlgo(1, validFunctionBasis, 10, validTimes, validSequence)
    }
  }

  "Param construction " should "throw an error if length of Sequence and length of Times don't match" in {

    var invalidTimes = DenseVector.tabulate(99) { i => i: Double }
    assertThrows[IllegalArgumentException] {
      EM.EMAlgo(1, validFunctionBasis, 1, invalidTimes, validSequence)
    }

    invalidTimes = DenseVector.vertcat(validTimes, DenseVector(1))
    assertThrows[IllegalArgumentException] {
      EM.EMAlgo(1, validFunctionBasis, 1, invalidTimes, validSequence)
    }
  }

  "Param construction " should " throw an error if there are duplicate values of Times" in {

    var invalidTimes = DenseVector.tabulate(99) { i => i: Double }
    invalidTimes = DenseVector.vertcat(invalidTimes, DenseVector(1))

    assertThrows[IllegalArgumentException] {
      EM.EMAlgo(1, validFunctionBasis, 1, invalidTimes, validSequence)
    }
  }

  "splitBelongingProbabilities method " should " work" in {

    val X = DenseMatrix(
      (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)).t

    val result: DenseMatrix[Double] = DenseMatrix(
      (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)).t

    val delta: DenseMatrix[Double] = EMFuncSegState.splitBelongingProbabilities(0, X) - result
    val error: DenseMatrix[Double] = DenseMatrix.fill(10, 4)(1E-3)
    val allCheck: DenseMatrix[Boolean] = delta <:< error
    assert(
      all(allCheck)
    )

    val X2 = DenseMatrix(
      (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)).t
    val result2: DenseMatrix[Double] = DenseMatrix(
      (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)).t

    assert(
      all(EMFuncSegState.splitBelongingProbabilities(0, X2) - result2 <:< DenseMatrix.fill(10, 4)(1E-3))
    )

    val X3 = DenseMatrix(
      (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)).t

    val result3: DenseMatrix[Double] = DenseMatrix(
      (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)).t

    assert(
      all(EMFuncSegState.splitBelongingProbabilities(2, X3) - result3 <:< DenseMatrix.fill(10, 4)(1E-3))
    )

  }

}
