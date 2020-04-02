package test.scala.segmentation_test

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.FlatSpec
import segmentation.Tools

class toolsSpecs extends FlatSpec {

  "blockReplace method " should " work" in {
    val X = DenseMatrix.ones[Double](4, 4)
    val Result = new DenseMatrix(4, 4, Array(1.0, 3, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    val replacement = new DenseMatrix(2, 2, Array(1.0, 3, 2, 4))
    assertResult(Result) {
      Tools.blockReplace((0 to 1).toVector, (0 to 1).toVector, X, replacement)
    }
  }

  "insertColumn method " should " throw error if dimensions mismatch" in {
    val X = DenseMatrix.ones[Double](4, 4)
    val x = DenseVector.ones[Double](3)
    assertThrows[IllegalArgumentException] {
      Tools.insertColumn(X, x, 0)
    }
    assertThrows[IllegalArgumentException] {
      Tools.insertColumn(X, x, -1)
    }
    assertThrows[IllegalArgumentException] {
      Tools.insertColumn(X, x, 6)
    }
  }

  "insertColumn method " should " insert a column correctly at 0 index" in {
    val X = DenseMatrix.zeros[Double](4, 4)
    val x = DenseVector.ones[Double](4)
    val result = DenseMatrix.zeros[Double](4, 5)
    result(::, 0) := 1.0
    assertResult(result) {
      Tools.insertColumn(X, x, 0)
    }
  }

  "insertColumn method " should " insert a column correctly at index between first and last" in {
    val X = DenseMatrix.zeros[Double](4, 4)
    val x = DenseVector.ones[Double](4)
    var result = DenseMatrix.zeros[Double](4, 5)

    result(::, 1) := 1.0
    assertResult(result) {
      Tools.insertColumn(X, x, 1)
    }

    result = DenseMatrix.zeros[Double](4, 5)
    result(::, 2) := 1.0
    assertResult(result) {
      Tools.insertColumn(X, x, 2)
    }

    result = DenseMatrix.zeros[Double](4, 5)
    result(::, 3) := 1.0
    assertResult(result) {
      Tools.insertColumn(X, x, 3)
    }
  }

  "insertColumn method " should " insert a column correctly at last index" in {
    val X = DenseMatrix.zeros[Double](4, 4)
    val x = DenseVector.ones[Double](4)
    val result = DenseMatrix.zeros[Double](4, 5)
    result(::, 4) := 1.0
    assertResult(result) {
      Tools.insertColumn(X, x, 4)
    }
  }

  "normalizeMatrix method " should " make the rows sum equal to 1" in {
    val X = DenseMatrix((1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0), (5.0, 5.0, 0.0, 0.0), (0.0, 0.0, 10.0, 0.0))
    val result = DenseMatrix((0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25), (0.5, 0.5, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    assertResult(result) {
      Tools.normalizeMatrix(X)
    }
  }

}
