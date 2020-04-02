package test.scala.segmentation_test

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FeatureSpec, GivenWhenThen}
import segmentation._

class SegmentationFunctionalSpecs extends FeatureSpec with GivenWhenThen {

  val evaluationCriterion = "BIC"

  feature("Segmentation with fixed number of segments") {

    scenario("One segment, first degree polynom") {

      val polyDegree = 1
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f0(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, evaluationCriterion = evaluationCriterion)
      assert((bestModel.polyRegressionCoefficients.head - bestModel.polyRegressionCoefficients(1) <:< DenseVector(1e-4, 1e-4)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients.head - DenseVector(74.5, 130)) <:< DenseVector(1e-4, 1)).reduce(_ & _))

    }

    scenario("Two segment, first degree polynom") {

      val polyDegree = 1
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f1(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, evaluationCriterion = evaluationCriterion)
      assert(((bestModel.polyRegressionCoefficients.head - DenseVector(0.500D, -130D)) <:< DenseVector(1e-4, 1)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(1) - DenseVector(-0.500D, 130D)) <:< DenseVector(1e-4, 1)).reduce(_ & _))

    }

    scenario("Two segment, Second degree polynom") {

      val polyDegree = 2
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f2(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, evaluationCriterion = evaluationCriterion)
      assert(((bestModel.polyRegressionCoefficients.head - DenseVector(12.3, -0.9, 55)) <:< DenseVector(1e-1, 1e-1, 1)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(1) - DenseVector(-24.6, 1.72, -110)) <:< DenseVector(1e-1, 1e-1, 1)).reduce(_ & _))

    }

    scenario("5 segment, Second degree polynom") {

      val polyDegree = 2
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f3(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, nSegments = 5, evaluationCriterion = evaluationCriterion)
      val results = List(DenseVector(-36.488134609695706, -146.354286417112, -27.29795035284182),
        DenseVector(49.56797972694033, 185.1877589417579, 60.60258867965643),
        DenseVector(17.985905140015674, -61.57787366099127, 81.28541232835012),
        DenseVector(-25.581875998112558, 65.28259614983558, -114.67588217615904),
        DenseVector(44.208907594450324, -119.60535365595871, 161.94924490097299))
      assert(((bestModel.polyRegressionCoefficients.head - DenseVector(-36.5, 146.3, -27)) <:< DenseVector(1e-1, 1e-1, 1e-1)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(1) - DenseVector(49.5, 185.8, 62.6)) <:< DenseVector(1, 1, 1e0)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(2) - DenseVector(18, -61.6, 81.3)) <:< DenseVector(1e-1, 1e-1, 1e-1)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(3) - DenseVector(-25.6, 65.3, -114.7)) <:< DenseVector(1e-1, 1e-1, 1e-1)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(4) - DenseVector(44.2, -119.6, 161.9)) <:< DenseVector(1e-1, 1e-1, 1e-1)).reduce(_ & _))

    }
  }

  feature("Segmentation with optimisation of segments number") {


    scenario("Two segment, first degree polynom") {

      val polyDegree = 1
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f1(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, "addSegment", evaluationCriterion = evaluationCriterion)
      assert(bestModel.polyRegressionCoefficients.length == 2)
      assert(((bestModel.polyRegressionCoefficients.head - DenseVector(0.500D, -130D)) <:< DenseVector(1e-4, 1)).reduce(_ & _))
      assert(((bestModel.polyRegressionCoefficients(1) - DenseVector(-0.500D, 130D)) <:< DenseVector(1e-4, 1)).reduce(_ & _))
    }

    scenario("5 segments, Second degree polynom") {

      val polyDegree = 2
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f3(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, "addSegmentOrAugmentPolyDegree", evaluationCriterion = evaluationCriterion)
      assert(bestModel.polyRegressionCoefficients.length == 5)

    }

    scenario("2 segments, one first degree polynom and one 2nd degree polynom") {

      val polyDegree = 1
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f6(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)
      val bestModel = EM.getBestModel(ts, functionBasis, "augmentPolyDegree", evaluationCriterion = evaluationCriterion)

      assert(bestModel.polyRegressionCoefficients.head.length == 3)
      assert(bestModel.polyRegressionCoefficients(1).length == 2)

    }

    scenario("Rectangular function") {

      val polyDegree = 1
      val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
      val times: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => i: Double }
      val sequence: DenseMatrix[Double] = DenseMatrix.tabulate(150, 1) { (i, j) => DataSimulation.f5(i.toDouble) }
      val ts: DenseMatrix[Double] = DenseMatrix.horzcat(times, sequence)

      val bestModelPolyDegree = EM.getBestModel(ts, functionBasis, "augmentPolyDegree", evaluationCriterion = evaluationCriterion)
      assertResult(DenseVector(0, 50, 100, 149)) {
        EMFuncSegState.extractRegimesLimitsProbabilisticThreshold(bestModelPolyDegree.membershipProbabilities)
      }
      assert(bestModelPolyDegree.polyRegressionCoefficients.length == 2)

      val bestModelNSegment = EM.getBestModel(ts, functionBasis, "addSegment", evaluationCriterion = evaluationCriterion)
      assertResult(DenseVector(0, 50, 100, 149)) {
        EMFuncSegState.extractRegimesLimitsProbabilisticThreshold(bestModelNSegment.membershipProbabilities)
      }
      assert(bestModelNSegment.polyRegressionCoefficients.length == 2)
    }
  }
}
