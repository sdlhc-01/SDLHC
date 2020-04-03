package segmentation

import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.numerics.sqrt
import segmentation.EM.initPolyBaseValue

object PostProcessing {

  def getCoefWithInvarianceFromModel(model: EMFuncSegState,
                                     times: DenseVector[Double], sequence: DenseVector[Double],
                                     invariancesToKeep: String = "None"): List[DenseVector[Double]] = {

    val regimesLimit = EMFuncSegState.extractRegimesLimitsProbabilisticThreshold(model.membershipProbabilities).toArray.toList
    val coefficients = (regimesLimit zip regimesLimit.tail).map(limits => {
      val timeRegime = 2.0 * ((times(limits._1 until limits._2) - times(limits._1)) / (times(limits._2 - 1) - times(limits._1))) - 1D
      val sequenceRegime = sequence(limits._1 until limits._2)
      val polyBaseValue: DenseMatrix[Double] = initPolyBaseValue(timeRegime)
      val res = segmentation.Tools.polyRegression(X = polyBaseValue, Y = sequenceRegime)
      DenseVector(res.toArray)
    })

    coefficients
  }

  def getCoefWithInvarianceFromSegmentsLimits(regimesLimit: DenseVector[Int],
                                              times: DenseVector[Double],
                                              sequence: DenseVector[Double],
                                              centerValue: Boolean,
                                              scaleValue: Boolean,
                                              scaleSupport: Boolean,
                                              degreeRegressionBase: Int): List[DenseVector[Double]] = {

    val regimesLimitList = regimesLimit.toArray.toList
    val coefficients = (regimesLimitList zip regimesLimitList.tail).map(limits => {

      val sequenceRegime: DenseVector[Double] = sequence(limits._1 until limits._2)

      val mu: Double = {
        segmentation.Tools.mean(sequenceRegime): Double
        //        min(sequenceRegime)
      }

      val sigma: Double = max({
        sqrt(segmentation.Tools.variance(sequenceRegime))
        //        max(sequenceRegime) - min(sequenceRegime)
      }, 1e-8)

      val duration: Double = {
        times(limits._2 - 1) - times(limits._1)
      }

      val timeRegime: DenseVector[Double] = 2.0 * ((times(limits._1 until limits._2) - times(limits._1)) / duration) - 1D
      val sequenceRegimePostProcessed = (sequenceRegime - mu) / sigma

      val polyBaseValue: DenseMatrix[Double] = initPolyBaseValue(timeRegime, degreeRegressionBase)

      val res0 = segmentation.Tools.polyRegression(X = polyBaseValue, Y = sequenceRegimePostProcessed)
      val res = if (!centerValue) {
        DenseVector.vertcat(res0, DenseVector(mu))
      }
      else if (!scaleValue) {
        DenseVector.vertcat(res0, DenseVector(sigma))
      }
      else if (!scaleSupport) {
        DenseVector.vertcat(res0, DenseVector(duration))
      }
      else {
        res0
      }
      res
    })

    coefficients
  }

}
