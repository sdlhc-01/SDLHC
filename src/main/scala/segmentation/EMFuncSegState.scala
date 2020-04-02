package segmentation

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, max, sum}
import breeze.numerics._
import breeze.stats.distributions.Gaussian
import segmentation.EM._
import segmentation.EMFuncSegState._

case class EMFuncSegState(polyRegressionCoefficients: List[DenseVector[Double]],
                          polyRegressionVariances: DenseVector[Double],
                          conditionalMembershipProbabilities: DenseMatrix[Double],
                          logisticWeights: DenseMatrix[Double],
                          membershipProbabilities: DenseMatrix[Double],
                          var jointLogDistrib: DenseMatrix[Double],
                          var scores: Map[String, Double]) {

  // Constructor empty model
  def this() {
    this(List(DenseVector(0D)), DenseVector(0D), DenseMatrix.zeros[Double](0, 0), DenseMatrix.zeros[Double](0, 0),
      DenseMatrix.zeros[Double](0, 0), DenseMatrix.zeros[Double](0, 0), Map.empty[String, Double])
  }

  // Main constructor
  def this(polyRegressionCoefficients: List[DenseVector[Double]],
           polyRegressionVariances: DenseVector[Double],
           conditionalMembershipProbabilities: DenseMatrix[Double],
           logisticWeights: DenseMatrix[Double],
           membershipProbabilities: DenseMatrix[Double],
           polyBaseValue: DenseMatrix[Double],
           sequence: DenseVector[Double]) {

    this(
      polyRegressionCoefficients,
      polyRegressionVariances,
      conditionalMembershipProbabilities,
      logisticWeights: DenseMatrix[Double],
      membershipProbabilities,
      DenseMatrix.zeros[Double](0, 0),
      Map.empty[String, Double])

    this.jointLogDistrib = computeJointDistrib(sequence, polyBaseValue)
    this.scores = computeScore()
  }


  // Constructor with segments degree and uniform segments limits
  def this(polyRegressionCoefficientsLength: List[Int], nSegments: Int, polyBaseValue: DenseMatrix[Double], polyBaseValueW: DenseMatrix[Double], sequence: DenseVector[Double]) {

    this(
      initializePolyRegressionCoef(polyRegressionCoefficientsLength, polyBaseValue, sequence, initializeUniformSegmentsLimitsIndexes(sequence.length, nSegments)),
      initializePolyRegressionVariance(nSegments),
      computeMembershipProbabilities(DenseMatrix.zeros[Double](polyBaseValueW.cols, nSegments), polyBaseValueW),
      DenseMatrix.zeros[Double](polyBaseValueW.cols, nSegments),
      computeMembershipProbabilities(DenseMatrix.zeros[Double](polyBaseValueW.cols, nSegments), polyBaseValueW),
      DenseMatrix.zeros[Double](0, 0),
      Map.empty[String, Double])

    this.jointLogDistrib = computeJointDistrib(sequence, polyBaseValue)
    this.scores = computeScore()

  }

  // Constructor with segments degree and regime limits
  def this(polyRegressionCoefficientsLength: List[Int], polyBaseValue: DenseMatrix[Double], polyBaseValueW: DenseMatrix[Double], sequence: DenseVector[Double], sequenceLimitsIndexes: DenseVector[Int]) {

    this(
      initializePolyRegressionCoef(polyRegressionCoefficientsLength, polyBaseValue, sequence, sequenceLimitsIndexes),
      initializePolyRegressionVariance(sequenceLimitsIndexes.length - 1),
      computeMembershipProbabilities(DenseMatrix.zeros[Double](polyBaseValueW.cols, sequenceLimitsIndexes.length - 1), polyBaseValueW),
      DenseMatrix.zeros[Double](polyBaseValueW.cols, sequenceLimitsIndexes.length - 1),
      computeMembershipProbabilities(DenseMatrix.zeros[Double](polyBaseValueW.cols, sequenceLimitsIndexes.length - 1), polyBaseValueW),
      DenseMatrix.zeros[Double](0, 0),
      Map.empty[String, Double])

    this.jointLogDistrib = computeJointDistrib(sequence, polyBaseValue)
    this.scores = computeScore()

  }

  def computeJointDistrib(sequence: DenseVector[Double], polyBaseValue: DenseMatrix[Double]): DenseMatrix[Double] = {

    val emissionPdf: DenseMatrix[Double] = computeEmissionPdfForPositivePosteriorBelonging(polyBaseValue,
      polyRegressionCoefficients,
      polyRegressionVariances,
      sequence,
      membershipProbabilities
    )
    val piEmissionPdf = membershipProbabilities *:* emissionPdf
    piEmissionPdf
  }

  def computeScore(): Map[String, Double] = {
    val logLikelihood = computeLogLikelihood()
    val BIC = computeBIC(logLikelihood)
    val entropy = computeEntropy()
    val ICL = computeICL(BIC, entropy)

    Map("logLikelihood" -> logLikelihood, "BIC" -> BIC, "entropy" -> entropy, "ICL" -> ICL)
  }

  def computeLogLikelihood(): Double = {
    sum(log(sum(jointLogDistrib(*, ::))))
  }

  def computeBIC(logLikelihood: Double): Double = {
    val coefsRegression = polyRegressionCoefficients.map(coef => coef.length).sum
    val nCoefs = coefsRegression + polyRegressionVariances.length + membershipProbabilities.cols - 1.0
    val BIC = log(membershipProbabilities.rows) * nCoefs - 2 * logLikelihood
    BIC
  }

  def computeEntropy(): Double = {
    -sum(conditionalMembershipProbabilities *:* log(conditionalMembershipProbabilities))
  }

  def computeICL(BIC: Double, entropy: Double): Double = {
    BIC - entropy
  }

  // Initialize with segments degree and belonging probabilities
  def this(polyRegressionCoefficientsLength: List[Int],
           polyBaseValue: DenseMatrix[Double],
           polyBaseValueW: DenseMatrix[Double],
           sequence: DenseVector[Double],
           membershipProbabilities: DenseMatrix[Double]) {

    this(
      UpdateRegressionCoefficient(polyRegressionCoefficientsLength, sequence, polyBaseValue, membershipProbabilities),
      initializePolyRegressionVariance(membershipProbabilities.cols),
      computeMembershipProbabilities(DenseMatrix.zeros[Double](polyBaseValueW.cols, membershipProbabilities.cols), polyBaseValueW),
      DenseMatrix.zeros[Double](polyBaseValueW.cols, membershipProbabilities.cols),
      computeMembershipProbabilities(DenseMatrix.zeros[Double](polyBaseValueW.cols, membershipProbabilities.cols), polyBaseValueW),
      DenseMatrix.zeros[Double](0, 0),
      Map.empty[String, Double])


    this.jointLogDistrib = computeJointDistrib(sequence, polyBaseValue)
    this.scores = computeScore()

  }
}

object EMFuncSegState {

  def initializeUniformSegmentsLimitsIndexes(sequenceLength: Int, nSegments: Int): DenseVector[Int] = {

    val segmentsBelonging = DenseVector.tabulate[Int](sequenceLength) { i => i % nSegments }.toArray.sorted
    val segmentsLimitsIndexes = (0 until nSegments).map(s => segmentsBelonging.indexOf(s)) :+ (sequenceLength - 1)
    DenseVector(segmentsLimitsIndexes.toArray)
  }

  private def initializePolyRegressionCoef(polyRegressionCoefsLength: List[Int],
                                           polyBaseValue: DenseMatrix[Double],
                                           sequence: DenseVector[Double],
                                           segmentsLimitsIndexes: DenseVector[Int]): List[DenseVector[Double]] = {

    val res = polyRegressionCoefsLength.indices.map(idx => {
      if ((segmentsLimitsIndexes(idx + 1) - segmentsLimitsIndexes(idx)) <= 2) {
        val start = List(0, segmentsLimitsIndexes(idx) - 2).max
        val end = List(sequence.length, segmentsLimitsIndexes(idx + 1) + 2).min
        Tools.polyRegression(
          polyRegressionCoefsLength(idx),
          polyBaseValue(start until end, ::),
          sequence(start until end)
        )
      } else {
        Tools.polyRegression(
          polyRegressionCoefsLength(idx),
          polyBaseValue(segmentsLimitsIndexes(idx) until segmentsLimitsIndexes(idx + 1), ::),
          sequence(segmentsLimitsIndexes(idx) until segmentsLimitsIndexes(idx + 1))
        )
      }
    }).toList

    res
  }

  def initializePolyRegressionVariance(nSegments: Int): DenseVector[Double] = {
    DenseVector.ones(nSegments)
  }

  def computeWeightedLogLikelihood(weights: DenseVector[Double],
                                   membershipProbabilities: DenseMatrix[Double],
                                   polyRegressionCoefficients: List[DenseVector[Double]],
                                   polyBaseValue: DenseMatrix[Double],
                                   polyRegressionVariances: DenseVector[Double],
                                   sequence: DenseVector[Double]): Double = {
    val emissionPdf: DenseMatrix[Double] = computeEmissionPdf(polyBaseValue, polyRegressionCoefficients, polyRegressionVariances, sequence)
    val piEmissionPdf = membershipProbabilities *:* emissionPdf
    sum(weights *:* log(sum(piEmissionPdf(*, ::)))) / sum(weights)
  }

  def isSegmentDivided(probabilities: DenseVector[Double]): Boolean = {
    val startSeg = probabilities.toArray.indexWhere(_ >= 0.5)
    val endSeg = probabilities.toArray.indexWhere(_ < 0.5, from = startSeg)
    probabilities.toArray.indexWhere(_ >= 0.5, from = endSeg) > 0
  }

  def makeModelEvolve(EM: EMFuncSegState,
                      functionBasis: Array[Double => Double],
                      polyBaseValue: DenseMatrix[Double],
                      polyBaseValueW: DenseMatrix[Double],
                      sequence: DenseVector[Double],
                      times: DenseVector[Double],
                      method: String = "reboot",
                      evaluationCriterion: String = "BIC"): List[EMFuncSegState] = {

    require(List("rebootUniform", "addSegment", "augmentPolyDegree", "addSegmentOrAugmentPolyDegree", "addAugmentedSegment").contains(method), "The strategy chosen for the addition of a segment should be in (\"addSegment\",\"augmentPolyDegree\",\"addSegmentOrAugmentPolyDegree\",\"addAugmentedSegment\") ")
    if (method == "addSegment") {

      List(newModelAddSegmentFromBelongingProbabilities(EM, polyBaseValue, polyBaseValueW, sequence, times))
      List(newModelAddSegmentFromRegimeChangePoints(EM, polyBaseValue, polyBaseValueW, sequence, times))

    } else if (method == "augmentPolyDegree") {

      List(newModelAugmentPolyDegreeFromBelongingProbabilities(EM, polyBaseValue, polyBaseValueW, sequence, times))

    } else if (method == "addSegmentOrAugmentPolyDegree") {

      val newModelAddedSegment = newModelAddSegmentFromBelongingProbabilities(EM, polyBaseValue, polyBaseValueW, sequence, times, choiceDegreeNewSegments = "previous")
      val modelLogAddSegment = EMAlgo(nIter = 200, functionBasis, 2, times, sequence, EMInitState = newModelAddedSegment, optimModelStrategy = "None")
      val bestModelAddSegment = modelLogAddSegment(argmax(-DenseVector[Double](modelLogAddSegment.map(model => model.scores(evaluationCriterion)).toArray)))

      val newModelAugmentedDegree = newModelAugmentPolyDegreeFromBelongingProbabilities(EM, polyBaseValue, polyBaseValueW, sequence, times)
      val modelLogAugmentedDegree = EMAlgo(nIter = 200, functionBasis, 2, times, sequence, EMInitState = newModelAugmentedDegree, optimModelStrategy = "None")
      val bestModelAugmentedDegree = modelLogAugmentedDegree(argmax(-DenseVector[Double](modelLogAugmentedDegree.map(model => model.scores(evaluationCriterion)).toArray)))

      if (bestModelAddSegment.scores(evaluationCriterion) >= bestModelAugmentedDegree.scores(evaluationCriterion)) {
        modelLogAugmentedDegree.slice(0, argmax(-DenseVector[Double](modelLogAugmentedDegree.map(model => model.scores(evaluationCriterion)).toArray))).toList
      } else {
        modelLogAddSegment.slice(0, argmax(-DenseVector[Double](modelLogAddSegment.map(model => model.scores(evaluationCriterion)).toArray))).toList
      }

    } else if (method == "addAugmentedSegment") {
      val newModelAddedSegment = newModelAddSegmentFromBelongingProbabilities(EM, polyBaseValue, polyBaseValueW, sequence, times, choiceDegreeNewSegments = "2")
      val modelLogAddSegment = EMAlgo(nIter = 1000, functionBasis, 2, times, sequence, EMInitState = newModelAddedSegment, optimModelStrategy = "augmentPolyDegree")
      val bestModelAddSegment = modelLogAddSegment(argmax(-DenseVector[Double](modelLogAddSegment.map(model => model.scores(evaluationCriterion)).toArray)))

      if (bestModelAddSegment.scores(evaluationCriterion) >= EM.scores(evaluationCriterion)) {
        List(EM)
      } else {
        modelLogAddSegment.slice(0, argmax(-DenseVector[Double](modelLogAddSegment.map(model => model.scores(evaluationCriterion)).toArray))).toList
      }

    } else {
      List(new EMFuncSegState(EM.polyRegressionCoefficients.map(_.length), EM.polyRegressionVariances.length + 1, polyBaseValue, polyBaseValueW, sequence))
    }
  }

  def newModelAddSegmentFromRegimeChangePoints(model: EMFuncSegState, polyBaseValue: DenseMatrix[Double], polyBaseValueW: DenseMatrix[Double], sequence: DenseVector[Double], times: DenseVector[Double]): EMFuncSegState = {
    val regimesLimit = extractRegimesLimitsProbabilisticThreshold(model.membershipProbabilities).toArray.toList

    val regimesLogLikelihoods = DenseVector((regimesLimit zip regimesLimit.tail).map(limits => {
      new EMFuncSegState(
        model.polyRegressionCoefficients: List[DenseVector[Double]],
        model.polyRegressionVariances: DenseVector[Double],
        model.conditionalMembershipProbabilities: DenseMatrix[Double],
        model.logisticWeights: DenseMatrix[Double],
        model.membershipProbabilities(limits._1 until limits._2, ::),
        polyBaseValue(limits._1 until limits._2, ::),
        sequence(limits._1 until limits._2): DenseVector[Double]).scores("logLikelihood")

    }).toArray)

    val segmentMinLogLikelihood: Int = argmax(-regimesLogLikelihoods)
    val newSegmentsLimitsIndexes = DenseVector((regimesLimit.slice(0, segmentMinLogLikelihood + 1) ++ List((0.5 * (regimesLimit(segmentMinLogLikelihood) + regimesLimit(segmentMinLogLikelihood + 1))).toInt) ++ regimesLimit.slice(segmentMinLogLikelihood + 1, regimesLimit.length)).toArray)
    new EMFuncSegState(List.fill(newSegmentsLimitsIndexes.length - 1)(polyBaseValue.cols), polyBaseValue, polyBaseValueW, sequence, newSegmentsLimitsIndexes)
  }

  def splitBelongingProbabilities(segment: Int, probaBelongings: DenseMatrix[Double]): DenseMatrix[Double] = {

    val idxSplit: Int = Tools.indexWeightedMedian(probaBelongings(::, segment))
    val n = probaBelongings.rows
    var newProbaBelonging = Tools.insertColumn(probaBelongings, DenseVector.fill(n)(precision), segment)
    newProbaBelonging(::, segment + 1) := precision

    newProbaBelonging(0 to idxSplit, segment) := probaBelongings(0 to idxSplit, segment)
    newProbaBelonging((idxSplit + 1) until n, segment + 1) := probaBelongings((idxSplit + 1) until n, segment)

    Tools.normalizeMatrix(newProbaBelonging)
  }

  def newModelAddSegmentFromBelongingProbabilities(EMModel: EMFuncSegState,
                                                   polyBaseValue: DenseMatrix[Double],
                                                   polyBaseValueW: DenseMatrix[Double],
                                                   sequence: DenseVector[Double],
                                                   times: DenseVector[Double],
                                                   choiceDegreeNewSegments: String = "previous"): EMFuncSegState = {

    require(List("previous", "1", "2").contains(choiceDegreeNewSegments), "Error: optional parameter choiceDegreeNewSegments should be in ('previous','1','2')")

    val expectationNumberBelonging: List[Double] = sum(EMModel.membershipProbabilities(::, *)).t.toArray.toList
    val validSegments: List[Int] = expectationNumberBelonging.indices.filter(expectationNumberBelonging(_) > 2).toList

    val validcoefPolyRegression: List[DenseVector[Double]] = DenseVector(EMModel.polyRegressionCoefficients.toArray)(validSegments).toArray.toList

    val logLikelihoods: DenseVector[Double] = DenseVector(
      validSegments.map(segment => {
        computeWeightedLogLikelihood(
          EMModel.membershipProbabilities(::, segment),
          EMModel.membershipProbabilities,
          EMModel.polyRegressionCoefficients,
          polyBaseValue,
          EMModel.polyRegressionVariances,
          sequence
        )
      }).toArray)

    val segmentMinLogLikelihood = validSegments(argmax(-logLikelihoods))
    val coefsDegree: List[Int] = EMModel.polyRegressionCoefficients.map(_.length)

    val newCoefDegree = if (choiceDegreeNewSegments == "previous") {
      coefsDegree(segmentMinLogLikelihood)
    } else {
      choiceDegreeNewSegments.toInt
    }

    var newCoefDegrees = Tools.insert(coefsDegree, segmentMinLogLikelihood, newCoefDegree)
    newCoefDegrees = newCoefDegrees.updated(segmentMinLogLikelihood + 1, newCoefDegree)

    val probaBelongings = EMModel.conditionalMembershipProbabilities
    val newProbaBelongings = splitBelongingProbabilities(segmentMinLogLikelihood, probaBelongings)
    new EMFuncSegState(newCoefDegrees, initPolyBaseValue(times, polyDegree = newCoefDegrees.max), polyBaseValueW, sequence, newProbaBelongings)

  }

  def newModelAugmentPolyDegreeFromBelongingProbabilities(EMModel: EMFuncSegState, polyBaseValue: DenseMatrix[Double], polyBaseValueW: DenseMatrix[Double], sequence: DenseVector[Double], times: DenseVector[Double]): EMFuncSegState = {

    val regimesBelongingsIdx: List[List[Int]] = idxBelongingToRegimes(EMModel.membershipProbabilities)
    val validRegimes: List[Int] = regimesBelongingsIdx.zipWithIndex.filter(_._1.length >= 2).map(_._2)

    if (validRegimes.isEmpty) {
      EMModel
    } else {
      val logLikelihoods = DenseVector(

        EMModel.polyRegressionCoefficients.indices.map(segment => {
          if (regimesBelongingsIdx(segment).length <= 2) {
            NaN
          } else {
            computeWeightedLogLikelihood(
              EMModel.membershipProbabilities(::, segment),
              EMModel.membershipProbabilities,
              EMModel.polyRegressionCoefficients,
              polyBaseValue,
              EMModel.polyRegressionVariances,
              sequence
            )
          }
        }).toArray)

      val segmentMinLogLikelihood = validRegimes(argmax(-logLikelihoods(validRegimes)))
      val newPolyRegressionCoefLength: List[Int] = (DenseVector(EMModel.polyRegressionCoefficients.map(_.length).toArray) + DenseVector.tabulate(EMModel.polyRegressionCoefficients.length) { i =>
        if (i == segmentMinLogLikelihood) {
          1
        } else 0
      }).toArray.toList

      new EMFuncSegState(newPolyRegressionCoefLength, initPolyBaseValue(times, polyDegree = newPolyRegressionCoefLength.max), polyBaseValueW, sequence, EMModel.conditionalMembershipProbabilities)
    }
  }

  def idxBelongingToRegimes(membershipProbabilities: DenseMatrix[Double]): List[List[Int]] = {
    (0 until membershipProbabilities.cols).map(seg => {
      membershipProbabilities(::, seg).map(p => p >= 0.5).toArray.zipWithIndex.collect { case (true, i) => i }.toList
    }).toList
  }

  def limitsPerRegimesProbabilisticThreshold(membershipProbabilities: DenseVector[Double]): List[Int] = {
    val k = membershipProbabilities.toArray.map(p => p >= 0.5).toList.sliding(2).toList
    val h: List[Int] = k.map(pair => pair.head == pair(1)).zipWithIndex.collect { case (false, i) => i }
    h
  }

  def extractRegimesLimitsProbabilisticThreshold(membershipProbabilities: DenseMatrix[Double]): DenseVector[Int] = {

    val changePoints = DenseVector((0 until membershipProbabilities.cols).map(segment => {
      limitsPerRegimesProbabilisticThreshold(membershipProbabilities(::, segment))
    }).reduce(_ ++ _).toArray.sorted.distinct)

    val limits = DenseVector.vertcat(DenseVector.zeros[Int](1), changePoints, DenseVector[Int](Array(membershipProbabilities.rows - 1))).toArray.toList
    var filteredLimits = List(0)
    var k = 1

    Tools.whileLoop(k < limits.length) {
      if (limits(k) - filteredLimits.last > 2) {
        filteredLimits = filteredLimits ++ List(limits(k))
      }
      k += 1
    }

    if (filteredLimits.last - limits.last <= 2) {
      filteredLimits.patch(filteredLimits.length - 1, Seq(limits.last), 1)
    }
    DenseVector(filteredLimits.toArray)
  }

  def limitsPerRegimesWeightedMedian(membershipProbabilities: DenseVector[Double], times: DenseVector[Double]): List[Int] = {

    val grad = DenseVector(membershipProbabilities.toArray.indices.dropRight(1).map(idx => abs((membershipProbabilities(idx + 1) - membershipProbabilities(idx)) / (times(idx + 1) - times(idx)))).toArray)
    List(Tools.weightedMean(DenseVector(times.toArray.dropRight(1)), grad).toInt)
  }

  def extractRegimesLimitsWeightedMedian(membershipProbabilities: DenseMatrix[Double], times: DenseVector[Double]): DenseVector[Int] = {

    val limits = DenseVector((0 until membershipProbabilities.cols).map(segment => {
      limitsPerRegimesWeightedMedian(membershipProbabilities(::, segment), times)
    }).reduce(_ ++ _).toArray.sorted.distinct)

    DenseVector.vertcat(DenseVector.zeros(1), limits, DenseVector[Int](Array(membershipProbabilities.rows)))

  }

  def Q1Gen(conditionalMembershipProbabilities: DenseMatrix[Double], polyBaseValue: DenseMatrix[Double]): DenseVector[Double] => Double = (logisticWeightsTruncatedVec: DenseVector[Double]) => {
    val logisticWeights = DenseMatrix.horzcat(logisticWeightsTruncatedVec.toDenseMatrix.reshape(polyBaseValue.cols, conditionalMembershipProbabilities.cols - 1), DenseMatrix.zeros[Double](polyBaseValue.cols, 1))
    val membershipProbabilities = computeMembershipProbabilities(logisticWeights, polyBaseValue)
    val mat = conditionalMembershipProbabilities *:* log(membershipProbabilities.toDenseMatrix.reshape(conditionalMembershipProbabilities.rows, conditionalMembershipProbabilities.cols))
    sum(sum(mat(::, *)))
  }

  def gradQ1Gen(conditionalMembershipProbabilities: DenseMatrix[Double], polyBaseValue: DenseMatrix[Double]): DenseVector[Double] => DenseVector[Double] = (logisticWeightsTruncatedVec: DenseVector[Double]) => {
    val logisticWeights = DenseMatrix.horzcat(logisticWeightsTruncatedVec.toDenseMatrix.reshape(polyBaseValue.cols, conditionalMembershipProbabilities.cols - 1), DenseMatrix.zeros[Double](polyBaseValue.cols, 1))
    val membershipProbabilities = computeMembershipProbabilities(logisticWeights, polyBaseValue)(::, 0 until conditionalMembershipProbabilities.cols - 1)
    gradient(membershipProbabilities, polyBaseValue, conditionalMembershipProbabilities(::, 0 until conditionalMembershipProbabilities.cols - 1))

  }

  def computeEmissionPdf(polyBaseValue: DenseMatrix[Double],
                         polyRegressionCoefficients: List[DenseVector[Double]],
                         polyRegressionVariances: DenseVector[Double],
                         sequence: DenseVector[Double]): DenseMatrix[Double] = {
    polyRegressionCoefficients.indices.map(idx => {
      computeEmissionPdfPerSegment(
        polyBaseValue,
        polyRegressionCoefficients(idx),
        polyRegressionVariances(idx),
        sequence).toDenseMatrix.t
    }
    ).reduce((a, b) => DenseMatrix.horzcat(a, b))
  }

  def computeEmissionPdfForPositivePosteriorBelonging(polyBaseValue: DenseMatrix[Double],
                                                      polyRegressionCoefficients: List[DenseVector[Double]],
                                                      polyRegressionVariances: DenseVector[Double],
                                                      sequence: DenseVector[Double],
                                                      membershipProbabilities: DenseMatrix[Double]): DenseMatrix[Double] = {
    polyRegressionCoefficients.indices.map(idx => {
      computeEmissionPdfPerSegmentForPositivePosteriorBelonging(
        polyBaseValue,
        polyRegressionCoefficients(idx),
        polyRegressionVariances(idx),
        sequence,
        membershipProbabilities(::, idx)).toDenseMatrix.t
    }
    ).reduce((a, b) => DenseMatrix.horzcat(a, b))
  }

  def computeEmissionPdfPerSegmentForPositivePosteriorBelonging(polyBaseValue: DenseMatrix[Double],
                                                                polyRegressionCoefficients: DenseVector[Double],
                                                                polyRegressionVariances: Double,
                                                                sequence: DenseVector[Double],
                                                                membershipProbabilities: DenseVector[Double]): DenseVector[Double] = {

    val sigma = polyRegressionVariances
    val emissionPDF: DenseVector[Double] = DenseVector(sequence.toArray.indices.map(idx => {
      if (membershipProbabilities(idx) > precision) {
        val mu = polyBaseValue(idx, 0 until polyRegressionCoefficients.length) * polyRegressionCoefficients
        max(precision, Gaussian(mu, sigma).pdf(sequence(idx)))
      } else {
        precision
      }
    }).toArray)
    emissionPDF

  }

  def computeEmissionPdfPerSegment(polyBaseValue: DenseMatrix[Double],
                                   polyRegressionCoefficients: DenseVector[Double],
                                   polyRegressionVariances: Double,
                                   sequence: DenseVector[Double]): DenseVector[Double] = {
    val sigma = polyRegressionVariances
    DenseVector.tabulate(sequence.length) { i =>
      val mu = polyBaseValue(i, 0 until polyRegressionCoefficients.length) * polyRegressionCoefficients
      max(precision, Gaussian(mu, sigma).pdf(sequence(i)))
    }
  }
}