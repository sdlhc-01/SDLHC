package segmentation

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, argmin, inv, max, min, sum}
import breeze.numerics._
import breeze.stats.distributions.RandBasis
import segmentation.EMFuncSegState.makeModelEvolve
import segmentation.Tools._

import scala.annotation.tailrec

package object EM {

  val precision = 1E-8
  val relativeConvergenceThreshold = 1E-4

  def computeConditionalSegmentMembership(model: EMFuncSegState, polyBaseValue: DenseMatrix[Double], sequence: DenseVector[Double]): DenseMatrix[Double] = {
    val nSegments = model.polyRegressionVariances.length
    val sumPiEmissionPdf = sum(model.jointLogDistrib(*, ::))
    val sumPiEmissionPdfMatrix: DenseMatrix[Double] = sumPiEmissionPdf * DenseVector.ones[Double](nSegments).t
    val res = max(model.jointLogDistrib /:/ sumPiEmissionPdfMatrix, precision)
    Tools.normalizeMatrix(res)
  }

  def UpdateRegressionCoefficient(polyRegressionCoefficientsLength: List[Int],
                                  sequence: DenseVector[Double],
                                  polyBaseValue: DenseMatrix[Double],
                                  conditionalSegmentMembership: DenseMatrix[Double],
                                  verbose: Boolean = false): List[DenseVector[Double]] = {
    polyRegressionCoefficientsLength.indices.map(
      seg => {
        Tools.weightedLeastSquareResolutionForWeightsHigherThanPrecision(polyBaseValue(::, 0 until polyRegressionCoefficientsLength(seg)),
          sequence,
          conditionalSegmentMembership(::, seg)
        )
      }).toList
  }

  def UpdateRegressionVariances(nSegments: Int,
                                sequence: DenseVector[Double],
                                polyBaseValue: DenseMatrix[Double],
                                conditionalSegmentMembership: DenseMatrix[Double],
                                polyRegressionCoefficients: List[DenseVector[Double]]): DenseVector[Double] = {

    DenseVector(polyRegressionCoefficients.indices.map(seg => {
      val mode = polyBaseValue(::, 0 until polyRegressionCoefficients(seg).length) * polyRegressionCoefficients(seg)
      val weights = conditionalSegmentMembership(::, seg)
      sqrt(weightedVariance(sequence, mode, weights))
    }).toArray)

  }

  def updateLogisticWeights(nSegments: Int, logisticWeights: DenseMatrix[Double],
                            polyBaseValue: DenseMatrix[Double],
                            polyBaseValueW: DenseMatrix[Double],
                            conditionalMembershipProbabilities: DenseMatrix[Double],
                            polyRegressionVariances: DenseVector[Double],
                            polyRegressionCoefficients: List[DenseVector[Double]],
                            sequence: DenseVector[Double]): Vector[DenseMatrix[Double]] = {

    // Iterative Reweighted Least Square
    @tailrec
    def go(nIterIRLSLeft: Int,
           nIterMax: Int,
           logisticWeights: DenseVector[Double],
           segmentMembership: DenseMatrix[Double],
           conditionalSegmentMembership: DenseMatrix[Double]): Vector[DenseMatrix[Double]] = {

      if (nIterIRLSLeft == 0) {
        val solution = DenseMatrix.horzcat(logisticWeights.toDenseMatrix.reshape(polyBaseValueW.cols, nSegments - 1), DenseMatrix.zeros[Double](polyBaseValueW.cols, 1))
        Vector(solution, computeMembershipProbabilities(solution, polyBaseValueW))
      } else {
        val g = gradient(segmentMembership(::, 0 until nSegments - 1), polyBaseValueW, conditionalSegmentMembership(::, 0 until nSegments - 1))
        val H = hessian(segmentMembership(::, 0 until nSegments - 1), polyBaseValueW)
        val logisticWeigthsMat = DenseMatrix.horzcat(logisticWeights.toDenseMatrix.reshape(polyBaseValueW.cols, nSegments - 1), DenseMatrix.zeros[Double](polyBaseValueW.cols, 1))
        var newLogisticWeights = logisticWeights - inv(H) * g
        var newLogisticWeightsMat = DenseMatrix.horzcat(newLogisticWeights.toDenseMatrix.reshape(polyBaseValueW.cols, nSegments - 1), DenseMatrix.zeros[Double](polyBaseValueW.cols, 1))

        val oldLikelihood = new EMFuncSegState(polyRegressionCoefficients,
          polyRegressionVariances,
          conditionalSegmentMembership,
          logisticWeigthsMat,
          computeMembershipProbabilities(logisticWeigthsMat, polyBaseValueW),
          polyBaseValue,
          sequence).scores("logLikelihood")

        var k = 1
        while (oldLikelihood > new EMFuncSegState(polyRegressionCoefficients,
          polyRegressionVariances,
          conditionalSegmentMembership,
          newLogisticWeightsMat,
          computeMembershipProbabilities(newLogisticWeightsMat, polyBaseValueW),
          polyBaseValue,
          sequence).scores("logLikelihood")) {
          k += 1
          newLogisticWeights = logisticWeights - (1 / Math.pow(2, k)) * inv(H) * g
          newLogisticWeightsMat = DenseMatrix.horzcat(newLogisticWeights.toDenseMatrix.reshape(polyBaseValueW.cols, nSegments - 1), DenseMatrix.zeros[Double](polyBaseValueW.cols, 1))
        }
        val newProbBelongingToSegment = computeMembershipProbabilities(newLogisticWeightsMat, polyBaseValueW)
        go(nIterIRLSLeft - 1, nIterMax, newLogisticWeights, newProbBelongingToSegment, conditionalSegmentMembership)
      }
    }

    val nIterIRLS = 2
    val segmentMembership = computeMembershipProbabilities(logisticWeights, polyBaseValueW)
    val w_old = logisticWeights(::, 0 until nSegments - 1).toDenseVector
    go(nIterIRLS, nIterIRLS, w_old, segmentMembership, conditionalMembershipProbabilities)
  }

  def computeMembershipProbabilities(logisticWeights: DenseMatrix[Double],
                                     polyBaseValueW: DenseMatrix[Double]): DenseMatrix[Double] = {
    val res = DenseMatrix((0 until logisticWeights.cols).flatMap(k => {
      val deltas = DenseMatrix((0 until logisticWeights.cols).flatMap(h =>
        (logisticWeights(::, h) - logisticWeights(::, k)).toArray): _*).reshape(logisticWeights.rows, logisticWeights.cols) // polyDegree x nSegment
      val polyDelta = polyBaseValueW * deltas
      val expPolyDeltas = polyDelta.map(x => exp(x))
      sum(expPolyDeltas(*, ::)).map(x => max(1 / x, precision)).toArray
    }): _*).reshape(polyBaseValueW.rows, logisticWeights.cols)

    Tools.normalizeMatrix(res)

  }

  def softEstim(polyRegressionCoefficients: List[DenseVector[Double]], membershipProbabilities: DenseMatrix[Double], functionBasis: Array[Double => Double], times: DenseVector[Double]): DenseVector[Double] = {
    val timesCenteredReduced = 2.0 * ((times - times(0)) / (times(-1) - times(0))) - 1.0
    val polyBaseValue: DenseMatrix[Double] = initPolyBaseValue(timesCenteredReduced, polyRegressionCoefficients.map(_.length).max)
    polyRegressionCoefficients.indices.map(idx => membershipProbabilities(::, idx) *:* (polyBaseValue(::, 0 until polyRegressionCoefficients(idx).length) * polyRegressionCoefficients(idx))).reduce((a, b) => a + b)
  }

  def gradient(membershipProbabilities: DenseMatrix[Double], polyBaseValue: DenseMatrix[Double], conditionalSegmentMembership: DenseMatrix[Double]): DenseVector[Double] = {
    val deltas = (conditionalSegmentMembership - membershipProbabilities).t * polyBaseValue
    deltas.t.toDenseVector
  }

  def hessian(membershipProbabilities: DenseMatrix[Double], polyBaseValue: DenseMatrix[Double]): DenseMatrix[Double] = {
    (0 until membershipProbabilities.cols).map(seg1 =>
      (0 until membershipProbabilities.cols).map(seg2 => {
        hessianBlock(seg1, seg2, membershipProbabilities, polyBaseValue)
      }).reduce((a, b) => DenseMatrix.horzcat(a, b))
    ).reduce((a, b) => DenseMatrix.vertcat(a, b))
  }

  def hessianBlock(segment1: Int, segment2: Int, segmentMembership: DenseMatrix[Double], polyBaseValue: DenseMatrix[Double]): DenseMatrix[Double] = {
    if (segment1 == segment2) {
      -(0 until segmentMembership.rows).map(i => segmentMembership(i, segment1) * (1.0 - segmentMembership(i, segment2)) * polyBaseValue(i, ::).t * polyBaseValue(i, ::)).reduce(_ + _)
    } else {
      (0 until segmentMembership.rows).map(i => segmentMembership(i, segment1) * segmentMembership(i, segment2) * polyBaseValue(i, ::).t * polyBaseValue(i, ::)).reduce(_ + _)
    }
  }

  def initPolyBaseValue(times: DenseVector[Double], polyDegree: Int = 2): DenseMatrix[Double] = {
    val functionBasis = FunctionBasis.PolynomialBasis(polyDegree, "Legendre")
    functionBasis.indices.map(p => DenseMatrix(times.map(x => functionBasis(p)(x)).toArray).t).reduce((a, b) => DenseMatrix.horzcat(a, b))
  }

  def EMAlgo(nIter: Int,
             functionBasis: Array[Double => Double],
             nSegments: Int,
             times: DenseVector[Double],
             sequence: DenseVector[Double],
             optimModelStrategy: String = "None",
             EMInitState: EMFuncSegState = new EMFuncSegState(),
             evaluationCriterion: String = "BIC",
             verbose: Boolean = false): Vector[EMFuncSegState] = {

    checkRequirements(nIter, functionBasis, nSegments, times, sequence, optimModelStrategy)

    val timesCenteredReduced = 2.0 * ((times - times(0)) / (times(-1) - times(0))) - 1.0
    val polyBaseValueW = initPolyBaseValue(timesCenteredReduced)
    var polyBaseValue = {
      if (EMInitState.conditionalMembershipProbabilities.rows == 0) {
        initPolyBaseValue(timesCenteredReduced, polyDegree = functionBasis.length - 1)
      } else initPolyBaseValue(timesCenteredReduced, polyDegree = EMInitState.polyRegressionCoefficients.map(_.length).max)
    }

    val log: Vector[EMFuncSegState] = {
      if (optimModelStrategy == "AddAugmentedSegment") {
        require(EMInitState.conditionalMembershipProbabilities.rows == 0,
          "Error : optimModelStrategy asked is addAugmentedSegment (which requires augmentDegree initialisation) " +
            "but another EM initialized state has been furnished in the same time. Please choose between those two options)")
        val resEM = EMAlgo(nIter = 1000, functionBasis, nSegments, times, sequence, optimModelStrategy = "augmentPolyDegree")
        resEM.slice(0, argmax(-DenseVector[Double](resEM.map(model => model.scores(evaluationCriterion)).toArray)))
      } else if (EMInitState.conditionalMembershipProbabilities.rows == 0) {
        Vector(new EMFuncSegState(List.fill(nSegments)(functionBasis.length), nSegments, polyBaseValue, polyBaseValueW, sequence))
      } else {
        Vector(EMInitState)
      }
    }

    @tailrec
    def go(nIterMax: Int,
           currentMinScore: Double,
           sequence: DenseVector[Double],
           polyBaseValue: DenseMatrix[Double],
           log: Vector[EMFuncSegState]
          ): Vector[EMFuncSegState] = {

      val modelPreviousIteration = log.last
      val polyRegressionCoefficients = log.last.polyRegressionCoefficients
      val logisticWeights = log.last.logisticWeights
      val nSegments = log.last.polyRegressionVariances.length

      if (verbose) {
        println("EM Iteration : ".concat(log.length.toString).concat(", nSegments = ").concat(nSegments.toString).concat("\r"))
        println(log.last.scores)
      }

      // Expectation step p(z|x;theta)
      val newConditionalSegmentMembership = computeConditionalSegmentMembership(modelPreviousIteration, polyBaseValue, sequence)

      // Maximisation Step
      val newPolyRegressionCoefficients = UpdateRegressionCoefficient(polyRegressionCoefficients.map(_.length),
        sequence, polyBaseValue, newConditionalSegmentMembership, verbose = verbose)
      val newPolyRegressionVariances = UpdateRegressionVariances(nSegments, sequence, polyBaseValue, newConditionalSegmentMembership, newPolyRegressionCoefficients)
      val resList = updateLogisticWeights(nSegments, logisticWeights, polyBaseValue, polyBaseValueW, newConditionalSegmentMembership, newPolyRegressionVariances, newPolyRegressionCoefficients, sequence)
      val newLogisticWeights = resList(0)
      val newProbBelongingToSegment = resList(1)
      val newLog: Vector[EMFuncSegState] = log :+ new EMFuncSegState(newPolyRegressionCoefficients, newPolyRegressionVariances, newConditionalSegmentMembership, newLogisticWeights, newProbBelongingToSegment, polyBaseValue, sequence)

      // nIter Check
      if (newLog.length >= nIterMax) {
        if (verbose) {
          println("EM Iteration : ".concat(newLog.length.toString).concat(", nSegments = ").concat(nSegments.toString).concat(": nIterMax reached"))
        }
        return newLog
      }

      // Convergence check
      if (abs((newLog.last.scores(evaluationCriterion) - log.last.scores(evaluationCriterion)) / log.last.scores(evaluationCriterion)) > relativeConvergenceThreshold) {
        // No Convergence
        go(nIterMax, min(currentMinScore, newLog.last.scores(evaluationCriterion)), sequence, polyBaseValue, newLog)
      } else {
        // Convergence

        if (optimModelStrategy != "None") {
          // Optimisation strategy is planned
          val Scores = DenseVector[Double](newLog.dropRight(1).map(model => model.scores(evaluationCriterion)).toArray)

          // Is current model optimal Score better than overall optimal Score ?
          if (min(currentMinScore, newLog.last.scores(evaluationCriterion)) <= min(Scores) && Scores.toArray.count(_ == min(Scores)) == 1) {
            // IF yes : Score is decreasing

            // Apply strategy
            val newLogWithNewModel = newLog ++ makeModelEvolve(newLog.last, functionBasis, polyBaseValue, polyBaseValueW, sequence, timesCenteredReduced, optimModelStrategy)

            if (optimModelStrategy == "augmentPolyDegree" || optimModelStrategy == "addSegmentOrAugmentPolyDegree" || optimModelStrategy == "addAugmentedSegment") {
              go(nIterMax,
                newLogWithNewModel.last.scores(evaluationCriterion),
                sequence,
                initPolyBaseValue(
                  timesCenteredReduced,
                  polyDegree = newLogWithNewModel.last.polyRegressionCoefficients.map(coef => coef.length).max),
                newLogWithNewModel)
            } else {
              go(nIterMax, newLogWithNewModel.last.scores(evaluationCriterion), sequence, polyBaseValue, newLogWithNewModel)
            }
          } else {
            //          Score is increasing or stagnating
            if (verbose) {
              println("EM Iteration : ".concat(newLog.length.toString)
                .concat(", nSegments = ")
                .concat(nSegments.toString)
                .concat(": Convergence after strategy ".concat(optimModelStrategy).concat(" applied")))
            }
            newLog
          }
        } else {
          if (verbose) {
            println("EM Iteration : "
              .concat(newLog.length.toString).concat(", nSegments = ")
              .concat(nSegments.toString)
              .concat(": Convergence without planned strategy"))
          }
          newLog
        }
      }
    }

    polyBaseValue = initPolyBaseValue(
      timesCenteredReduced,
      polyDegree = log.last.polyRegressionCoefficients.map(coef => coef.length).max - 1)

    go(nIter, log.last.scores(evaluationCriterion), sequence, polyBaseValue, log)
  }

  def checkRequirements(nIter: Int, functionBasis: Array[Double => Double], nSegments: Int, times: DenseVector[Double], sequence: DenseVector[Double], optimModel: String): Unit = {
    require(nIter >= 0, "Error during param_EM construction, The number of iterations cannot be <=0")
    require(nSegments > 1, "Error during param_EM construction, the number of segments should be >=2")
    require(times.toScalaVector().sorted == times.toScalaVector(), "Vector of times is not sorted.. ")
    require(functionBasis.length > 0, "Error during param_EM construction, The length of the function basis  cannot be <=0")
    require(nSegments >= 1, "Error during param_EM construction, The number of segments cannot be <1")
    require(times.toArray.distinct.length == times.length, "Error during param_EM construction, There can't be duplicate values in 'times' argument")
    require(times.length == sequence.length, "Error during param_EM construction, Arguments length don't match : times and sequence")
    require(sequence.length / nSegments > 10, "Error during param_EM construction, Time serie length must be greater than (10*nSegments)")
    require(List("None", "addSegment", "augmentPolyDegree", "addSegmentOrAugmentPolyDegree", "addAugmentedSegment").contains(optimModel), "Error: optional parameter optimModelStrategy should be in ('None' (Default), 'addSegment','augmentPolyDegree','addSegmentOrAugmentPolyDegree','addAugmentedSegment')")
  }

  def getBestModel(data: DenseMatrix[Double],
                   functionBasis: Array[Double => Double],
                   optimModelStrategy: String = "None",
                   nIter: Int = 500,
                   nSegments: Int = 2,
                   evaluationCriterion: String = "BIC",
                   verbose: Boolean = false): EMFuncSegState = {

    require(data.cols == 2, "Error: data matrix should have 2 columns (times and values)")

    val times = data(::, 0)
    val sequence = data(::, 1)
    val res = EM.EMAlgo(nIter, functionBasis, nSegments, times, sequence,
      optimModelStrategy = optimModelStrategy,
      evaluationCriterion = evaluationCriterion,
      verbose = verbose)
    val Scores = DenseVector[Double](res.map(model => model.scores(evaluationCriterion)).toArray)

    res(argmin(Scores))
  }

}
