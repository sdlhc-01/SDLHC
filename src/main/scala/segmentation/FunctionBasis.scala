package segmentation

import breeze.numerics.sqrt

import scala.annotation.tailrec

object FunctionBasis {

  def addFunc(f1: Double => Double, f2: Double => Double): Double => Double = (x: Double) => {
    f1(x) + f2(x)
  }

  def prodFunc(f1: Double => Double, f2: Double => Double): Double => Double = (x: Double) => {
    f1(x) * f2(x)
  }

  def scalarFunc(alpha: Double, f: Double => Double): Double => Double = (x: Double) => {
    alpha * f(x)
  }

  def f1Gen(method: String): Double => Double = method match {
    case "HermitePhysic" | "ChebychevSecond" => x: Double => 2.0 * x
    case _ => x: Double => x
  }

  def recurrentRelationGen(method: String) = method match {
    case "Canonical" => (f1: Double => Double, fn_1: Double => Double, fn_2: Double => Double, degree: Int) => prodFunc(f1, fn_1)
    case "Legendre" => (f1: Double => Double, fn_1: Double => Double, fn_2: Double => Double, degree: Int) => scalarFunc(1 / (degree + 1.0), addFunc(scalarFunc(2 * degree + 1.0, prodFunc(f1, fn_1)), scalarFunc(-1.0 * degree, fn_2)))
    case "HermiteProba" => (f1: Double => Double, fn_1: Double => Double, fn_2: Double => Double, degree: Int) => addFunc(prodFunc(f1, fn_1), scalarFunc(-1.0 * degree, fn_2))
    case "HermitePhysic" => (f1: Double => Double, fn_1: Double => Double, fn_2: Double => Double, degree: Int) => addFunc(prodFunc(f1, fn_1), scalarFunc(-2.0 * degree, fn_2))
    case "ChebychevFirst" => (f1: Double => Double, fn_1: Double => Double, fn_2: Double => Double, degree: Int) => addFunc(scalarFunc(2.0, prodFunc(f1, fn_1)), scalarFunc(-1.0, fn_2))
    case "ChebychevSecond" => (f1: Double => Double, fn_1: Double => Double, fn_2: Double => Double, degree: Int) => addFunc(scalarFunc(1.0, prodFunc(f1, fn_1)), scalarFunc(-1.0, fn_2))
  }

  def normalizationGen(method: String) = method match {
    case "Canonical" => (f: Double => Double, degree: Int) => f
    case "Legendre" => (f: Double => Double, degree: Int) => scalarFunc(1 / sqrt(2 * degree + 1.0), f)
    case "ChebychevSecond" => (f: Double => Double, degree: Int) => scalarFunc(1 / sqrt(Math.PI / 2), f)
    case "HermitePhysic" => (f: Double => Double, degree: Int) => scalarFunc(1 / sqrt(Math.pow(2, degree) * Tools.factorial(degree) * sqrt(Math.PI)), f)
  }

  def PolynomialBasis(degree: Int, method: String, normalization: Boolean = true): Array[Double => Double] = {
    val f0 = (x: Double) => 1.0
    val f1 = f1Gen(method)
    val RecurrentRelation = recurrentRelationGen(method)

    @tailrec
    def go(degree: Int, basis: Array[Double => Double]): Array[Double => Double] = {
      if (degree == 0) {
        Array(f0)
      } else if (degree == 1) {
        Array(f0, f1)
      } else {
        if (basis.length == degree + 1) {
          basis
        } else {
          go(degree, basis :+ RecurrentRelation(f1, basis.takeRight(2)(1), basis.takeRight(2)(0), basis.length - 1))
        }
      }
    }

    val functionBasis = go(degree, Array(f0, f1)).slice(0, degree + 1)

    if (normalization) {
      val normalize = normalizationGen(method)
      val normalizedFunctionBasis = functionBasis.indices.map(idx => normalize(functionBasis(idx), idx)).toArray
      normalizedFunctionBasis
    } else {
      functionBasis
    }

  }
}
