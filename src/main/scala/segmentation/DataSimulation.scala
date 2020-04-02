package segmentation

import breeze.numerics.{abs, sin}

object DataSimulation {
  implicit def bool2int(b: Boolean) = if (b) 1 else 0

  // One Segment first degree
  def f0(x: Double): Double = {
    x
  }

  // Two Segments first degree
  def f1(x: Double): Double = {
    abs(x - 75)
  }

  // Two Segments first degree
  def f1_1(x: Double): Double = {
    abs(x - 20)
  }

  // Two Segments Second degree
  def f2(x: Double): Double = {
    if (x < 75) {
      (x - 75) * (x - 75) / 150
    } else {
      -(x - 75) * (x - 75) / 75
    }
  }

  // Three Segments Second degree
  def f2_2(x: Double): Double = {
    if (x < 60) {
      (x - 40) * (x - 40) / 20
    } else if (x < 100) {
      -(x - 80) * (x - 80) / 20 + 40
    } else {
      (x - 120) * (x - 120) / 20
    }
  }

  // 5 Segments, Second degree
  def f3(x: Double): Double = {
    abs(abs(x - 56) - 0.00005 * x * x * x + 0.3 * x - 20)
  }

  // Dirac
  def f4(x: Double): Double = {
    if (x == 50) 1.0 else 0D
  }

  // Rectangular function
  def f5(x: Double): Double = {
    0D + (x > 50.0) * 1D + (x > 100.0) * (-1D)
  }

  // Rectangular function
  def f5_1(x: Double): Double = {
    0D + (x % 75 < 30.0) * 1D + (x % 75 >= 30.0) * (-1D)
  }

  // Rectangular function
  def f5_2(x: Double): Double = {
    0D + (x % 75 < 37.0) * 1D + (x % 75 >= 37.0) * (-1D)
  }

  // Two segment, one segment as 2nd degree polynom, the second as 1st degree polynom
  def f6(x: Double): Double = {
    (x <= 75D) * ((x - 50D) * (x - 50D)) / 50D + (x > 75D) * (-25D + x / 2D)
  }

  // Two segment, one segment as 2nd degree polynom, the second as 1st degree polynom, the 3rd as 1st degree
  def f7(x: Double): Double = {
    (x <= 50D) * ((x - 33D) * (x - 33D)) / 50D + (x > 50D) * (x <= 101D) * (-19.22 + x / 2D) + (x > 101D) * (30.78)
  }

}
