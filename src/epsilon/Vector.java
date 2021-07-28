package epsilon;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class Vector implements Serializable {
  transient private final Random rand = new Random();
  private double[] vect;

  public Vector(int size) {
    vect = new double[size];
  }

  public Vector(int[] vals) {
    assert vect.length == 0;

    vect = new double[vals.length];
    for (int i = 0; i < vals.length; i++) {
      vect[i] = vals[i];
    }
  }

  Vector(double[] vals) {
    assert vect.length == 0;
    vect = vals;
  }

  double[] values() {
    return vect;
  }

  @Override
  public String toString() {
    return Arrays.toString(vect);
  }

  public void set(int index, double item) {
    vect[index] = item;
  }

  public int length() {
    return vect.length;
  }

  public double get(int index) {
    if (index < 0) {
      throw new IllegalArgumentException("get(index): index must be non-negative, but got '" + index + "'.");
    } else if (index >= length()) {
      throw new IllegalArgumentException("get(index): index out of bounds. Value: '" + index + "'");
    }
    return vect[index];
  }

  double total() {
    double sum = 0;
    int size = length();
    for (int index = 0; index < size; index++) {
      sum += get(index);
    }
    return sum;
  }

  Vector add(Vector vec2) {
    if (vec2.length() != length()) {
      throw new IllegalArgumentException("add(Vector): Vectors must be of same length to be added together. Sizes given: {" + length() + ", " + vec2.length() + "}.");
    }

    int size = length();
    Vector sum = new Vector(size);
    for (int i = 0; i < size; i++) {
      sum.set(i, get(i) + vec2.get(i));
    }

    assert sum.length() == size : "Sum length different from original. Implementation error!";

    return sum;
  }

  Vector add(double value) {
    int size = length();
    Vector sum = new Vector(size);
    for (int i = 0; i < size; i++) {
      sum.set(i, get(i) + value);
    }

    return sum;
  }

  Vector subtract(Vector vec2) {
    if (vec2.length() != length()) {
      throw new IllegalArgumentException("subtract(Vector): Vectors must be of same length to be subtracted. Sizes given: {" + length() + ", " + vec2.length() + "}.");
    }

    int size = length();
    Vector diff = new Vector(length());
    for (int i = 0; i < size; i++) {
      diff.set(i, get(i) - vec2.get(i));
    }

    assert diff.length() == size : "Diff length different from original. Implementation error!";

    return diff;
  }

  Vector subtract(double value) {
    int size = length();
    Vector diff = new Vector(size);
    for (int i = 0; i < size; i++) {
      diff.set(i, get(i) - value);
    }

    return diff;
  }

  Vector mult(Vector vec2) {
    if (vec2.length() != length()) {
      throw new IllegalArgumentException("mult(Vector): Vectors must be of same length to be multiplied together. Sizes given: {" + length() + ", " + vec2.length() + "}.");
    }

    int size = length();
    Vector prod = new Vector(length());
    for (int i = 0; i < size; i++) {
      prod.set(i, get(i) * vec2.get(i));
    }

    assert prod.length() == size : "Product vector length different from original. Implementation error!";

    return prod;
  }

  Vector mult(double value) {
    int size = length();
    Vector prod = new Vector(size);
    for (int i = 0; i < size; i++) {
      prod.set(i, get(i) * value);
    }

    return prod;
  }

  Vector fill(double value) {
    for (int i = 0; i < length(); i++) {
      set(i, value);
    }
    return this;
  }

  Vector div(Vector vec2) {
    if (vec2.length() != length()) {
      throw new IllegalArgumentException("div(Vector): Vectors must be of same length to be divided. Sizes given: {" + length() + ", " + vec2.length() + "}.");
    }

    int size = length();
    Vector quo = new Vector(size);
    for (int i = 0; i < size; i++) {
      quo.set(i, get(i) / vec2.get(i));
    }

    assert quo.length() == size : "Quotient vector length different from original. Implementation error!";

    return quo;
  }

  Vector div(double value) {
    int size = length();
    Vector quo = new Vector(size);
    for (int i = 0; i < size; i++) {
      quo.set(i, get(i) / value);
    }

    return quo;
  }

  Vector sqrt() {
    int size = length();
    Vector sqrt = new Vector(size);
    for (int i = 0; i < size; i++) {
      sqrt.set(i, Math.sqrt(get(i)));
    }
    return sqrt;
  }

  Vector log() {
    int size = length();
    Vector log = new Vector(size);
    for (int i = 0; i < size; i++) {
      log.set(i, Math.log(get(i)));
    }
    return log;
  }

  Vector initGaussianDistribute() {
    for (int i = 0; i < length(); i++) {
      set(i, rand.nextGaussian());
    }
    return this;
  }

  Vector fillGaussian(double average, double deviation) {
    for (int i = 0; i < length(); i++) {
      set(i, rand.nextGaussian() * average + deviation);
    }
    return this;
  }

  Vector fillZeros() {
    return fill(0.0);
  }

  Vector fillRandom() {
    for (int i = 0; i < length(); i++) {
      set(i, rand.nextDouble());
    }
    return this;
  }
}