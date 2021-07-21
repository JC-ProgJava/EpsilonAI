package epsilon;

import java.util.Arrays;

public class Matrix {
  private Vector[] vectors;

  public Matrix(int size) {
    vectors = new Vector[size];
  }

  public Matrix(Vector[] vectorList) {
    vectors = vectorList;
  }

  public Vector[] vectors() {
    return vectors;
  }

  public void set(Vector[] vectors) {
    this.vectors = vectors;
  }

  @Override
  public String toString() {
    return Arrays.toString(vectors);
  }

  public void set(int index, Vector item) {
    vectors[index] = item;
  }

  // Note: This only returns the number of vectors in the matrix,
  // not the number of items in the vectors
  public int length() {
    return vectors.length;
  }

  public Vector get(int index) {
    if (index < 0) {
      throw new IllegalArgumentException("get(index): index must be non-negative, but got '" + index + "'.");
    } else if (index >= length()) {
      throw new IllegalArgumentException("get(index): index out of bounds. Value: '" + index + "'");
    }

    return vectors[index];
  }
}
