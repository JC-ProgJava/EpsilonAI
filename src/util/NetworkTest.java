package util;

import epsilon.Network;
import epsilon.Vector;
import epsilon.dataset.MNISTValidate;

public final class NetworkTest {
  public static void main(String[] args) {
    String filename = "mynetwork.epsilon";
    Network network = new Network(filename);
    int corrects = 0;
    MNISTValidate validate = new MNISTValidate();
    for (int i = 0; i < validate.inputs().length; i++) {
      int x = getMax(network.test(validate.inputs()[i]));
      int actual = -1;
      for (int j = 0; j < validate.target()[i].length; j++) {
        if (validate.target()[i][j] == 1) {
          actual = j;
          break;
        }
      }
      corrects += x == actual ? 1 : 0;
    }
    System.out.println(filename);
    System.out.println("Testing: " + corrects + " / " + validate.inputs().length + " correct.");
  }

  public static int getMax(Vector output) {
    int index = 0;
    double val = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < output.length(); i++) {
      if (output.get(i) > val) {
        val = output.get(i);
        index = i;
      }
    }
    return index;
  }
}
