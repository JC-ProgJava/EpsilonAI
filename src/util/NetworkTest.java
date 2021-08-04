package util;

import epsilon.Network;
import epsilon.Vector;
import epsilon.dataset.MNIST;

public final class NetworkTest {
  public static void main(String[] args) {
    String filename = "mynetwork.epsilon";
    Network network = new Network(filename);
    int corrects = 0;
    MNIST testData = new MNIST().subset(60000, 70000);
    for (int i = 0; i < testData.inputs().length; i++) {
      int x = getMax(network.test(testData.inputs()[i]));
      int actual = -1;
      for (int j = 0; j < testData.target()[i].length; j++) {
        if (testData.target()[i][j] == 1) {
          actual = j;
          break;
        }
      }
      corrects += x == actual ? 1 : 0;
    }
    System.out.println(filename);
    System.out.println("Testing: " + corrects + " / " + testData.inputs().length + " correct.");
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
