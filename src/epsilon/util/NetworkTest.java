package epsilon.util;

import epsilon.Network;
import epsilon.Vector;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class NetworkTest {
  public static double actual = 0;

  public static void main(String[] args) throws IOException {
    String filename = "/Users/JC/IdeaProjects/EpsilonAI/mynetwork.epsilon";
    ArrayList<String> filenames = new ArrayList<>();
    for (int x = 60001; x <= 70000; x++) {
      filenames.add("/Users/JC/Desktop/Digit-Recogn/data/" + String.format("%05d", x) + ".txt");
    }

    Network network = new Network(filename);
    int corrects = 0;
    for (String z : filenames) {
      double[] a = scan(z);
      int x = getMax(network.test(a));
      corrects += x == actual ? 1 : 0;
    }
    System.out.println(filename);
    System.out.println("Testing: " + corrects + " / " + filenames.size() + " correct.");
  }

  public static double[] scan(String filename) throws FileNotFoundException {
    Scanner in = new Scanner(new File(filename));
    double[] a = new double[784];
    for (int i = 0; i < 784; i++) {
      a[i] = in.nextDouble() / 255;
    }
    actual = in.nextDouble();
    return a;
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
