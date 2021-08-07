package tests;

import epsilon.fast.Network;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public final class TestFast {
  public static void main(String[] args) throws IOException {
    Network network = new Network("mynetwork.epsilon");
    File file = new File("test.csv");
    FileWriter write = new FileWriter("results.csv");
    write.write("ImageId,Label\n");
    int count = 1;
    Scanner in = new Scanner(file);
    double[] digits = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    while (in.hasNextLine()) {
      String[] a = in.nextLine().split(",");
      double[] aa = new double[784];
      for (int i = 0; i < 784; i++) {
        aa[i] = Double.parseDouble(a[i]) / 255.0;
      }
      write.write(count + "," + getMax(network.test(aa)) + "\n");
      digits[getMax(network.test(aa))]++;
      count++;
    }
    write.close();
    in.close();
    System.out.println("Done!");
    System.out.println(Arrays.toString(digits));
  }

  public static int getMax(double[] output) {
    int index = 0;
    double val = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < output.length; i++) {
      if (output[i] > val) {
        val = output[i];
        index = i;
      }
    }
    return index;
  }
}
