package util;

import java.io.*;
import java.util.Scanner;

public final class initKaggle {
  public static void main(String[] args) throws IOException {
    long start = System.currentTimeMillis();
    double[][] inputValues = new double[42000][784];
    double[][] targetValues = new double[42000][10];

    File file = new File("train.csv");
    Scanner in = new Scanner(file);
    in.nextLine();

    for (int index = 0; index < inputValues.length; index++) {
      if (index % 1000 == 0) {
        System.out.println((index) + " / " + 10000);
      }


      String[] vals = in.nextLine().split(",");

      double[] target = new double[10];
      target[Integer.parseInt(vals[0])] = 1;

      double[] x = new double[784];
      for (int i = 0; i < 784; i++) {
        x[i] = Double.parseDouble(vals[i + 1]) / 255.0;
      }

      inputValues[index] = x;
      targetValues[index] = target;
    }
    in.close();

    try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream("input.ser")))) {
      oos.writeObject(inputValues);
    } catch (IOException ex) {
      ex.printStackTrace();
    }

    try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream("target.ser")))) {
      oos.writeObject(targetValues);
    } catch (IOException ex) {
      ex.printStackTrace();
    }
    long stop = System.currentTimeMillis();
    System.out.println(stop - start + " milliseconds.");
  }
}
