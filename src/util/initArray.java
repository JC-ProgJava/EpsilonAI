package util;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public final class initArray {
  public static void main(String[] args) throws IOException {
    long start = System.currentTimeMillis();
    ArrayList<Integer> arr = new ArrayList<>();
    double[][] inputValues = new double[70000][784];
    double[][] targetValues = new double[70000][10];

    for (int i = 1; i <= 60000; i++) {
      arr.add(i);
    }
    Collections.shuffle(arr);

    ArrayList<Integer> test = new ArrayList<>();
    for (int i = 60001; i <= 70000; i++) {
      test.add(i);
    }
    Collections.shuffle(test);
    arr.addAll(test);

    for (int index = 0; index < 70000; index++) {
      if (index % 1000 == 0) {
        System.out.println((index) + " / " + 70000);
      }

      File file = new File("data/" + String.format("%05d", arr.get(index)) + ".txt");

      FileInputStream fis = new FileInputStream(file);
      byte[] data = new byte[(int) file.length()];
      fis.read(data);
      fis.close();

      Scanner in = new Scanner(file);
      double[] x = new double[784];
      for (int i = 0; i < 784; i++) {
        x[i] = in.nextDouble() / 255;
      }

      double[] target = new double[10];
      target[(int) in.nextDouble()] = 1;

      inputValues[index] = x;
      targetValues[index] = target;
      in.close();
    }

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
