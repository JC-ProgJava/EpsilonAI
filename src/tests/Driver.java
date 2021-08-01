package tests;

import epsilon.*;
import epsilon.Error;
import epsilon.dataset.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public final class Driver {
  public static void main(String[] args) throws IOException {
//    Network network = new Network("mynetwork.epsilon");
//    File file = new File("test.csv");
//    FileWriter write = new FileWriter("results.csv");
//    write.write("ImageId,Label\n");
//    int count = 1;
//    Scanner in = new Scanner(file);
//    double[] digits = new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//    while (in.hasNextLine()) {
//      String[] a = in.nextLine().split(",");
//      double[] aa = new double[784];
//      for (int i = 0; i < 784; i++) {
//        aa[i] = Double.parseDouble(a[i]);
//      }
//      write.write(count + "," + getMax(network.test(aa)) + "\n");
//      digits[getMax(network.test(aa))]++;
//      count++;
//    }
//    write.close();
//    System.out.println("Done!");
//    System.out.println(Arrays.toString(digits));

    Vector config = new Vector(new int[]{784, 256, 10});

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.SIGMOID,
            ActivationFunction.SOFTMAX
    };

    Network network = new Network(config, af, Error.CROSS_ENTROPY);
    MNIST data = new MNIST();
    //Network network = new Network("mynetwork.epsilon");
    network.setVerbose(true);
    network.train(data.inputs(), data.target(), 5, 0.001, Optimizer.ADAM, 100);

    System.out.println(network);
    network.export("mynetwork");
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