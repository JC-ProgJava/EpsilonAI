package tests;

import epsilon.Error;
import epsilon.*;
import epsilon.dataset.MNIST;

import java.io.IOException;

public class Driver {
  public static void main(String[] args) throws IOException {
//    Network network = new Network("/Users/JC/IdeaProjects/EpsilonAI/mynetwork.epsilon");
//    File file = new File("/Users/JC/Desktop/test.csv");
//    FileWriter write = new FileWriter("/Users/JC/Desktop/results.csv");
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

    Vector config = new Vector(new int[]{784, 32, 10});

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.SIGMOID,
            ActivationFunction.SOFTMAX
    };

    Network network = new Network(config, af, Error.CROSS_ENTROPY);
    MNIST data = new MNIST();
    //Network network = new Network("mynetwork.epsilon");
    network.train(data.inputs(), data.target(), 5, 0.002, Optimizer.ADAM);

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