package tests;

import epsilon.Error;
import epsilon.*;
import epsilon.dataset.ExtendMNIST;
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


    Matrix config = new Matrix(4);
    config.set(0, new Vector(new double[]{784, 128}));
    config.set(1, new Vector(new double[]{128, 64}));
    config.set(2, new Vector(new double[]{64, 32}));
    config.set(3, new Vector(new double[]{32, 10}));

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.SIGMOID,
            ActivationFunction.SIGMOID,
            ActivationFunction.SIGMOID,
            ActivationFunction.SIGMOID
    };


    MNIST data = new MNIST();
    ExtendMNIST data2 = new ExtendMNIST();
    Network network = new Network(config, af, Error.MEAN_SQUARED);
    network.train(data2.inputs(), data2.target(), 5, 0.005, Optimizer.MOMENTUM);
    network.train(data.inputs(), data.target(), 10, 0.01, Optimizer.MOMENTUM);
    network.train(data2.inputs(), data2.target(), 5, 0.005, Optimizer.MOMENTUM);
    network.train(data.inputs(), data.target(), 5, 0.005, Optimizer.MOMENTUM);

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