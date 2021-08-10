package tests;

import epsilon.Error;
import epsilon.*;
import epsilon.dataset.QMNIST;

public final class Driver {
  public static void main(String[] args) {
    Vector config = new Vector(new int[]{784, 64, 32, 10});

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.SOFTMAX
    };

    Network network = new Network(config, af, Error.CROSS_ENTROPY, InitChoice.GAUSSIAN);
    QMNIST mnist = new QMNIST().subset(0, 110000);

    Network network2 = new Network("mynetwork.epsilon");
    network.setVerbose(true);
    network.useDefaultLearningRate(true);

    QMNIST test = new QMNIST().subset(110000, 120000);

    network.train(mnist.inputs(), mnist.target(), 5, 0.01, Optimizer.ADAM, 1);

    {
      int corrects = 0;
      for (int i = 0; i < mnist.inputs().length; i++) {
        int x = getMax(network.test(mnist.inputs()[i]));
        int actual = -1;
        for (int j = 0; j < mnist.target()[i].length; j++) {
          if (mnist.target()[i][j] == 1) {
            actual = j;
            break;
          }
        }
        corrects += x == actual ? 1 : 0;
      }
      System.out.println("Training: " + corrects + " / " + (mnist.inputs().length) + " correct.");
    }

    {
      int corrects = 0;
      for (int i = 0; i < test.inputs().length; i++) {
        int x = getMax(network.test(test.inputs()[i]));
        int actual = -1;
        for (int j = 0; j < test.target()[i].length; j++) {
          if (test.target()[i][j] == 1) {
            actual = j;
            break;
          }
        }
        corrects += x == actual ? 1 : 0;
      }
      System.out.println("Test: " + corrects + " / " + test.inputs().length + " correct.");
      System.out.println("Accuracy: " + String.format("%.2f", 100.0 * (double) corrects / (double) test.inputs().length) + "%");
    }

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