package tests;

import epsilon.ActivationFunction;
import epsilon.Error;
import epsilon.Optimizer;
import epsilon.Regularization;
import epsilon.dataset.QMNIST;
import epsilon.fast.Network;

public final class DriverFast {
  public static void main(String[] args) {
    double[] config = new double[]{784, 128, 10};

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.SOFTMAX
    };

    Network network2 = new Network(config, af, Error.CROSS_ENTROPY).regularize(Regularization.L2);
    network2.initCustomDistributedGaussian(0.0, 0.1);
    QMNIST mnist = new QMNIST().subset(0, 100000);

    Network network = new Network("mynetwork.epsilon");
    network.setVerbose(false);
    network.useDefaultLearningRate(true);

    QMNIST test = new QMNIST().subset(100000, 120000);

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
      System.out.println("Accuracy: " + String.format("%.2f", 100.0 * (double) corrects / (double) mnist.inputs().length) + "%");
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