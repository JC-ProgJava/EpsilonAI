package tests;

import epsilon.ActivationFunction;
import epsilon.Error;
import epsilon.Optimizer;
import epsilon.dataset.MNIST;
import epsilon.fast.Network;

public final class DriverFast {
  public static void main(String[] args) {
    double[] config = new double[]{784, 64, 32, 10};

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.SIGMOID,
            ActivationFunction.LEAKY_RELU,
            ActivationFunction.SOFTMAX
    };

    Network network = new Network(config, af, Error.CROSS_ENTROPY);
    MNIST mnist = new MNIST().subset(0, 60000);

    Network network2 = new Network("mynetwork.epsilon");
    network.setVerbose(true);
    network.useDefaultLearningRate(true);
    /*
    MEAN_SQUARED
    Sigmoid
    Batch-20
    Alpha: 0.001
    Epoch: 5
    RMSPROP - 8972
    ADAM - 8898
    NONE - 8759
    MOMENTUM - 8753
    ADADELTA - 8475
    ADAGRAD - 8020

    CROSS_ENTROPY
    Softmax
    Batch-20
    Alpha: 0.001
    Epoch: 5
    ADAM - 8891


    MEAN_SQUARED
    Sigmoid
    Alpha: 0.001
    Epoch: 15
    Batch-1
      RMSPROP - 9053, 9042
    Batch-10
      RMSPROP - 9050
    Batch-20
      RMSPROP - 9019
    Batch-40
      RMSPROP - 9031
    Batch-100
      RMSPROP - 9025
     */

    MNIST test = new MNIST().subset(60000, 70000);

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