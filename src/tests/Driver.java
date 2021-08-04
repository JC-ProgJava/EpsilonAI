package tests;

import epsilon.Error;
import epsilon.*;
import epsilon.dataset.QMNIST;

public final class Driver {
  public static void main(String[] args) {
    Vector config = new Vector(new int[]{784, 10});

    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.SIGMOID
    };

    Network network2 = new Network(config, af, Error.MEAN_SQUARED);
    QMNIST mnist = new QMNIST().subset(0, 100000);

    Network network = new Network("mynetwork.epsilon");
    network.setVerbose(true);
    network.useDefaultLearningRate(false);
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

    QMNIST test = new QMNIST().subset(100000, 120000);

    network.train(mnist.inputs(), mnist.target(), 5, 0.001, Optimizer.RMSPROP, 1);

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