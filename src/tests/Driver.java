package tests;

import epsilon.*;
import epsilon.dataset.MNIST;

public class Driver {
  public static void main(String[] args) {
    Matrix config = new Matrix(1);
    config.set(0, new Vector(new double[]{784, 10}));
    ActivationFunction[] af = new ActivationFunction[]{
            ActivationFunction.SIGMOID
    };
    MNIST data = new MNIST();
    Network network = new Network(config, af);
    network.train(data.inputs(), data.target(), 5, 0.001, Optimizer.ADAM);
    network.export("mynetwork");
  }
}