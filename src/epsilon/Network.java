package epsilon;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

public class Network implements Serializable {
  private Layer[] layers;

  public Network(String filepath) {
    if (!Files.exists(Path.of(filepath))) {
      throw new IllegalArgumentException("Network(file): Please input a valid filepath to the exported network.");
    }

    try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filepath)))) {
      Object[] objects = (Object[]) ois.readObject();
      double[][][] weights = (double[][][]) objects[0];
      ActivationFunction[] activationFunctions = (ActivationFunction[]) objects[1];
      Error type = (Error) objects[2];
      boolean isOutputLayer = false;
      layers = new Layer[weights.length];

      for (int i = 0; i < weights.length; i++) {
        Vector[] vectors = new Vector[weights[i].length];
        for (int j = 0; j < weights[i].length; j++) {
          vectors[j] = new Vector(weights[i][j]);
        }
        if (i == weights.length - 1) {
          isOutputLayer = true;
        }
        Layer matrix = new Layer(vectors, isOutputLayer, activationFunctions[i], type);
        layers[i] = matrix;
      }
    } catch (IOException | ClassNotFoundException | ClassCastException e) {
      e.printStackTrace();
    }
  }

  public Network(Vector config, ActivationFunction[] activationFunction, Error errorType) {
    if ((config.length() - 1) != activationFunction.length) {
      throw new IllegalArgumentException("Network(Vector, ActivationFunction[], Error errorType): Vector length must be one more than ActivationFunction[] length.");
    }

    layers = new Layer[activationFunction.length];

    for (int i = 0; i < config.length() - 1; i++) {
      if (i == layers.length - 1) {
        layers[i] = new Layer(true, activationFunction[i], errorType).fillGaussian((int) config.get(i), (int) config.get(i + 1));
      } else {
        layers[i] = new Layer(false, activationFunction[i], errorType).fillGaussian((int) config.get(i), (int) config.get(i + 1));
      }
    }
  }

  public Network(Vector config, ActivationFunction[] activationFunction, Error errorType, InitChoice initChoice) {
    if ((config.length() - 1) != activationFunction.length) {
      throw new IllegalArgumentException("Network(Vector, ActivationFunction[], Error, InitChoice): Vector length must be one more than ActivationFunction[] length.");
    }

    for (int i = 0; i < config.length() - 1; i++) {
      Layer layer = new Layer(i == (config.length() - 2), activationFunction[i], errorType);
      switch (initChoice) {
        case ZERO:
          layer = layer.fillZeros((int) config.get(i), (int) config.get(i + 1));
          break;
        case RANDOM:
          layer = layer.fillRandom((int) config.get(i), (int) config.get(i + 1));
          break;
        case GAUSSIAN:
          layer = layer.fillGaussian((int) config.get(i), (int) config.get(i + 1));
          break;
      }
      layers[i] = layer;
    }
  }

  public void train(double[][] input, double[][] target, int epoch, double alpha, Optimizer optimizer) {
    if (input.length != target.length) {
      throw new IllegalArgumentException("Input and target (expected) indices must have the same number of examples.");
    }

    for (int iter = 1; iter <= epoch; iter++) {
      for (int index = 0; index < input.length; index++) {
        if (index % 1000 == 0 && index > 0) {
          System.out.println(index + " / " + input.length);
        }
        // todo() <--
        layers[0].feed(new Vector(input[index]));
        // --> Optimize: creates object for each input (60000 total in MNIST)
        int size = layers.length;
        for (int indice = 1; indice < size; indice++) {
          layers[indice].feed(layers[indice - 1].getOutput());
        }

        // todo() <--
        layers[layers.length - 1].learn(new Vector(target[index]), null, alpha, optimizer);
        // --> Optimize: creates object for each target (10000 in MNIST)
        for (int indice = 1; indice < size; indice++) {
          layers[layers.length - indice - 1].learn(null, layers[layers.length - indice], alpha, optimizer);
        }
      }
      if (epoch < 100 || iter % 50 == 0) {
        System.out.println("Epoch: " + iter + " Error: " + layers[layers.length - 1].getDisplayError().total());
      }
    }
  }

  public void export(String name) {
    name += ".epsilon";
    System.out.printf("Exporting to %s.\n", name);

    double[][][] weights = new double[layers.length][][];
    ActivationFunction[] activationFunctions = new ActivationFunction[layers.length];
    Error type = layers[layers.length - 1].getErrorType();
    for (int i = 0; i < layers.length; i++) {
      double[][] layerWeights = new double[layers[i].length()][];
      for (int j = 0; j < layerWeights.length; j++) {
        layerWeights[j] = layers[i].get(j).values();
      }
      activationFunctions[i] = layers[i].getActivationFunction();
      weights[i] = layerWeights;
    }

    Object[] objects = {weights, activationFunctions, type};

    try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(name)))) {
      oos.writeObject(objects);
    } catch (IOException ex) {
      System.err.printf("Cannot export to %s.\n", name);
    }
  }

  public Vector test(double[] input) {
    Vector out = layers[0].test(new Vector(input));
    for (int index = 1; index < layers.length; index++) {
      out = layers[index].test(out);
    }
    return out;
  }
}
