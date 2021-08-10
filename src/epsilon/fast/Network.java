package epsilon.fast;

import epsilon.Error;
import epsilon.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public final class Network implements Serializable {
  private ArrayList<Regularization> regularizations = new ArrayList<>();
  private Layer[] layers;
  private boolean verbose = false;
  private boolean useDefaultLearningRate = false;

  public Network(String filepath) {
    if (!Files.exists(Path.of(filepath))) {
      throw new IllegalArgumentException("Network(file): Please input a valid filepath to the exported network.");
    }

    try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filepath)))) {
      Object[] objects = (Object[]) ois.readObject();

      if (objects.length < 5) {
        throw new IllegalArgumentException("Network(file): You're network cannot be imported with this version of EpsilonAI. Consider reverting to a previous version.\nSorry for all inconveniences caused.");
      }

      double[][][] weights = (double[][][]) objects[0];
      double[][] bias = (double[][]) objects[1];
      ActivationFunction[] activationFunctions = (ActivationFunction[]) objects[2];
      Error type = (Error) objects[3];
      ArrayList<Regularization> regularization = (ArrayList<Regularization>) objects[4];
      this.regularizations = regularization;
      boolean isOutputLayer = false;
      layers = new Layer[weights.length];

      for (int i = 0; i < weights.length; i++) {
        double[][] vectors = new double[weights[i].length][];
        System.arraycopy(weights[i], 0, vectors, 0, weights[i].length);
        if (i == weights.length - 1) {
          isOutputLayer = true;
        }
        Layer matrix = new Layer(vectors, bias[i], isOutputLayer, activationFunctions[i], type).setRegularization(regularization);
        layers[i] = matrix;
      }
    } catch (IOException | ClassNotFoundException | ClassCastException e) {
      System.err.println("An error occurred when importing an exported network.");
      System.err.println(e);
    }
  }

  public Network(double[] config, ActivationFunction[] activationFunction, Error errorType) {
    if ((config.length - 1) != activationFunction.length) {
      throw new IllegalArgumentException("Network(Vector, ActivationFunction[], Error errorType): Vector length must be one more than ActivationFunction[] length.");
    }

    layers = new Layer[activationFunction.length];

    for (int i = 0; i < config.length - 1; i++) {
      if (i == layers.length - 1) {
        layers[i] = new Layer(true, activationFunction[i], errorType).fillGaussian((int) config[i], (int) config[i + 1]);
      } else {
        layers[i] = new Layer(false, activationFunction[i], errorType).fillGaussian((int) config[i], (int) config[i + 1]);
      }
    }
  }

  public Network(double[] config, ActivationFunction[] activationFunction, Error errorType, InitChoice initChoice) {
    if ((config.length - 1) != activationFunction.length) {
      throw new IllegalArgumentException("Network(Vector, ActivationFunction[], Error, InitChoice): Vector length must be one more than ActivationFunction[] length.");
    }

    layers = new Layer[activationFunction.length];

    for (int i = 0; i < config.length - 1; i++) {
      Layer layer = new Layer(i == (config.length - 2), activationFunction[i], errorType);
      switch (initChoice) {
        case ZERO:
          layer.fillZeros((int) config[i], (int) config[i + 1]);
          break;
        case RANDOM:
          layer.fillRandom((int) config[i], (int) config[i + 1]);
          break;
        case GAUSSIAN:
          layer.fillGaussian((int) config[i], (int) config[i + 1]);
          break;
        default:
          throw new IllegalArgumentException("Network(): No such InitChoice '" + initChoice + "'.");
      }
      layers[i] = layer;
    }
  }

  public Network(Layer[] layers) {
    this.layers = layers;
  }

  public void initCustomDistributedGaussian(double average, double deviation) {
    if (layers == null) {
      throw new IllegalArgumentException("initCustomDistributeGaussian(): Please use this function after network initialization.");
    }

    for (Layer layer : layers) {
      layer.fillGaussian(average, deviation);
    }
  }

  public Network regularize(Regularization regularizationTechnique) {
    this.regularizations.add(regularizationTechnique);
    for (Layer layer : layers) {
      layer.setRegularization(regularizations);
    }
    return this;
  }

  public void removeAllRegularization() {
    regularizations.clear();
  }

  public void setVerbose(boolean isVerbose) {
    verbose = isVerbose;
  }

  public void useDefaultLearningRate(boolean isDefaultLearningRate) {
    this.useDefaultLearningRate = isDefaultLearningRate;
  }

  public void train(double[][] input, double[][] target, int epoch, double alpha, Optimizer optimizer, int BATCH_SIZE) {
    if (input.length != target.length) {
      throw new IllegalArgumentException("train(): Input and target (expected) indices must have the same number of examples.");
    }

    if (BATCH_SIZE <= 0) {
      throw new IllegalArgumentException("train(): Batch size must be positive.");
    } else if (input.length % BATCH_SIZE != 0) {
      throw new IllegalArgumentException("train(): Batch size must evenly divide input dataset.");
    }

    for (Layer layer : layers) {
      layer.setBatchSize(BATCH_SIZE);
    }

    if (useDefaultLearningRate) {
      if (optimizer == Optimizer.ADAGRAD || optimizer == Optimizer.NONE || optimizer == Optimizer.MOMENTUM || optimizer == Optimizer.NESTEROV) {
        alpha = 0.01;
      } else if (optimizer == Optimizer.RMSPROP || optimizer == Optimizer.ADAM) {
        alpha = 0.001;
      } else if (optimizer == Optimizer.ADADELTA) {
        alpha = 1.0;
      }
    }

    {
      long start = System.currentTimeMillis();
      for (int i = 0; i < 1000; i++) {
        layers[0].feed(input[i]);
        int size = layers.length;
        for (int indice = 1; indice < size; indice++) {
          layers[indice].feed(layers[indice - 1].getOutput());
        }

        layers[layers.length - 1].learn(target[i], null, alpha, optimizer);
        for (int indice = 1; indice < size; indice++) {
          layers[layers.length - indice - 1].learn(null, layers[layers.length - indice], alpha, optimizer);
        }
      }
      double time = (System.currentTimeMillis() - start); // time elapsed (ms)
      time *= input.length * epoch; // multiply by number of epoch and number of examples
      time /= 1000.0; // divide by 1000 to get seconds (s)
      time /= 1000.0; // divide by 1000 to get average per sample
      if ((time / 60.0) > 60.0) {
        System.out.println("Training will take approximately " + Math.round(time / 3600.0) + " hours.");
      } else {
        System.out.println("Training will take approximately " + Math.round(time / 60.0) + " minutes.");
      }
    }

    for (int iter = 1; iter <= epoch; iter++) {
      long start = System.currentTimeMillis();
      for (int index = 0; index < input.length; index++) {
        if (index % 1000 == 0 && index > 0 && verbose) {
          System.out.println(index + " / " + input.length);
        }

        layers[0].feed(input[index]);
        int size = layers.length;
        for (int indice = 1; indice < size; indice++) {
          layers[indice].feed(layers[indice - 1].getOutput());
        }

        layers[layers.length - 1].learn(target[index], null, alpha, optimizer);
        for (int indice = 1; indice < size; indice++) {
          layers[layers.length - indice - 1].learn(null, layers[layers.length - indice], alpha, optimizer);
        }
      }
      if (Double.isNaN(Layer.total(layers[layers.length - 1].getError()))) {
        System.err.println("Error exploded, producing NaN (not a number). Program terminated automatically.");
        System.exit(-1);
      }
      if (epoch < 100 || iter % 50 == 0) {
        System.out.println("Epoch: " + iter + " Error: " + Layer.total(layers[layers.length - 1].getDisplayError()) + " Time: " + ((System.currentTimeMillis() - start) / 1000.0) + " seconds.");
      }
    }

    for (Layer layer : layers) {
      layer.clearCache();
    }
  }

  public void export(String name) {
    name += ".epsilon";
    System.out.printf("Exporting to %s.\n", name);

    double[][][] weights = new double[layers.length][][];
    double[][] bias = new double[layers.length][];
    ActivationFunction[] activationFunctions = new ActivationFunction[layers.length];
    Error type = layers[layers.length - 1].getErrorType();
    for (int i = 0; i < layers.length; i++) {
      double[][] layerWeights = new double[layers[i].length()][];
      for (int j = 0; j < layerWeights.length; j++) {
        layerWeights[j] = layers[i].get(j);
      }
      activationFunctions[i] = layers[i].getActivationFunction();
      weights[i] = layerWeights;
      bias[i] = layers[i].getBias();
    }

    Object[] objects = {weights, bias, activationFunctions, type, regularizations};

    try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(name)))) {
      oos.writeObject(objects);
    } catch (IOException ex) {
      System.err.printf("Cannot export to %s.\n", name);
    }
  }

  public double[] test(double[] input) {
    double[] out = layers[0].test(input);
    for (int index = 1; index < layers.length; index++) {
      out = layers[index].test(out);
    }
    return out;
  }

  public void exportToExternal(String name) {
    name += ".txt";
    System.out.printf("Exporting to %s. You can read this file using other programming languages.\n", name);

    double[][][] weights = new double[layers.length][][];
    double[][] bias = new double[layers.length][];
    for (int i = 0; i < layers.length; i++) {
      double[][] layerWeights = new double[layers[i].length()][];
      for (int j = 0; j < layerWeights.length; j++) {
        layerWeights[j] = layers[i].get(j);
      }
      weights[i] = layerWeights;
      bias[i] = layers[i].getBias();
    }

    try (FileWriter fileWriter = new FileWriter(name)) {
      for (int i = 0; i < layers.length; i++) {
        if (i != 0) {
          fileWriter.write("!");
        }
        for (int j = 0; j < weights[i].length; j++) {
          if (j != 0) {
            fileWriter.write("\n");
          }
          for (int k = 0; k < weights[i][j].length; k++) {
            if (k != 0) {
              fileWriter.write(",");
            }
            fileWriter.write(String.valueOf(weights[i][j][k]));
          }
        }
      }

      fileWriter.write("\nbias:");

      for (int i = 0; i < layers.length; i++) {
        if (i != 0) {
          fileWriter.write("\n");
        }
        for (int j = 0; j < bias[i].length; j++) {
          if (j != 0) {
            fileWriter.write(",");
          }

          fileWriter.write(String.valueOf(bias[i][j]));
        }
      }
    } catch (IOException e) {
      System.err.println("An error occurred during the export.");
    }
  }

  @Override
  public String toString() {
    int trainableParams = 0;
    for (Layer layer : layers) {
      trainableParams += (layer.get(0).length + 1) * (layer.length());
    }
    StringBuilder out = new StringBuilder();
    out.append("Network with ").append(layers.length).append(" layers and ").append(trainableParams).append(" trainable parameters.\n");
    System.out.println("Regularizers: " + regularizations);
    out.append("Size: {");
    out.append(layers[0].get(0).length).append(", ");
    for (int i = 0; i < layers.length; i++) {
      out.append(layers[i].length());
      if (i != layers.length - 1) {
        out.append(", ");
      } else {
        out.append("}");
      }
    }
    for (int i = 0; i < layers.length; i++) {
      out.append("\n\tLayer ").append(i + 1).append(":\n");
      out.append("\t\tSize: ").append(layers[i].get(0).length).append("-").append(layers[i].length());
      out.append("\n\t\tActivation Function: ").append(layers[i].getActivationFunction());
      out.append("\n\t\tError: ").append(layers[i].getErrorType());
    }
    return out.toString();
  }
}
