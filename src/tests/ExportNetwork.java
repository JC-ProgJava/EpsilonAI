package tests;

import epsilon.ActivationFunction;
import epsilon.Error;
import epsilon.dataset.MNIST;
import epsilon.fast.Layer;
import epsilon.fast.Network;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ExportNetwork {
  public static void main(String[] args) throws IOException {
    Network network = new Network("mynetwork.epsilon");
    network.exportToExternal("network");
    String networkString = new String(Files.readAllBytes(Path.of("network.txt")));
    String[] layerAndBiasString = networkString.split("bias:");
    String[] layers = layerAndBiasString[0].split("!");
    double[][][] weights = new double[2][][];
    for (int i = 0; i < layers.length; i++) {
      String[] toNeurons = layers[i].split("\n");
      weights[i] = new double[toNeurons.length][];
      for (int j = 0; j < toNeurons.length; j++) {
        String[] weightArray = toNeurons[j].split(",");
        weights[i][j] = new double[weightArray.length];
        for (int k = 0; k < weightArray.length; k++) {
          weights[i][j][k] = Double.parseDouble(weightArray[k]);
        }
      }
    }
    String[] bias = layerAndBiasString[1].split("\n");
    double[][] biases = new double[2][];
    for (int i = 0; i < bias.length; i++) {
      String[] toNeurons = bias[i].split("\n");
      biases[i] = new double[toNeurons.length];
      for (int j = 0; j < toNeurons.length; j++) {
        String[] weightArray = toNeurons[j].split(",");
        biases[i] = new double[weightArray.length];
        for (String s : weightArray) {
          biases[i][j] = Double.parseDouble(s);
        }
      }
    }

    Layer[] layersNetwork = new Layer[weights.length];
    for (int index = 0; index < weights.length; index++) {
      Layer layer;
      if (index == 0) {
        layer = new Layer(weights[index], biases[index], false, ActivationFunction.LEAKY_RELU, Error.CROSS_ENTROPY);
      } else {
        layer = new Layer(weights[index], biases[index], false, ActivationFunction.SOFTMAX, Error.CROSS_ENTROPY);
      }
      layersNetwork[index] = layer;
    }
    Network importedNetwork = new Network(layersNetwork);
    MNIST test = new MNIST().subset(60000, 70000);
    {
      int corrects = 0;
      for (int i = 0; i < test.inputs().length; i++) {
        int x = TestFast.getMax(importedNetwork.test(test.inputs()[i]));
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
  }
}
