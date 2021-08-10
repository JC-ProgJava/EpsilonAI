package epsilon.fast;

import epsilon.ActivationFunction;
import epsilon.Error;
import epsilon.Optimizer;
import epsilon.Regularization;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public final class Layer implements Serializable {
  private static final Random rand = new Random();
  private final boolean isOutputLayer;
  private final ActivationFunction activationFunction;
  private final Error errorType;
  private ArrayList<Regularization> regularization = new ArrayList<>();
  private double[][] vectors;
  private double[] bias;
  private int BATCH_SIZE;
  private int currentInputBatchId = 0;
  private Layer nextLayer;
  transient private double[] targetOutput;
  transient private double[][] deltas;
  transient private double[][] prevUpdate;
  transient private double[][] adaptDelta;
  transient private double[] deltaBias;
  transient private Optimizer optimizer;
  transient private double[] input;
  transient private double[] output;
  transient private double[] error;
  transient private double[] displayError;

  Layer(boolean isOutputLayer, ActivationFunction activationFunction, Error errorType) {
    this.isOutputLayer = isOutputLayer;
    this.activationFunction = activationFunction;
    this.errorType = errorType;
  }

  public Layer(double[][] vectorList, double[] bias, boolean isOutputLayer, ActivationFunction activationFunction, Error errorType) {
    vectors = vectorList;
    this.isOutputLayer = isOutputLayer;
    this.activationFunction = activationFunction;
    this.errorType = errorType;
    this.bias = bias;
    initDeltaArray();
  }

  static double total(double[] val) {
    double sum = 0;
    for (double v : val) {
      sum += v;
    }
    return sum;
  }

  static double[] add(double[] vec, double[] vec2) {
    if (vec2.length != vec.length) {
      throw new IllegalArgumentException("add(Vector): Vectors must be of same length to be added together. Sizes given: {" + vec.length + ", " + vec2.length + "}.");
    }

    int size = vec.length;
    double[] sum = new double[size];
    for (int i = 0; i < size; i++) {
      sum[i] = vec[i] + vec2[i];
    }

    return sum;
  }

  static double[] add(double[] vec, double value) {
    int size = vec.length;
    double[] sum = new double[size];
    for (int i = 0; i < size; i++) {
      sum[i] = vec[i] + value;
    }

    return sum;
  }

  static double[] subtract(double[] vec, double[] vec2) {
    if (vec2.length != vec.length) {
      throw new IllegalArgumentException("subtract(Vector): Vectors must be of same length to be subtracted. Sizes given: {" + vec.length + ", " + vec2.length + "}.");
    }

    int size = vec.length;
    double[] diff = new double[vec.length];
    for (int i = 0; i < size; i++) {
      diff[i] = vec[i] - vec2[i];
    }

    return diff;
  }

  static double[] subtract(double[] vec, double value) {
    int size = vec.length;
    double[] diff = new double[size];
    for (int i = 0; i < size; i++) {
      diff[i] = vec[i] - value;
    }

    return diff;
  }

  static double[] mult(double[] vec, double[] vec2) {
    if (vec2.length != vec.length) {
      throw new IllegalArgumentException("mult(Vector): Vectors must be of same length to be multiplied together. Sizes given: {" + vec.length + ", " + vec2.length + "}.");
    }

    int size = vec.length;
    double[] prod = new double[vec.length];
    for (int i = 0; i < size; i++) {
      prod[i] = vec[i] * vec2[i];
    }

    return prod;
  }

  static double[] mult(double[] vec, double value) {
    int size = vec.length;
    double[] prod = new double[size];
    for (int i = 0; i < size; i++) {
      prod[i] = vec[i] * value;
    }

    return prod;
  }

  static double[] fill(double[] vec, double value) {
    Arrays.fill(vec, value);
    return vec;
  }

  static double[] div(double[] vec, double[] vec2) {
    if (vec2.length != vec.length) {
      throw new IllegalArgumentException("div(Vector): Vectors must be of same length to be divided. Sizes given: {" + vec.length + ", " + vec2.length + "}.");
    }

    int size = vec.length;
    double[] quo = new double[size];
    for (int i = 0; i < size; i++) {
      quo[i] = vec[i] / vec2[i];
    }

    return quo;
  }

  static double[] div(double[] vec, double value) {
    int size = vec.length;
    double[] quo = new double[size];
    for (int i = 0; i < size; i++) {
      quo[i] = vec[i] / value;
    }

    return quo;
  }

  static double[] sqrt(double[] vec) {
    int size = vec.length;
    double[] sqrt = new double[size];
    for (int i = 0; i < size; i++) {
      sqrt[i] = Math.sqrt(vec[i]);
    }
    return sqrt;
  }

  static double[] log(double[] vec) {
    int size = vec.length;
    double[] log = new double[size];
    for (int i = 0; i < size; i++) {
      log[i] = Math.log(vec[i]);
    }
    return log;
  }

  static double[] fillGaussian(double[] vec) {
    for (int i = 0; i < vec.length; i++) {
      vec[i] = rand.nextGaussian();
    }
    return vec;
  }

  static double[] fillGaussian(double[] vec, double average, double deviation) {
    for (int i = 0; i < vec.length; i++) {
      vec[i] = rand.nextGaussian() * deviation + average;
    }
    return vec;
  }

  Layer setRegularization(ArrayList<Regularization> regularization) {
    this.regularization = regularization;
    return this;
  }

  private void initDeltaArray() {
    deltas = new double[vectors.length][];
    for (int i = 0; i < vectors.length; i++) {
      deltas[i] = new double[vectors[i].length];
    }

    deltaBias = new double[vectors.length];
  }

  void setBatchSize(int BATCH_SIZE) {
    this.BATCH_SIZE = BATCH_SIZE;
  }

  double[][] vectors() {
    return vectors;
  }

  void clearCache() {
    prevUpdate = null;
    adaptDelta = null;
  }

  void set(double[][] vectors) {
    this.vectors = vectors;
  }

  void set(int index, double[] item) {
    vectors[index] = item;
  }

  int length() {
    return vectors.length;
  }

  double[] get(int index) {
    if (index < 0) {
      throw new IllegalArgumentException("get(index): index must be non-negative, but got '" + index + "'.");
    } else if (index >= length()) {
      throw new IllegalArgumentException("get(index): index out of bounds. Value: '" + index + "'");
    }

    return vectors[index];
  }

  double[] getBias() {
    return bias;
  }

  double[] getError() {
    return error;
  }

  double[] getDisplayError() {
    return displayError;
  }

  double[] getOutput() {
    return output;
  }

  double[] test(double[] input) {
    if (input.length != vectors()[0].length) {
      throw new IllegalArgumentException("test(Vector): Input vector not the same size as weight vectors.");
    }

    double[] output = new double[vectors().length];
    for (int i = 0; i < vectors().length; i++) {
      output[i] = total(mult(vectors()[i], input));
    }

    output = add(output, bias);
    output = activation(output);

    return output;
  }

  void feed(double[] input) {
    this.input = input;

    if (input.length != vectors()[0].length) {
      throw new IllegalArgumentException("feed(Vector): Input vector not the same size as weight vectors.");
    }

    double[] output = new double[vectors().length];
    for (int i = 0; i < vectors().length; i++) {
      output[i] = total(mult(vectors()[i], input));
    }

    output = add(output, bias);
    output = activation(output);

    this.output = output;
  }

  private double[] activation(double[] val) {
    int valSize = val.length;
    double[] out = new double[valSize];
    for (int index = 0; index < valSize; index++) {
      switch (activationFunction) {
        case IDENTITY:
          return val;
        case TANH:
          out[index] = tanh(val[index]);
          break;
        case RELU:
          out[index] = relu(val[index]);
          break;
        case LEAKY_RELU:
          out[index] = leaky_relu(val[index]);
          break;
        case SIGMOID:
          out[index] = sigmoid(val[index]);
          break;
        case SOFTPLUS:
          out[index] = softplus(val[index]);
          break;
        case SOFTMAX:
          return softmax(val);
        case ELU:
          out[index] = elu(val[index]);
          break;
        default:
          throw new IllegalArgumentException("activation(): No such ActivationFunction '" + activationFunction + "'.");
      }
    }

    return out;
  }

  private double elu(double val) {
    return val > 0 ? val : Math.exp(val) - 1.0;
  }

  private double sigmoid(double val) {
    return 1.0 / (1.0 + Math.exp(-val));
  }

  private double relu(double val) {
    return val <= 0 ? 0 : val;
  }

  private double leaky_relu(double val) {
    return val <= 0 ? 0.01 * val : val;
  }

  private double softplus(double val) {
    return Math.log(1.0 + Math.exp(val));
  }

  private double[] softmax(double[] val) {
    int valLength = val.length;
    double[] out = new double[valLength];
    double total = 0.0;

    for (double v : val) {
      total += Math.exp(v);
    }

    if (total == 0.0) {
      throw new ArithmeticException("softmax(Vector val): total is 0, cannot divide by 0.");
    }
    for (int i = 0; i < valLength; i++) {
      out[i] = (Math.exp(val[i]) / total);
    }

    return out;
  }

  private double tanh(double val) {
    return (Math.exp(val) - Math.exp(-val)) / (Math.exp(val) + Math.exp(-val));
  }

  double[] derivative(int index) {
    int outLength = output.length;
    double[] returnVector = new double[outLength];
    switch (activationFunction) {
      case IDENTITY:
        return new double[output.length];
      case TANH:
        for (int i = 0; i < outLength; i++) {
          returnVector[i] = 1.0 - sigmoid(output[i]) * sigmoid(output[i]);
        }
        return returnVector;
      case RELU: {
        for (double v : output) {
          returnVector[index] = v < 0.0 ? 0.0 : 1.0;
        }
        return returnVector;
      }
      case LEAKY_RELU: {
        for (int i = 0; i < outLength; i++) {
          returnVector[i] = (output[i] < 0.0 ? 0.01 : 1.0);
        }
        return returnVector;
      }
      case SIGMOID:
        return add(mult(mult(activation(output), activation(output)), -1.0), 1.0);
      case SOFTPLUS: {
        for (double v : output) {
          returnVector[index] = sigmoid(v);
        }
        return returnVector;
      }
      case SOFTMAX: {
        for (int i = 0; i < outLength; i++) {
          if (i == index) {
            returnVector[i] = softmax(output)[index] * (1.0 - softmax(output)[index]);
          } else {
            returnVector[i] = -1.0 * softmax(output)[index] * softmax(output)[index];
          }
        }

        return returnVector;
      }
      case ELU: {
        for (int i = 0; i < outLength; i++) {
          if (output[i] >= 0) {
            returnVector[i] = 1.0;
          } else {
            returnVector[i] = Math.exp(output[i]);
          }
        }

        return returnVector;
      }
      default:
        throw new IllegalArgumentException(activationFunction + " is not a valid ActivationFunction type.");
    }
  }

  void learn(double[] targetOutput, Layer nextLayer, double alpha, Optimizer optimizer) {
    if (alpha == 0) {
      throw new NullPointerException("learn(): learning rate can't be 0 (no learning).");
    }

    this.optimizer = optimizer;
    this.targetOutput = targetOutput;
    this.nextLayer = nextLayer;
    initOptimizer(vectors()[0].length, vectors().length);

    if (isOutputLayer) {
      if (displayError == null) {
        displayError = new double[vectors().length];
      } else {
        displayError = mult(displayError, 0.0);
      }
    }

    IntStream stream = IntStream.range(0, vectors.length);

    stream.parallel().forEach(index -> {
      double[] error;
      if (isOutputLayer) {
        double[][] errorCalc = calcError(output, targetOutput, index);
        error = errorCalc[0];
        displayError = add(displayError, errorCalc[1]);
      } else {
        error = new double[vectors().length];
        for (int indice = 0; indice < nextLayer.vectors().length; indice++) {
          error = add(error, nextLayer.vectors()[indice][index] * nextLayer.getError()[indice]);
        }
        error = mult(error, derivative(index));
      }
      this.error = error;
      double[] delta = mult(input, error[index]);
      delta = applyOptimizer(delta, index, alpha);
      deltas[index] = add(deltas[index], delta);
      deltaBias[index] += error[index] * alpha;
    });

    currentInputBatchId++;

    if (currentInputBatchId == BATCH_SIZE) {
      currentInputBatchId = 0;
      applyRegularizer();
      for (int index = 0; index < vectors().length; index++) {
        vectors[index] = subtract(vectors[index], deltas[index]);
        bias[index] -= deltaBias[index];
        deltas[index] = mult(deltas[index], 0.0);
        deltaBias[index] = 0.0;
      }
    }
  }

  Error getErrorType() {
    return errorType;
  }

  private double[][] calcError(double[] output, double[] targetOutput, int index) {
    switch (errorType) {
      case MEAN_SQUARED: {
        double[] err = subtract(output, targetOutput);
        return new double[][]{mult(err, derivative(index)), mult(err, err)};
      }
      case MEAN_ABSOLUTE: {
        double[] err = subtract(output, targetOutput);
        for (int i = 0; i < err.length; i++) {
          err[i] = err[i] >= 0 ? err[i] : -err[i];
        }
        return new double[][]{mult(err, derivative(index)), err};
      }
      case CROSS_ENTROPY: {
        double[] display = mult(mult(targetOutput, log(output)), -1.0);
        return new double[][]{subtract(output, targetOutput), display};
      }
      default:
        throw new IllegalArgumentException("calcError(): Error type " + errorType + " unknown.");
    }
  }

  double[] applyOptimizer(double[] delta, int index, double alpha) {
    switch (optimizer) {
      case MOMENTUM:
        delta = add(mult(delta, alpha), mult(prevUpdate[index], 0.9));
        prevUpdate[index] = delta;
        break;
      case ADAGRAD: {
        double epsilon = 1e-4;
        double[] finalAlpha = new double[vectors()[index].length];
        for (int i = 0; i < vectors()[index].length; i++) {
          finalAlpha[i] = alpha / Math.sqrt(prevUpdate[index][i] + epsilon);
          prevUpdate[index][i] = prevUpdate[index][i] + delta[i] * delta[i];
        }
        delta = mult(delta, finalAlpha);
        break;
      }
      case RMSPROP: {
        double epsilon = 1e-4;
        double beta = 0.9;
        double[] finalAlpha = new double[vectors()[index].length];
        for (int i = 0; i < vectors()[index].length; i++) {
          finalAlpha[i] = alpha / Math.sqrt(prevUpdate[index][i] + epsilon);
          prevUpdate[index][i] = prevUpdate[index][i] * beta + (1.0 - beta) * delta[i] * delta[i];
        }
        delta = mult(delta, finalAlpha);
        break;
      }
      case ADADELTA: {
        double epsilon = 1e-6;
        double beta = 0.95;
        double[] finalAlpha = new double[vectors()[index].length];
        for (int i = 0; i < vectors()[index].length; i++) {
          finalAlpha[i] = Math.sqrt(adaptDelta[index][i] + epsilon) / Math.sqrt(prevUpdate[index][i] + epsilon);
          prevUpdate[index][i] = prevUpdate[index][i] * beta + (1.0 - beta) * delta[i] * delta[i];
          double adaptD = delta[i] * -finalAlpha[i];
          adaptDelta[index][i] = adaptDelta[index][i] * beta + (1.0 - beta) * adaptD * adaptD;
        }
        delta = mult(delta, finalAlpha);
        break;
      }
      case ADAM: {
        double epsilon = 1e-7;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double[] finalAlpha = new double[vectors()[index].length];
        for (int i = 0; i < vectors()[index].length; i++) {
          prevUpdate[index][i] = prevUpdate[index][i] * beta1 + (1.0 - beta1) * delta[i];
          adaptDelta[index][i] = adaptDelta[index][i] * beta2 + (1.0 - beta2) * delta[i] * delta[i];
          finalAlpha[i] = alpha / (Math.sqrt((adaptDelta[index][i]) / (1.0 - beta2)) + epsilon);
        }
        delta = mult(div(prevUpdate[index], 1.0 - beta1), finalAlpha);
        break;
      }
      case NESTEROV: {
        prevUpdate[index] = add(mult(prevUpdate[index], 0.9), mult(nesterovGradient(subtract(vectors[index], mult(prevUpdate[index], 0.9)), index), alpha));
        delta = prevUpdate[index];
        break;
      }
      default: {
        delta = mult(delta, alpha);
        break;
      }
    }
    return delta;
  }

  void applyRegularizer() {
    for (double[] vector : vectors) {
      if (regularization.contains(Regularization.L1)) {
        for (int i = 0; i < vector.length; i++) {
          if (vector[i] >= 0) {
            vector[i] += 0.001;
          } else {
            vector[i] -= 0.001;
          }
        }
      } else if (regularization.contains(Regularization.L2)) {
        for (int i = 0; i < vector.length; i++) {
          vector[i] -= 0.001 * vector[i];
        }
      }
    }
  }

  private double[] nesterovGradient(double[] newWeights, int index) {
    double[] output = new double[vectors().length];
    for (int i = 0; i < vectors().length; i++) {
      if (index != i) {
        output[i] = total(mult(vectors()[i], input));
      } else {
        output[i] = total(mult(newWeights, input));
      }
    }

    output = add(output, bias);
    output = activation(output);

    double[] error;
    if (isOutputLayer) {
      double[][] errorCalc = calcError(output, targetOutput, index);
      error = errorCalc[0];
    } else {
      error = new double[vectors().length];
      for (int indice = 0; indice < nextLayer.vectors().length; indice++) {
        error = add(error, nextLayer.vectors()[indice][index] * nextLayer.getError()[indice]);
      }
      error = mult(error, derivative(index));
    }
    return mult(input, error[index]);
  }

  void initOptimizer(int size, int numVectors) {
    if (prevUpdate == null) {
      prevUpdate = new double[numVectors][];
      if (optimizer == Optimizer.MOMENTUM || optimizer == Optimizer.ADAGRAD || optimizer == Optimizer.RMSPROP || optimizer == Optimizer.NESTEROV) {
        for (int i = 0; i < numVectors; i++) {
          prevUpdate[i] = new double[size];
        }
      } else if (optimizer == Optimizer.ADADELTA || optimizer == Optimizer.ADAM) {
        adaptDelta = new double[numVectors][];
        for (int i = 0; i < numVectors; i++) {
          prevUpdate[i] = new double[size];
          adaptDelta[i] = new double[size];
        }
      }
    }
  }

  ActivationFunction getActivationFunction() {
    return activationFunction;
  }

  Layer fillGaussian(int size, int numVectors) {
    this.set(new double[numVectors][]);
    for (int i = 0; i < numVectors; i++) {
      this.set(i, fillGaussian(new double[size]));
    }

    bias = fillGaussian(new double[numVectors]);
    initDeltaArray();

    return this;
  }

  void fillGaussian(double average, double deviation) {
    double[][] replaceVect = new double[vectors.length][];

    for (int i = 0; i < replaceVect.length; i++) {
      replaceVect[i] = fillGaussian(new double[this.vectors[i].length], average, deviation);
    }

    bias = fillGaussian(new double[vectors.length], average, deviation);

    set(replaceVect);
    initDeltaArray();
  }

  void fillZeros(int size, int numVectors) {
    this.set(new double[numVectors][]);
    for (int i = 0; i < numVectors; i++) {
      this.set(i, new double[size]);
    }

    bias = new double[vectors.length];
    initDeltaArray();
  }

  void fillRandom(int size, int numVectors) {
    this.set(new double[numVectors][]);
    for (int i = 0; i < numVectors; i++) {
      this.set(i, fillRandom(new double[size]));
    }

    bias = fillRandom(new double[numVectors]);
    initDeltaArray();
  }

  double[] fillRandom(double[] vec) {
    for (int i = 0; i < vec.length; i++) {
      vec[i] = rand.nextDouble();
    }
    return vec;
  }
}
