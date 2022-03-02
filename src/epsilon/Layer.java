package epsilon;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.stream.IntStream;

final class Layer implements Serializable {
  private final boolean isOutputLayer;
  private final ActivationFunction activationFunction;
  private final Error errorType;
  private ArrayList<Regularization> regularization = new ArrayList<>();
  private Vector[] vectors;
  private int BATCH_SIZE;
  private int currentInputBatchId = 0;
  private Layer nextLayer;
  private Vector bias;
  transient private Vector targetOutput;
  transient private Vector[] deltas;
  transient private Vector[] prevUpdate;
  transient private Vector[] adaptDelta;
  transient private Vector deltaBias;
  transient private Optimizer optimizer;
  transient private Vector input;
  transient private Vector output;
  transient private Vector error;
  transient private Vector displayError;

  Layer(boolean isOutputLayer, ActivationFunction activationFunction, Error errorType) {
    this.isOutputLayer = isOutputLayer;
    this.activationFunction = activationFunction;
    this.errorType = errorType;
  }

  Layer(Vector[] vectorList, Vector bias, boolean isOutputLayer, ActivationFunction activationFunction, Error errorType) {
    vectors = vectorList;
    this.isOutputLayer = isOutputLayer;
    this.activationFunction = activationFunction;
    this.errorType = errorType;
    this.bias = bias;
    initDeltaArray();
  }

  Layer setRegularization(ArrayList<Regularization> regularization) {
    this.regularization = regularization;
    return this;
  }

  private void initDeltaArray() {
    deltas = new Vector[vectors.length];
    for (int i = 0; i < vectors.length; i++) {
      deltas[i] = new Vector(vectors[i].length()).fillZeros();
    }

    deltaBias = new Vector(vectors.length).fillZeros();
  }

  void setBatchSize(int BATCH_SIZE) {
    this.BATCH_SIZE = BATCH_SIZE;
  }

  Vector[] vectors() {
    return vectors;
  }

  void clearCache() {
    prevUpdate = null;
    adaptDelta = null;
  }

  void set(Vector[] vectors) {
    this.vectors = vectors;
  }

  void set(int index, Vector item) {
    vectors[index] = item;
  }

  int length() {
    return vectors.length;
  }

  Vector get(int index) {
    if (index < 0) {
      throw new IllegalArgumentException("get(index): index must be non-negative, but got '" + index + "'.");
    } else if (index >= length()) {
      throw new IllegalArgumentException("get(index): index out of bounds. Value: '" + index + "'");
    }

    return vectors[index];
  }

  Vector getBias() {
    return bias;
  }

  Vector getError() {
    return error;
  }

  Vector getDisplayError() {
    return displayError;
  }

  Vector getOutput() {
    return output;
  }

  Vector test(Vector input) {
    if (input.length() != vectors()[0].length()) {
      throw new IllegalArgumentException("test(Vector): Input vector not the same size as weight vectors.");
    }

    Vector output = new Vector(vectors().length);
    for (int i = 0; i < vectors().length; i++) {
      output.set(i, vectors()[i].mult(input).total());
    }

    output = output.add(bias);
    output = activation(output);

    return output;
  }

  void feed(Vector input) {
    this.input = input;

    if (input.length() != vectors()[0].length()) {
      throw new IllegalArgumentException("feed(Vector): Input vector not the same size as weight vectors.");
    }

    Vector output = new Vector(vectors().length);
    for (int i = 0; i < vectors().length; i++) {
      output.set(i, vectors()[i].mult(input).total());
    }

    output = output.add(bias);
    output = activation(output);

    this.output = output;
  }

  private Vector activation(Vector val) {
    int valSize = val.length();
    Vector out = new Vector(valSize);
    for (int index = 0; index < valSize; index++) {
      switch (activationFunction) {
        case IDENTITY:
          return val;
        case TANH:
          out.set(index, tanh(val.get(index)));
          break;
        case RELU:
          out.set(index, relu(val.get(index)));
          break;
        case LEAKY_RELU:
          out.set(index, leaky_relu(val.get(index)));
          break;
        case SIGMOID:
          out.set(index, sigmoid(val.get(index)));
          break;
        case SOFTPLUS:
          out.set(index, softplus(val.get(index)));
          break;
        case SOFTMAX:
          return softmax(val);
        case ELU:
          out.set(index, elu(val.get(index)));
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

  private Vector softmax(Vector val) {
    int valLength = val.length();
    Vector out = new Vector(valLength);
    double total = 0.0;

    for (int i = 0; i < valLength; i++) {
      total += Math.exp(val.get(i));
    }

    if (total == 0.0) {
      throw new ArithmeticException("softmax(Vector val): total is 0, cannot divide by 0.");
    }
    for (int i = 0; i < valLength; i++) {
      out.set(i, (Math.exp(val.get(i)) / total));
    }

    return out;
  }

  private double tanh(double val) {
    return (Math.exp(val) - Math.exp(-val)) / (Math.exp(val) + Math.exp(-val));
  }

  Vector derivative(int index) {
    int outLength = output.length();
    Vector returnVector = new Vector(outLength);
    switch (activationFunction) {
      case IDENTITY:
        return new Vector(output.length()).fill(1.0);
      case TANH:
        for (int i = 0; i < outLength; i++) {
          returnVector.set(i, 1.0 - sigmoid(output.get(i)) * sigmoid(output.get(i)));
        }
        return returnVector;
      case RELU: {
        for (int i = 0; i < outLength; i++) {
          returnVector.set(i, output.get(i) < 0.0 ? 0.0 : 1.0);
        }
        return returnVector;
      }
      case LEAKY_RELU: {
        for (int i = 0; i < outLength; i++) {
          returnVector.set(i, output.get(i) < 0.0 ? 0.01 : 1.0);
        }
        return returnVector;
      }
      case SIGMOID:
        return output.mult(output.mult(-1.0).add(1.0));
      case SOFTPLUS: {
        for (int i = 0; i < outLength; i++) {
          returnVector.set(i, sigmoid(output.get(i)));
        }
        return returnVector;
      }
      case SOFTMAX: {
        for (int i = 0; i < outLength; i++) {
          if (i == index) {
            returnVector.set(i, softmax(output).get(i) * (1.0 - softmax(output).get(i)));
          } else {
            returnVector.set(i, -1.0 * softmax(output).get(i) * softmax(output).get(index));
          }
        }

        return returnVector;
      }
      case ELU: {
        for (int i = 0; i < outLength; i++) {
          if (output.get(i) >= 0) {
            returnVector.set(i, 1.0);
          } else {
            returnVector.set(i, Math.exp(output.get(i)));
          }
        }

        return returnVector;
      }
      default:
        throw new IllegalArgumentException(activationFunction + " is not a valid ActivationFunction type.");
    }
  }

  void learn(Vector targetOutput, Layer nextLayer, double alpha, Optimizer optimizer) {
    if (alpha == 0) {
      throw new NullPointerException("learn(): learning rate can't be 0 (no learning).");
    }

    this.optimizer = optimizer;
    this.targetOutput = targetOutput;
    this.nextLayer = nextLayer;
    initOptimizer(vectors()[0].length(), vectors().length);

    if (isOutputLayer) {
      if (displayError == null) {
        displayError = new Vector(vectors().length).fillZeros();
      } else {
        displayError = displayError.mult(0.0);
      }
    }

    IntStream stream = IntStream.range(0, vectors.length);

    stream.parallel().forEach(index -> {
      Vector error;
      if (isOutputLayer) {
        Vector[] errorCalc = calcError(output, targetOutput, index);
        error = errorCalc[0];
        displayError = displayError.add(errorCalc[1]);
      } else {
        error = new Vector(vectors().length).fillZeros();
        for (int indice = 0; indice < nextLayer.vectors().length; indice++) {
          error = error.add(nextLayer.vectors()[indice].get(index) * nextLayer.getError().get(indice));
        }
        error = error.mult(derivative(index));
      }
      this.error = error;
      Vector delta = input.mult(error.get(index));
      delta = applyOptimizer(delta, index, alpha);
      deltas[index] = deltas[index].add(delta);
      deltaBias.set(index, deltaBias.get(index) + error.get(index) * alpha);
    });

    currentInputBatchId++;

    if (currentInputBatchId == BATCH_SIZE) {
      currentInputBatchId = 0;
      applyRegularizer();
      for (int index = 0; index < vectors().length; index++) {
        vectors[index] = vectors[index].subtract(deltas[index]);
        deltas[index] = deltas[index].mult(0.0);
      }
      bias = bias.subtract(deltaBias);
      deltaBias = deltaBias.mult(0.0);
    }
  }

  Error getErrorType() {
    return errorType;
  }

  private Vector[] calcError(Vector output, Vector targetOutput, int index) {
    switch (errorType) {
      case MEAN_SQUARED: {
        Vector err = (output.subtract(targetOutput));
        return new Vector[]{err.mult(derivative(index)), err.mult(err)};
      }
      case MEAN_ABSOLUTE: {
        Vector err = (output.subtract(targetOutput));
        for (int i = 0; i < err.length(); i++) {
          err.set(i, err.get(i) > 0 ? err.get(i) : -err.get(i));
        }
        return new Vector[]{err.mult(derivative(index)), err};
      }
      case CROSS_ENTROPY: {
        Vector display = targetOutput.mult(output.log()).mult(-1.0);
        return new Vector[]{output.subtract(targetOutput), display};
      }
      default:
        throw new IllegalArgumentException("calcError(): Error type " + errorType + " unknown.");
    }
  }


  Vector applyOptimizer(Vector delta, int index, double alpha) {
    switch (optimizer) {
      case MOMENTUM:
        delta = delta.mult(alpha).add(prevUpdate[index].mult(0.9));
        prevUpdate[index] = delta;
        break;
      case ADAGRAD: {
        double epsilon = 1e-4;
        Vector finalAlpha = new Vector(vectors()[index].length());
        for (int i = 0; i < vectors()[index].length(); i++) {
          finalAlpha.set(i, alpha / Math.sqrt(prevUpdate[index].get(i) + epsilon));
          prevUpdate[index].set(i, prevUpdate[index].get(i) + delta.get(i) * delta.get(i));
        }
        delta = delta.mult(finalAlpha);
        break;
      }
      case RMSPROP: {
        double epsilon = 1e-4;
        double beta = 0.9;
        Vector finalAlpha = new Vector(vectors()[index].length());
        for (int i = 0; i < vectors()[index].length(); i++) {
          finalAlpha.set(i, alpha / Math.sqrt(prevUpdate[index].get(i) + epsilon));
          prevUpdate[index].set(i, prevUpdate[index].get(i) * beta + (1.0 - beta) * delta.get(i) * delta.get(i));
        }
        delta = delta.mult(finalAlpha);
        break;
      }
      case ADADELTA: {
        double epsilon = 1e-6;
        double beta = 0.95;
        Vector finalAlpha = new Vector(vectors()[index].length());
        for (int i = 0; i < vectors()[index].length(); i++) {
          finalAlpha.set(i, Math.sqrt(adaptDelta[index].get(i) + epsilon) / Math.sqrt(prevUpdate[index].get(i) + epsilon));
          prevUpdate[index].set(i, prevUpdate[index].get(i) * beta + (1.0 - beta) * delta.get(i) * delta.get(i));
          double adaptD = delta.get(i) * -finalAlpha.get(i);
          adaptDelta[index].set(i, adaptDelta[index].get(i) * beta + (1.0 - beta) * adaptD * adaptD);
        }
        delta = delta.mult(finalAlpha);
        break;
      }
      case ADAM: {
        double epsilon = 1e-7;
        double beta1 = 0.9;
        double beta2 = 0.999;
        Vector finalAlpha = new Vector(vectors()[index].length());
        for (int i = 0; i < vectors()[index].length(); i++) {
          prevUpdate[index].set(i, prevUpdate[index].get(i) * beta1 + (1.0 - beta1) * delta.get(i));
          adaptDelta[index].set(i, adaptDelta[index].get(i) * beta2 + (1.0 - beta2) * delta.get(i) * delta.get(i));
          finalAlpha.set(i, alpha / (Math.sqrt((adaptDelta[index].get(i)) / (1.0 - beta2)) + epsilon));
        }
        delta = prevUpdate[index].div(1.0 - beta1).mult(finalAlpha);
        break;
      }
      case NESTEROV: {
        prevUpdate[index] = (prevUpdate[index].mult(0.9)).add(nesterovGradient(vectors[index].subtract(prevUpdate[index].mult(0.9)), index).mult(alpha));
        delta = prevUpdate[index];
        break;
      }
      default: {
        delta = delta.mult(alpha);
        break;
      }
    }
    return delta;
  }

  void applyRegularizer() {
    for (Vector vector : vectors) {
      if (regularization.contains(Regularization.L1)) {
        for (int i = 0; i < vector.length(); i++) {
          if (vector.get(i) >= 0) {
            vector.set(i, vector.get(i) + 0.001);
          } else {
            vector.set(i, vector.get(i) - 0.001);
          }
        }
      } else if (regularization.contains(Regularization.L2)) {
        for (int i = 0; i < vector.length(); i++) {
          vector.set(i, vector.get(i) - 0.001 * vector.get(i));
        }
      }
    }
  }

  private Vector nesterovGradient(Vector newWeights, int index) {
    Vector output = new Vector(vectors().length);
    for (int i = 0; i < vectors().length; i++) {
      if (index != i) {
        output.set(i, vectors()[i].mult(input).total());
      } else {
        output.set(i, newWeights.mult(input).total());
      }
    }

    output = output.add(bias);
    output = activation(output);

    Vector error;
    if (isOutputLayer) {
      Vector[] errorCalc = calcError(output, targetOutput, index);
      error = errorCalc[0];
    } else {
      error = new Vector(vectors().length).fillZeros();
      for (int indice = 0; indice < nextLayer.vectors().length; indice++) {
        error = error.add(nextLayer.vectors()[indice].get(index) * nextLayer.getError().get(indice));
      }
      error = error.mult(derivative(index));
    }
    return input.mult(error.get(index));
  }

  void initOptimizer(int size, int numVectors) {
    if (prevUpdate == null) {
      prevUpdate = new Vector[numVectors];
      if (optimizer == Optimizer.MOMENTUM || optimizer == Optimizer.ADAGRAD || optimizer == Optimizer.RMSPROP || optimizer == Optimizer.NESTEROV) {
        for (int i = 0; i < numVectors; i++) {
          prevUpdate[i] = new Vector(size).fillZeros();
        }
      } else if (optimizer == Optimizer.ADADELTA || optimizer == Optimizer.ADAM) {
        adaptDelta = new Vector[numVectors];
        for (int i = 0; i < numVectors; i++) {
          prevUpdate[i] = new Vector(size).fillZeros();
          adaptDelta[i] = new Vector(size).fillZeros();
        }
      }
    }
  }

  ActivationFunction getActivationFunction() {
    return activationFunction;
  }

  Layer fillGaussian(int size, int numVectors) {
    set(new Vector[numVectors]);
    for (int i = 0; i < numVectors; i++) {
      set(i, new Vector(size).fillGaussian());
    }

    bias = new Vector(numVectors).fillGaussian();
    initDeltaArray();

    return this;
  }

  void fillGaussian(double average, double deviation) {
    Vector[] replaceVect = new Vector[vectors.length];

    for (int i = 0; i < replaceVect.length; i++) {
      replaceVect[i] = new Vector(this.vectors[i].length()).fillGaussian(average, deviation);
    }

    bias = new Vector(vectors.length).fillGaussian(average, deviation);

    set(replaceVect);
    initDeltaArray();
  }

  void fillZeros(int size, int numVectors) {
    set(new Vector[numVectors]);
    for (int i = 0; i < numVectors; i++) {
      set(i, new Vector(size).fillZeros());
    }

    bias = new Vector(numVectors).fillZeros();
    initDeltaArray();
  }

  void fillRandom(int size, int numVectors) {
    set(new Vector[numVectors]);
    for (int i = 0; i < numVectors; i++) {
      set(i, new Vector(size).fillRandom());
    }

    bias = new Vector(numVectors).fillRandom();
    initDeltaArray();
  }
}
