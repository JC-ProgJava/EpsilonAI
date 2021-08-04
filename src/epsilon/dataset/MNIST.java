package epsilon.dataset;

public class MNIST extends Dataset {
  public MNIST() {
    super("MNIST");
  }

  @Override
  public MNIST subset(int startIndex, int endIndexExcluded) {
    return (MNIST) super.subset(startIndex, endIndexExcluded);
  }
}
