package epsilon.dataset;

public final class ExtendMNIST extends Dataset {
  public ExtendMNIST() {
    super("ExtendMNIST");
  }

  @Override
  public ExtendMNIST subset(int startIndex, int endIndexExcluded) {
    return (ExtendMNIST) super.subset(startIndex, endIndexExcluded);
  }
}
