package epsilon.dataset;

public final class QMNIST extends Dataset {
  public QMNIST() {
    super("QMNIST");
  }

  @Override
  public QMNIST subset(int startIndex, int endIndexExcluded) {
    return (QMNIST) super.subset(startIndex, endIndexExcluded);
  }
}
