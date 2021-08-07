package epsilon.dataset;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

class Dataset {
  protected static final int BUFFER_SIZE = 4096;
  protected double[][] inputs;
  protected double[][] target;
  protected boolean isVerbose = true;

  public Dataset(String name) {
    try {
      if (isVerbose) System.out.println("Creating temporary file to store unzipped dataset...");
      Path path = Files.createTempFile(null, null);

      if (isVerbose) System.out.println("Gather resources...");
      InputStream stream = MNIST.class.getResourceAsStream(name + ".zip");
      OutputStream outStream = new FileOutputStream(path.toFile());


      if (isVerbose) System.out.println("Reading dataset...");
      byte[] buffer = new byte[2048];
      int bytesRead;
      while ((bytesRead = stream.read(buffer)) != -1) {
        outStream.write(buffer, 0, bytesRead);
      }

      if (isVerbose) System.out.println("Extracting dataset...");
      extract(new ZipInputStream(new FileInputStream(path.toString())), new File(path.getParent().toString()));

      if (isVerbose) System.out.println("Initializing data array...");
      ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path.getParent().toString() + File.separator + name + File.separator + "input.ser")));
      ObjectInputStream oiss = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path.getParent().toString() + File.separator + name + File.separator + "target.ser")));
      inputs = (double[][]) ois.readObject();
      target = (double[][]) oiss.readObject();
      ois.close();
      oiss.close();
    } catch (IOException | ClassNotFoundException | NullPointerException ex) {
      ex.printStackTrace();
    }
    if (isVerbose) System.out.println("Finished creating " + this.getClass().toString() + " dataset!\n");
  }

  protected void extract(ZipInputStream zip, File target) throws IOException {
    try (zip) {
      ZipEntry entry;

      while ((entry = zip.getNextEntry()) != null) {
        File file = new File(target, entry.getName());

        if (!file.toPath().normalize().startsWith(target.toPath())) {
          throw new IOException("Bad zip entry");
        }

        if (entry.isDirectory()) {
          file.mkdirs();
          continue;
        }

        byte[] buffer = new byte[BUFFER_SIZE];
        file.getParentFile().mkdirs();
        BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(file));
        int count;

        while ((count = zip.read(buffer)) != -1) {
          out.write(buffer, 0, count);
        }

        out.close();
      }
    }
  }

  public void setVerbose(boolean isVerbose) {
    this.isVerbose = isVerbose;
  }

  public Dataset subset(int startIndex, int endIndexExcluded) {
    if (startIndex < 0 || endIndexExcluded < 1) {
      throw new IllegalArgumentException("subset(startIndex, endIndexExcluded): Invalid parameters.");
    }
    inputs = Arrays.copyOfRange(inputs, startIndex, endIndexExcluded);
    target = Arrays.copyOfRange(target, startIndex, endIndexExcluded);
    return this;
  }

  public double[][] inputs() {
    return inputs;
  }

  public double[][] target() {
    return target;
  }
}
