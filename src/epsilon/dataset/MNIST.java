package epsilon.dataset;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class MNIST {
  private static final int BUFFER_SIZE = 4096;
  private double[][] inputs;
  private double[][] target;

  public MNIST() {
    try {
      File dir = new File(String.valueOf(MNIST.class.getResource("MNIST.zip").getFile())).getParentFile();
      if (!Files.exists(Path.of(MNIST.class.getResource("MNIST.zip").getFile()))) {
        extract(new ZipInputStream(new FileInputStream(String.valueOf(MNIST.class.getResource("MNIST.zip").getFile()))), dir);
      }
      ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(dir.getAbsolutePath() + File.separator + "MNIST" + File.separator + "input.ser")));
      ObjectInputStream oiss = new ObjectInputStream(new BufferedInputStream(new FileInputStream(dir.getAbsolutePath() + File.separator + "MNIST" + File.separator + "target.ser")));
      inputs = (double[][]) ois.readObject();
      target = (double[][]) oiss.readObject();
    } catch (IOException | ClassNotFoundException | NullPointerException ex) {
      ex.printStackTrace();
    }
  }

  private void extract(ZipInputStream zip, File target) throws IOException {
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

  public double[][] inputs() {
    return inputs;
  }

  public double[][] target() {
    return target;
  }
}
