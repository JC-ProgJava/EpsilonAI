package epsilon.dataset;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class MNISTValidate {
  private static final int BUFFER_SIZE = 4096;
  private double[][] inputs;
  private double[][] target;

  public MNISTValidate() {
    try {
      Path path = Files.createTempFile(null, null);
      InputStream stream = MNIST.class.getResourceAsStream("MNISTValidate.zip");
      OutputStream outStream = new FileOutputStream(path.toFile());

      byte[] buffer = new byte[2048];
      int bytesRead;
      while ((bytesRead = stream.read(buffer)) != -1) {
        outStream.write(buffer, 0, bytesRead);
      }

      extract(new ZipInputStream(new FileInputStream(path.toString())), new File(path.getParent().toString()));
      ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path.getParent().toString() + File.separator + "MNISTValidate" + File.separator + "input.ser")));
      ObjectInputStream oiss = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path.getParent().toString() + File.separator + "MNISTValidate" + File.separator + "target.ser")));
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
