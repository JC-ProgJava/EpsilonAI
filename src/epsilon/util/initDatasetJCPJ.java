package epsilon.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class initDatasetJCPJ {
  public static void main(String[] args) throws IOException {
    double[][] input = new double[107730][784];
    System.out.println(Arrays.toString(scan("/Users/JC/Desktop/dataset/0/0\\0.png")));
  }

  public static double[] scan(String filename) throws IOException {
    File input = new File(filename);
    BufferedImage image = ImageIO.read(input);
    Graphics2D bGr = image.createGraphics();
    bGr.drawImage(image, 0, 0, null);
    bGr.dispose();
    double[] a = new double[784];
    int counter = 0;
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        Color c = new Color(image.getRGB(j, i));
        a[counter] = 255.0 - (((double) c.getRed()) * 0.30) - (((double) c.getGreen()) * 0.59) - (((double) c.getBlue()) * 0.11);
        a[counter] /= 255.0;
        counter++;
      }
    }
    return a;
  }
}
