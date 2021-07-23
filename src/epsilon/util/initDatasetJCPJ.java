package epsilon.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;

public class initDatasetJCPJ {
  public static void main(String[] args) throws IOException {
    double[][] input = new double[107730][784];
    double[][] output = new double[107730][10];
    String filepath = "/Users/JC/Desktop/dataset/";
    int counter = 0;
    for (int i = 0; i < 10773; i++) {
      if (i % 1000 == 0) {
        System.out.println(i + "/" + 10772);
      }
      for (int fold = 0; fold < 10; fold++) {
        input[counter] = scan(filepath + fold + "/" + fold + "\\" + i + ".png");
        Arrays.fill(output[counter], 0.0);
        output[counter][fold] = 1.0;
        counter++;
      }
    }
    try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream("inputJCPJ.ser")))) {
      oos.writeObject(input);
    } catch (IOException ex) {
      ex.printStackTrace();
    }

    try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream("outputJCPJ.ser")))) {
      oos.writeObject(output);
    } catch (IOException ex) {
      ex.printStackTrace();
    }
  }

  private static boolean isTransparent(int val) {
    return val >> 24 == 0;
  }

  public static double[] scan(String filename) throws IOException {
    BufferedImage image = ImageIO.read(new File(filename));
    double[] in = new double[784];
    int counter = 0;
    for (int i = 0; i < image.getHeight(); i++) {
      for (int j = 0; j < image.getWidth(); j++) {
        Color color = new Color(image.getRGB(j, i), true);
        if (isTransparent(image.getRGB(j, i))) {
          color = new Color(255, 255, 255);
        }
        int r = color.getRed();
        int g = color.getGreen();
        int b = color.getBlue();
        in[counter] = 255.0 - (((double) r) * 0.30) - (((double) g) * 0.59) - (((double) b) * 0.11);
        in[counter] /= 255;
        counter++;
      }
    }

    return in;
  }
}
