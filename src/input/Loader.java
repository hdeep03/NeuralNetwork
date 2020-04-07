package input;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import perceptron.Network;

/**
 * A Reader file that takes in input from various
 * sources and acts as a driver for the Network.
 * 
 * @author Harsh Deep Period 2
 * @version 4.7.20
 */
public class Loader
{
   public static final String DEFAULT_WEIGHT_OUTPUT_FILE = "weights.output";

   /**
    * Reads weights from the given file
    * 
    * @param filename the path to the weights file
    * @param weights  an array of dimensions to be
    *                 populated with read values
    * @return an array with same array as that of
    *         weights populated with the weights read
    *         from file
    * @throws FileNotFoundException File could be
    *                               found.
    */
   public static double[][][] readWeights(String filename, double[][][] weights) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new File(filename));
      for (int n = 0; n < weights.length; n++)
         for (int k = 0; k < weights[n].length; k++)
            for (int j = 0; j < weights[n][k].length; j++)
            {
               weights[n][k][j] = sc.nextDouble();
            }
      sc.close();

      return weights;

   }

   /**
    * Write out the weights to file
    * 
    * @param filename the filepath to write the files
    *                 to
    * @param weights  the weights to be written out
    *                 to file
    * @throws IOException
    */
   public static void writeWeights(String filename, double[][][] weights) throws IOException
   {
      PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename)));
      for (int n = 0; n < weights.length; n++)
      {
         for (int k = 0; k < weights[n].length; k++)
         {
            for (int j = 0; j < weights[n][k].length; j++)
            {
               out.print(weights[n][k][j] + " ");
            }
         }
         out.println();
      }

      out.flush();
      out.close();
   }

   /**
    * Loads the training set from file
    * 
    * @param filename The file from which to read the
    *                 training set
    * @return A map with input activations mapping to
    *         the expected output
    * @throws FileNotFoundException
    */
   public static Map<double[], double[]> loadTrainSet(String filename) throws FileNotFoundException
   {
      Map<double[], double[]> train = new HashMap<double[], double[]>();
      Scanner sc = new Scanner(new File(filename));

      while (sc.hasNext())                              // iterates across the training set
      {
         String ln = sc.nextLine();
         String[] dat = ln.split(" ");
         double[] inputs = new double[dat.length];

         for (int i = 0; i < dat.length; i++)
         {
            if (!dat[i].trim().equals(""))
               inputs[i] = Double.valueOf(dat[i].trim());
         }

         ln = sc.nextLine();
         dat = ln.split(" ");
         double[] outputs = new double[dat.length];

         for (int i = 0; i < dat.length; i++)
         {
            if (!dat[i].trim().equals(""))
               outputs[i] = Double.valueOf(dat[i].trim());
         }

         train.put(inputs, outputs);
      }
      sc.close();

      System.out.println("Loaded " + train.size() + " training cases.");

      return train;
   }

   /**
    * A driver class for the network
    * 
    * @param args the command line arguments.
    * @throws IOException
    */
   public static void main(String[] args) throws IOException
   {
      if (args.length != 1)
      {
         System.err.println("Expected 1 argument(path to config file)");
      }

      String config = args[0];

      Scanner sc = new Scanner(new File(config));

      int inputs = Integer.valueOf(sc.nextLine());              // Sets number of input activations

      int[] hidden = new int[1];
      hidden[0] = Integer.valueOf(sc.nextLine());               // Number of hidden nodes in hidden layer

      int outputs = Integer.valueOf(sc.nextLine());             // Sets number of output nodes

      double lambda = Double.valueOf(sc.nextLine());            // Sets lambda(learning rate)

      Network n = new Network(inputs, hidden, outputs, lambda);

      String weight = sc.nextLine();

      try
      {
         /*
          * Sets the random weights for the network between
          * given high and low weights or reads in from
          * file
          */
         double lowerWeightbound = Double.valueOf(weight.split(",")[0].trim());
         double higherWeightbound = Double.valueOf(weight.split(",")[1].trim());
         n.setRandWeights(lowerWeightbound, higherWeightbound);
      }
      catch (Exception e)
      {
         /*
          * Loads weights from seperate file
          */
         n.setWeights(readWeights(weight, n.getWeights()));
      }

      String trainset = sc.nextLine();
      Map<double[], double[]> training = new HashMap<double[], double[]>();             // Loads the training set for the network
      training = loadTrainSet(trainset);

      int maxIterations = Integer.valueOf(sc.nextLine());                               // Sets the stopping conditions for the
                                                                                        // network
      double error = Double.valueOf(sc.nextLine());

      n.setTrainingHyperparams(maxIterations, error);                                   // Starts training on the network
      n.trainNetwork(training);

      /*
       * Writes out the final weights to provided file(or default if none was provided)
       */
      if (!sc.hasNext())
      {
         writeWeights(DEFAULT_WEIGHT_OUTPUT_FILE, n.getWeights());
         System.out.println("Weights written to: " + DEFAULT_WEIGHT_OUTPUT_FILE);
      }
      else
      {
         String output = sc.nextLine();
         writeWeights(output, n.getWeights());
         System.out.println("Weights written to: " + output);
      }
      sc.close();

   }

}// public class Loader
