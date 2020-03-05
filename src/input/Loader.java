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
 * @version 3.5.20
 */
public class Loader
{
   public static double LAMDA = 5;
   public static final String DEFAULT_WEIGHT_OUTPUT_FILE = "WEIGHTS.out";
   public static final int DEFAULT_MAX_ITERATIONS = 10000;
   public static final double DEFAULT_ERROR_THRESHOLD = 0.001;

   /**
    * Populate the given array with weights read from
    * the console.
    * 
    * @param weights an array with correct dimensions
    * @return the weights array populated with values
    *         from the console.
    */
   public static double[][][] readWeightsFromConsole(double[][][] weights)
   {
      Scanner sc = new Scanner(System.in);
      for (int n = 0; n < weights.length; n++)
         for (int k = 0; k < weights[n].length; k++)
            for (int j = 0; j < weights[n][k].length; j++)
            {
               System.out.println("Enter Weight for W(" + n + ", " + k + ", " + j + "): ");
               weights[n][k][j] = sc.nextDouble();
            }
      return weights;
   }

   /**
    * Asks the user for the number of input nodes
    * 
    * @return the number of input nodes chosen by the
    *         user.
    */
   public static int readNumInputNodesFromConsole()
   {
      System.out.println("Enter the number of Input Nodes: ");
      Scanner sc = new Scanner(System.in);
      return sc.nextInt();
   }

   /**
    * Asks the user for the number of nodes in hidden
    * layers
    * 
    * @return the number of nodes in hidden layers
    *         chosen by the user.
    */
   public static int readNumHiddenNodesFromConsole()
   {
      System.out.println("Enter the number of nodes in the hidden layer: ");
      Scanner sc = new Scanner(System.in);
      return sc.nextInt();
   }

   /**
    * Reads the activations for the network from the
    * user
    * 
    * @param numInputActivations the number of input
    *                            activations the
    *                            network has
    * @return an array with the activations for the
    *         network chosen by user.
    */
   public static double[] readActivationsFromConsole(int numInputActivations)
   {
      double[] activations = new double[numInputActivations];
      Scanner sc = new Scanner(System.in);
      for (int i = 0; i < numInputActivations; i++)
      {
         System.out.println("Enter the value of the node A0" + i);
         activations[i] = sc.nextDouble();
      }
      return activations;
   }

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
   public static Map<double[], Double> loadTrainSet(String filename) throws FileNotFoundException
   {
      Map<double[], Double> train = new HashMap<double[], Double>();
      Scanner sc = new Scanner(new File(filename));
      while (sc.hasNext())
      {
         String ln = sc.nextLine();
         String[] dat = ln.split(" ");
         double[] inputs = new double[dat.length];
         for (int i = 0; i < dat.length; i++)
         {
            if (!dat[i].trim().equals(""))
               inputs[i] = Double.valueOf(dat[i].trim());
         }
         double truth = Double.valueOf(sc.nextLine());
         train.put(inputs, truth);
      }
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
      int[] hidden = new int[1];

      int inputActivations = Integer.valueOf(args[0]);
      hidden[0] = Integer.valueOf(args[1]);
      double weightHigh = 2.0;
      double weightLow = -2.0;

      double lambda = Double.valueOf(args[2]);
      Map<double[], Double> training = new HashMap<double[], Double>();

      Network n = new Network(inputActivations, hidden, 1, lambda);

      if (args.length >= 4 && !args[3].trim().equals("")) 
      {
         weightLow = -Double.valueOf(args[3]);
         weightHigh = Double.valueOf(args[3]);
      }
      n.setRandWeights(weightLow, weightHigh);
      if (args.length < 5 || args[4].trim().equals(""))
      {
         System.err.println("NEEDS INPUT FILE FOR TRAINING SET!");
         System.exit(1);
      }
      else
      {
         training = loadTrainSet(args[4]);
      }
      int maxIterations = 0;
      if (args.length < 7 || args[6].trim().equals(""))
      {
         maxIterations = DEFAULT_MAX_ITERATIONS;
      }
      else
      {
         maxIterations = Integer.valueOf(args[6].trim());
      }
      double error = 0.0;

      if (args.length < 8 || args[7].trim().equals(""))
      {
         error = DEFAULT_ERROR_THRESHOLD;
      }
      else
      {
         error = Double.valueOf(args[7].trim());
      }
      n.setTrainingHyperparams(maxIterations, error);
      n.trainNetwork(training);
      System.out.println("Weight Range: "+ weightLow+" to "+weightHigh);
      if (args.length < 6 || args[5].trim().equals(""))
      {
         writeWeights(DEFAULT_WEIGHT_OUTPUT_FILE, n.getWeights());
         System.out.println("Weights written to: " + DEFAULT_WEIGHT_OUTPUT_FILE);
      }
      else
      {
         writeWeights(args[5], n.getWeights());
         System.out.println("Weights written to: " + args[5]);
      }
   }

}// public class Reader
