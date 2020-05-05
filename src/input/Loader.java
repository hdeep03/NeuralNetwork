package input;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import perceptron.Network;

/**
 * A Reader file that takes in input from various
 * sources and acts as a driver for the Network.
 * 
 * Methods:
 * Public:
 *  - double[][][] readWeights(String filename, double[][][] weights)
 *  - void writeWeights(String filename, double[][][] weights)
 *  - Map<double[], double[]> loadTrainSet(String filename)
 *  - void main(String[] args)
 *  
 * @author Harsh Deep Period 2
 * @version 5.1.20
 */
public class Loader
{
   public static final String DEFAULT_WEIGHT_OUTPUT_FILE = "./data/weights/weights";

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
      /*
       * Iterates across all of the weights in file and
       * populates the weights array
       */
      for (int n = 0; n < weights.length; n++)
         for (int k = 0; k < weights[n].length; k++)
            for (int j = 0; j < weights[n][k].length; j++)
            {
               weights[n][k][j] = sc.nextDouble();
            }
      sc.close();

      return weights;

   } // double[][][] readWeights(String filename, double[][][] weights)

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
      /*
       * Iterates across all of the weights in the array
       * and writes to file
       */
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
   } // writeWeights(String filename, double[][][] weights)

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
      /*
       * Iterates across all of the training sets in the
       * network.
       */
      while (sc.hasNext())
      {
         String ln = sc.nextLine();
         String[] dat = ln.split(" ");
         double[] inputs = new double[dat.length];

         /*
          * Stores all of the inputs for the given testing
          * case
          */
         for (int i = 0; i < dat.length; i++)
         {
            if (!dat[i].trim().equals(""))
               inputs[i] = Double.valueOf(dat[i].trim());
         }

         ln = sc.nextLine();
         dat = ln.split(" ");
         double[] outputs = new double[dat.length];

         /*
          * Stores all of the outputs for the given testing
          * case
          */
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
   } // Map<double[], double[]> loadTrainSet(String filename)

   /**
    * A driver class for the network
    * 
    * @param args the first argument should be the
    *             file path to the configuration
    *             file. The configuration file should
    *             list the number of nodes in each
    *             layer, the lambda value, the random
    *             weight range, the training set
    *             file, the max iterations, the error
    *             threshold, and (optionally) the
    *             weight output file all on a separate line.
    * @throws IOException if file cannot be opened
    */
   public static void main(String[] args) throws IOException
   {
      if (args.length != 1)                                                     // Checks for the presence of config filepath
      {
         System.err.println("Expected 1 argument(path to config file)");
         System.exit(1);
      }

      String config = args[0];                                                  // Gets the config filepath from the cmd arguments

      Scanner sc = new Scanner(new File(config));

      String[] structure = sc.nextLine().split(" ");
      
      int inputs = Integer.valueOf(structure[0]);                               // Sets number of input activations

      int[] hidden = new int[structure.length-2];
      
      /*
       * Sets the dimensions of each hidden layer 
       */
      for(int i = 0;i<structure.length-2;i++) 
      {
         hidden[i] = Integer.valueOf(structure[i+1]);
      }               

      int outputs = Integer.valueOf(structure[structure.length-1]);             // Sets number of output nodes

      double lambda = Double.valueOf(sc.nextLine());                            // Sets lambda(learning rate)

      Network n = new Network(inputs, hidden, outputs, lambda);

      String weight = sc.nextLine();

      /*
       * Sets the random weights for the network between
       * given high and low weights or reads in from
       * file
       */
      try
      {       
         double lowerWeightbound = Double.valueOf(weight.split(",")[0].trim());
         double higherWeightbound = Double.valueOf(weight.split(",")[1].trim());
         System.out.println("Set random weights in the range between "+lowerWeightbound+" and "+higherWeightbound);
         n.setRandWeights(lowerWeightbound, higherWeightbound);
      }
      catch (Exception e)
      {
         /*
          * Loads weights from separate file
          */
         n.setWeights(readWeights(weight, n.getWeights()));
         System.out.println("Loaded weights from file");
      }

      String trainset = sc.nextLine();
      Map<double[], double[]> training = new HashMap<double[], double[]>();             // Loads the training set for the network
      training = loadTrainSet(trainset);

      int maxIterations = Integer.valueOf(sc.nextLine());                               // Sets the stopping conditions for the
                                                                                        // network
      double error = Double.valueOf(sc.nextLine());
      
      System.out.println("Stopping Conditions: Max Iterations: "+maxIterations+"; Error threshold: "+error);

      n.setTrainingHyperparams(maxIterations, error);                                   // Starts training on the network
      n.trainNetwork(training);

      /*
       * Writes out the final weights to provided
       * file(or default with timestamp if none was provided)
       */ 
      String timestamp = new SimpleDateFormat("yyyy-MM-dd-HH.mm.ss").format(new Date());
      String targetFile = DEFAULT_WEIGHT_OUTPUT_FILE+timestamp+".out";
      
      if (sc.hasNext())
      {
          targetFile = sc.nextLine();
      }
      
      writeWeights(targetFile, n.getWeights());
      System.out.println("Weights written out to: \""+targetFile+"\"");
      
      sc.close();

   }// main(String[] args)

}// public class Loader
