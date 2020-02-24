package input;
import java.util.Scanner;
import perceptron.Network;

/**
 * A Reader file that takes in input from various
 * sources and acts as a driver for the Network.
 * 
 * @author Harsh Deep Period 2
 * @version 2.23.20
 */
public class Reader
{
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
    * A driver class for the network
    * 
    * @param args the command line arguments.
    */
   public static void main(String[] args)
   {
      int[] hidden = new int[1];
      hidden[0] = readNumHiddenNodesFromConsole();
      int inputActivations = readNumInputNodesFromConsole();
      Network n = new Network(inputActivations, hidden, 1);

      n.setWeights(readWeightsFromConsole(n.getWeights()));
      Scanner sc = new Scanner(System.in);
      boolean flag = true;
      while (flag)
      {

         n.setInputActivations(readActivationsFromConsole(inputActivations));
         System.out.println(n.forwardPropagation()[0]);
         System.out.println("Would you like to continue?[Y/N]");
         if (sc.next().equalsIgnoreCase("N"))
         {
            flag = false;
         }
      } // while(flag)
   }

}// public class Reader
