package perceptron;

/**
 * A feed-forward neural network
 * 
 * @author Harsh Deep Period 2
 * @version 2.20.20
 */
public class Network
{
   private double[][] nodes;
   private double[][][] weights;
   private int numLayers;

   /**
    * Initializes the nodes and weights for the
    * network
    * 
    * @param inputNodes       the number of input
    *                         nodes to the network
    * @param hiddenLayerNodes an array with the
    *                         length of each hidden
    *                         layer nodes
    * @param outputNodes      the number of output
    *                         nodes
    */
   public Network(int inputNodes, int[] hiddenLayerNodes, int outputNodes)
   {
      nodes = new double[2 + hiddenLayerNodes.length][];
      numLayers = nodes.length;

      nodes[0] = new double[inputNodes];
      nodes[numLayers - 1] = new double[outputNodes];
      /*
       * Sets up the nodes array with the jagged array
       * which is defined by the values in
       * "hiddenLayerNodes"
       * 
       */
      for (int n = 0; n < hiddenLayerNodes.length; n++)
      {
         nodes[n + 1] = new double[hiddenLayerNodes[n]];
      }

      weights = new double[numLayers - 1][][]; // The number of connection layers is one less
                                               // than the number of layer

      /*
       * Initializes the weights array based on the
       * dimensions of the nodes array. Weight array may be jagged.
       */
      for (int n = 0; n < numLayers - 1; n++)

      {
         weights[n] = new double[nodes[n].length][nodes[n + 1].length];
      }
   }

   /**
    * Retrieves the current weights for the network.
    * 
    * @return the weights of the network.
    */
   public double[][][] getWeights()
   {
      return weights;
   }

   /**
    * Produces random values from lower(inclusive) to
    * upper(exclusive)
    * 
    * @param lower the lower bound(inclusive) to
    *              generate random numbers from
    * @param upper the upper bound(exclusive) to
    *              generate numbers to
    * @return A random number in the range
    *         lower(inclusive) to upper(exclusive)
    */
   public double random(double lower, double upper)
   {
      return lower + Math.random() * (upper - lower);
   }

   /**
    * Sets up the weights with random numbers in the
    * range
    * 
    * @param lower a double that is the lower
    *              bound(inclusive) of random values
    *              for each weight
    * @param upper a double that is the upper
    *              bound(exclusive) of random values
    *              for each weight
    */
   public void setRandWeights(double lower, double upper)
   {
      for (int n = 0; n < weights.length; n++)
         for (int j = 0; j < weights[n].length; j++)
            for (int i = 0; i < weights[n][j].length; i++)
               weights[n][j][i] = random(lower, upper); // Iterates across all elements in the weights
                                                        // array and sets each one to a random value
                                                        // between the upper and lower bound
   }

   /**
    * Sets the weights to the given weights
    * 
    * @param weights the new weights
    */
   public void setWeights(double[][][] weights)
   {
      this.weights = weights;
   }

   /**
    * Sets the input activations to the given values.
    * 
    * @param input the new input activations
    */
   public void setInputActivations(double[] input)
   {
      nodes[0] = input;
   }

   /**
    * Computes the intermediate values for the
    * network based on the input activations and the
    * weights.
    * 
    * @precondition Assumes that the first layer of
    *               the nodes array is populated with
    *               the input
    * @return the output layer of the network
    */
   public double[] forwardPropagation()
   {
      /*
       * Iterates across all connectivity layers. n is
       * the layer of the child node.
       * 
       */
      for (int n = 1; n < numLayers; n++)
      {
         /*
          * Iterates across all of the child nodes and
          * computes their new values
          */
         for (int j = 0; j < nodes[n].length; j++)
         {

            System.out.print("DEBUG: a[" + n + "][" + j + "] = ");

            double dotProduct = 0.0;
            for (int i = 0; i < nodes[n - 1].length; i++) // Computes the dot product of the input values to
                                                          // a node and the weights.
            {
               System.out.print("a[" + (n - 1) + "][" + i + "]*w[" + (n - 1) + "][" + i + "][" + j + "]+");
               dotProduct += nodes[n - 1][i] * weights[n - 1][i][j];
            }

            double activated = activation(dotProduct);   // Applies the activation function to the
                                                         // dotproduct

            System.out.println("|\t a[" + n + "][" + j + "]= " + activated);

            nodes[n][j] = activated;
         } // for (int j = 0; j < nodes[n].length; j++)

      } // for (int n = 1; n < nodes.length; n++)
      return nodes[numLayers - 1];
   }// forwardPropogation

   /**
    * Applies the activation function to the input
    * variable
    * 
    * @param x The value to evaluate the activation
    *          function at.
    * 
    * @return the value of the activation function
    *         evaluated at the given value.
    */
   public double activation(double x)
   {
      return x; // The activation function should be changed to
                // suit needs. If changed, the activation_deriv
                // should be changed as well to be the derivative
                // of this function
   }

   /**
    * The derivative of the activation function.
    * 
    * @param x the value to evaluate the derivative
    *          at
    * @return the derivative of the activation
    *         function at the given value.
    */
   public double activation_deriv(double x)
   {
      return 1.0; // The derivative of the activation function. This
                  // should be changed to be the derivative of the
                  // activation function.
   }
}// public class Network
