package perceptron;

import java.util.Map;

/**
 * A feed-forward neural network that can do
 * forward propagation.
 * 
 * @author Harsh Deep Period 2
 * @version 2.24.20
 */
public class Network
{
   private double[][] nodes;        // Nodes will be referenced by the n, j notation
                                    // where n is the layer and j is the particular
                                    // node in this layer

   private double[][][] weights;    // Weights will be referenced by n,j,i notation
                                    // where n is starting node layer, j is parent
                                    // node in this node layer, and i is the child
                                    // node in the n+1 layer

   private double[][][] tempDelta;
   
   private static final int MAX_ITERATIONS = 100;
   private static final double ERROR_THRESHOLD = 0.001;
   
   private int numLayers;
   private double lambda = 5;

   /**
    * Initializes the nodes and weights for the
    * network. Nodes and weights are all set to
    * values of 0.
    * 
    * @param inputNodes       the number of input
    *                         nodes to the network
    * @param hiddenLayerNodes an array with the
    *                         length of each hidden
    *                         layer nodes
    * @param outputNodes      the number of output
    *                         nodes
    */
   public Network(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda)
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

      tempDelta = new double[numLayers - 1][][]; // A temporary array with the same dimensions of
                                                 // weights that is used to store the intermediate
                                                 // deltas during minimizing error

      /*
       * Initializes the weights array based on the
       * dimensions of the nodes array. Weight array may
       * be jagged.
       */
      for (int n = 0; n < numLayers - 1; n++)

      {
         weights[n] = new double[nodes[n].length][nodes[n + 1].length];
         tempDelta[n] = new double[nodes[n].length][nodes[n + 1].length];
      }

      this.lambda = lambda;                     // Sets the lambda value for the network's training.
   }// Network

   /**
    * Trains the network based on the given training
    * data set.
    * 
    * @param trainSet A map that maps input
    *                 activations the expected
    *                 output.
    */
   public void trainNetwork(Map<double[], Double> trainSet)
   {
      for (int i = 0; i < MAX_ITERATIONS; i++)
      {
         for (double[] inputs : trainSet.keySet())
         {
            updateWeights(inputs, trainSet.get(inputs));
         }
         double sum = 0.0;
         for (double[] inputs : trainSet.keySet())
         {
            nodes[0] = inputs;
            double out = forwardPropagation()[0];
            double err = error(out, trainSet.get(inputs));
            sum += err;
         }
         if (sum < ERROR_THRESHOLD)
         {
            System.out.println("TRAINING ENDED EARLY: Error below threshold");
            return;
         }
         System.out.println("Step #" + i + ", Total Error: " + sum);
      }
      System.out.println("TRAINING ENDED: FINISHED " + MAX_ITERATIONS + " iterations.");

   }

   /**
    * Updates the weights based on the given
    * activations and the expected truth values.
    * 
    * @param activations The activations are the
    *                    input activation values
    * @param truth       the truth value for the
    *                    given activations
    */
   private void updateWeights(double[] activations, double truth)
   {
      nodes[0] = activations;                   // Runs the activations through the network and
      double F = forwardPropagation()[0];       // does a forward propagation

      double omega = truth - F;                 // Computes the omega value defined as the
                                                // difference between the truth and output value
      /*
       * Computes the theta sub zero value, the
       * dotproduct of the activations in the hidden
       * layer with the weights from hidden to output
       * layer.
       */

      double theta = 0.0;
      for (int j = 0; j < nodes[numLayers - 2].length; j++)
      {
         theta += nodes[numLayers - 2][j] * weights[numLayers - 2][j][0];
      }

      double psi = omega * activationDerivative(theta);        // computes psi zero based on the omega and theta
                                                               // sub zero.

      /*
       * Puts the change in weights for each weight in
       * the final connectivity layer(between last
       * hidden layer and output node)in the temp array
       * based on the previously computed psi and the
       * value of the nodes
       * 
       */
      for (int j = 0; j < nodes[numLayers - 2].length; j++)
      {
         tempDelta[numLayers - 2][j][0] = lambda * psi * nodes[numLayers - 2][j];
      }

      /*
       * Iterates across all weights between the input
       * layer and the hidden layer and computes the
       * change in weights for each of them.
       */
      for (int k = 0; k < nodes[numLayers - 3].length; k++)
      {
         for (int j = 0; j < nodes[numLayers - 2].length; j++)
         {
            /*
             * Computes theta sub small j, the dotproduct of
             * the input activations and the set of weights
             * between the input layer and the first hidden layer
             */
            double thetaj = 0.0;
            for (int K = 0; K < nodes[numLayers - 3].length; K++)
            {
               thetaj += nodes[numLayers - 3][K] * weights[numLayers - 3][K][j];
            }
            
            
            double delta = lambda * nodes[numLayers - 3][k] * activationDerivative(thetaj) * psi * weights[numLayers - 2][j][0]; // Computes the needed weight change
            
            tempDelta[numLayers - 3][k][j] = delta;                                                                              // Stores the result in the weight delta array
         }
      }

      /*
       * Updates the weights array based on the weight
       * deltas and zeros out the weight delta array
       */
      for (int n = 0; n < weights.length; n++)
      {
         for (int j = 0; j < weights[n].length; j++)
         {
            for (int i = 0; i < weights[n][j].length; i++)
            {
               weights[n][j][i] += tempDelta[n][j][i];      // Updates the weights array

               tempDelta[n][j][i] = 0.0;                    // Zeroes out the weight delta array
            }
         }
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
       */
      for (int n = 1; n < numLayers; n++)
      {
         /*
          * Iterates across all of the child nodes and
          * computes their new values
          */
         for (int j = 0; j < nodes[n].length; j++)
         {

            double dotProduct = 0.0;
            /*
             * Computes the dot product of the input values to
             * a node and the weights.
             */
            for (int i = 0; i < nodes[n - 1].length; i++)
            {
               dotProduct += nodes[n - 1][i] * weights[n - 1][i][j];
            }

            double activated = activation(dotProduct);   // Applies the activation function to the
                                                         // dotproduct

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
      return 1.0 / (1.0 + Math.exp(-x)); // The activation function should be changed to
                                         // suit needs. If changed, the activation_deriv
                                         // should be changed as well to be the derivative
                                         // of this function
   }

   private double error(double T, double F)
   {
      double omega = (T - F);
      return 0.5 * omega * omega;
   }

   /**
    * The derivative of the activation function.
    * 
    * @param x the value to evaluate the derivative
    *          at
    * @return the derivative of the activation
    *         function at the given value.
    */
   public double activationDerivative(double x)
   {
      double activated = activation(x);
      return activated * (1.0 - activated); // The derivative of the activation function. This
                                            // should be changed to be the derivative of the
                                            // activation function.
   }
}// public class Network
