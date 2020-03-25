package perceptron;

import java.util.Arrays;
import java.util.Map;

/**
 * A feed-forward neural network that can do
 * forward propagation and has the ability to
 * train with multiple outputs simultaneously but
 * only 1 hidden layer
 * 
 * @author Harsh Deep Period 2
 * @version 3.23.20
 */
public class Network
{

   private double[][] nodes;        // Stores the activations of the nodes of the
                                    // network

   private double[][][] weights;    // Stores the weights of the network

   private double[][][] tempDelta;  // A temporary array for weight deltas, needed for
                                    // training

   private int max_iterations;      // The stopping conditions for the network
   private double error_threshold;

   private int numLayers;           // Stores the number of layers
   private int outputs;             // Stores the number of output nodes

   private double lambda;           // The lambda value for the network.

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

      tempDelta = new double[numLayers - 1][][]; // A temporary array with to store the
                                                 // intermediate deltas during minimizing error

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

      this.lambda = lambda;                     // Sets the lambda value for the network's
                                                // training.
      this.outputs = outputNodes;
   }// Network

   /**
    * Sets the training parameters
    * 
    * @param max_iterations  the number of iterations
    * @param error_threshold the error below which
    */
   public void setTrainingHyperparams(int max_iterations, double error_threshold)
   {
      this.error_threshold = error_threshold;
      this.max_iterations = max_iterations;
   }

   /**
    * Trains the network based on the given training
    * data set.
    * 
    * @param trainSet A map that maps input
    *                 activations the expected
    *                 output.
    */
   public void trainNetwork(Map<double[], double[]> trainSet)
   {
      printSummary();
      for (int iteration = 0; iteration < max_iterations; iteration++)  // Updates the weights up to a max number of
                                                                        // times. Stops after these many iterations
      {
         for (double[] inputs : trainSet.keySet())
         {
            updateWeights(inputs, trainSet.get(inputs));                            // Updates the weights in the network based on
                                                                                    // all of the training cases
         }
         double error = 0.0;
         double cases = 0;
         /*
          * Computes total error after every epoch and prints it
          */
         for (double[] inputs : trainSet.keySet())                          
         {
            nodes[0] = inputs;
            double[] out = forwardPropagation();
            double[] truth = trainSet.get(inputs);
            for (int i = 0; i < out.length; i++)
            {
               error += error(out[i], truth[i]);
            }
            cases++;
         }
         error /= (cases*(double)(outputs));                                        // Computes the average error
         
         System.out.println("Epoch #" + iteration + ", Average Error: " + error + ", Lambda: " + lambda);

         if (error < error_threshold)                                                 // Early stoppping if error below predefined
                                                                                      // stopping point
         {
            System.out.println("TRAINING ENDED EARLY: Error below threshold");
            printCases(trainSet);
            return;
         }
      }// for (int i = 0; i < MAX_ITERATIONS; i++)

      System.out.println("TRAINING ENDED: FINISHED " + max_iterations + " iterations.");
      printCases(trainSet);

   }// trainNetwork

   /**
    * Prints the outputs of the network on all of the
    * training set.
    * 
    * @param trainSet the training set of the
    *                 network.
    */
   public void printCases(Map<double[], double[]> trainSet)
   {
      /*
       * Prints out the produced value for each of the
       * outputs as well as the error
       */
      for (double[] inputs : trainSet.keySet())
      {
         nodes[0] = inputs;
         double[] out = forwardPropagation();
         double[] err = new double[out.length];
         double[] truth = trainSet.get(inputs);
         for (int i = 0; i < out.length; i++)
         {

            err[i] = error(out[i], truth[i]);
         }
         System.out.println("Inputs: " + Arrays.toString(inputs) + "\tOutput: " + Arrays.toString(out) + "\t\tExpected: "
               + Arrays.toString(trainSet.get(inputs)) + "\t\t Error: " + Arrays.toString(err));
      }

      printSummary();
   }// printCases(Map<double[], Double> trainSet)

   /**
    * Updates the weights based on the given
    * activations and the expected truth values.
    * 
    * @param activations The activations are the
    *                    input activation values
    * @param truth       the truth value for the
    *                    given activations
    */
   private void updateWeights(double[] activations, double[] truth)
   {
      nodes[0] = activations;                   // Sets the input activations and
      double[] F = forwardPropagation();       // does a forward propagation

      double[] omega = new double[outputs];    // computes the omega values
      for (int i = 0; i < outputs; i++)
      {
         omega[i] = truth[i] - F[i];
      }

      /*
       * Computes the theta values for last layer, the
       * dotproduct of the activations in the hidden
       * layer with the weights from hidden to output
       * layer.
       */

      double[] theta = new double[outputs];
      for (int i = 0; i < outputs; i++)
      {
         for (int j = 0; j < nodes[numLayers - 2].length; j++)
         {
            theta[i] += nodes[numLayers - 2][j] * weights[numLayers - 2][j][0];
         }
      }
      double[] psi = new double[outputs];
      /*
       * Computes psi based on the values for theta and
       * omega
       */
      for (int i = 0; i < outputs; i++)
      {
         psi[i] = omega[i] * activationDerivative(theta[i]);
      }

      /*
       * Puts the change in weights for each weight in
       * the final connectivity layer(between last
       * hidden layer and output node)in the temp array
       * based on the previously computed psi and the
       * value of the nodes
       * 
       */
      for (int i = 0; i < outputs; i++)
      {
         for (int j = 0; j < nodes[numLayers - 2].length; j++)
         {
            tempDelta[numLayers - 2][j][i] = lambda * psi[i] * nodes[numLayers - 2][j];
         }
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
             * between the input layer and the first hidden
             * layer
             */
            double thetaj = 0.0;
            for (int K = 0; K < nodes[numLayers - 3].length; K++)
            {
               thetaj += nodes[numLayers - 3][K] * weights[numLayers - 3][K][j];
            }

            /*
             * Computes the omega sub big j, the
             */
            double omegaj = 0.0;
            for (int i = 0; i < outputs; i++)
            {
               omegaj += psi[i] * weights[numLayers - 2][j][i];
            }

            double delta = lambda * nodes[numLayers - 3][k] * activationDerivative(thetaj) * omegaj;
            tempDelta[numLayers - 3][k][j] = delta;                             // Stores the result in the delta weight array
         }
      }

      /*
       * Updates the weights array based on the weight
       * deltas
       */
      for (int n = 0; n < weights.length; n++)
      {
         for (int k = 0; k < weights[n].length; k++)
         {
            for (int j = 0; j < weights[n][k].length; j++)
            {
               weights[n][k][j] += tempDelta[n][k][j];
               tempDelta[n][k][j] = 0.0;
            }
         }
      }
   }// updateWeights(double[] activations, double
    // truth)

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
      /*
       * Iterates across all elements in the weights
       * array and sets each one to a random value
       * between the upper and lower bound
       */
      for (int n = 0; n < weights.length; n++)
         for (int k = 0; k < weights[n].length; k++)
            for (int j = 0; j < weights[n][k].length; j++)
               weights[n][k][j] = random(lower, upper);
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
    * Prints a summary of the network
    */
   public void printSummary()
   {
      System.out.println("-------------------------");
      System.out.println("Network Summary:");
      System.out.println("Input Layer: " + nodes[0].length + " nodes");
      for (int n = 1; n < numLayers - 1; n++)
      {

         System.out.println("Hidden Layer " + n + ": " + nodes[n].length + " nodes");
      }
      System.out.println("Output Layer: " + nodes[numLayers - 1].length + " nodes");
      System.out.println("Lambda: " + lambda);
      System.out.println("Max_Iterations: " + max_iterations);
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

            double dotProduct = 0.0; // An accumulator for the dotproduct
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
      return 1.0 / (1.0 + Math.exp(-x));    // Sigmoid activation function
   }

   /**
    * Computes the error function based on the truth
    * and generated value
    * 
    * @param T the truth value
    * @param F the value to be compared to the truth
    * @return the error of the truth and generated
    *         value
    */
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
      return activated * (1.0 - activated); // Derivative of the activation function:
                                            // df/dx = f(x)(1-f(x))
   }
}// public class Network
