package perceptron;

import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Map;

/**
 * A feed-forward neural network that can do
 * forward propagation and has the ability to
 * train with multiple outputs simultaneously with
 * any number of hidden layers
 * 
 * Methods:
 * Constructor: 
 *  - Network(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda)
 * Public:
 *  - void setTrainingHyperparams(int max_iterations, double error_threshold)
 *  - void trainNetwork(Map<double[], double[]> trainSet)
 *  - void printCases(Map<double[], double[]> trainSet)
 *  - double[][][] getWeights()
 *  - double random(double lower, double upper)
 *  - void setRandWeights(double lower, double upper)
 *  - void setWeights(double[][][] weights)
 *  - void setInputActivations(double[] input)
 *  - void printSummary()
 *  - double[] forwardPropagation()
 *  - 
 * Private:
 *  - void updateWeights(double[] activations, double[] truth)
 *  - double activation(double x)
 *  - double error(double T, double F)
 *  - double activationDerivative(double x)
 * 
 * @author Harsh Deep Period 2
 * @version 5.1.20
 */
public class Network
{

   private double[][] nodes;        // Stores the activations of the nodes of the
                                    // network

   private double[][][] weights;    // Stores the weights of the network

   private int[] dimensions;        // Stores the dimensions of each layer of the
                                    // network

   private int max_iterations;      // The stopping conditions for the network
   private double error_threshold;

   private int numLayers;           // Stores the number of layers
   private int outputs;             // Stores the number of output nodes

   private double lambda;           // The lambda value for the network.

   private double[][] theta;        // The theta values array used during training

   private double[][] psi;          // The psi values array used during training

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

      nodes[0] = new double[inputNodes];                        // Sets the correct dimensions for the nodes 2d-array
      nodes[numLayers - 1] = new double[outputNodes];

      dimensions = new int[numLayers];

      dimensions[0] = nodes[0].length;                          // Creates a dimensions array for the number of nodes in each layer
      dimensions[numLayers - 1] = nodes[numLayers - 1].length;

      /*
       * Sets up the nodes array with the jagged array
       * which is defined by the values in
       * "hiddenLayerNodes"
       * 
       */
      for (int n = 0; n < hiddenLayerNodes.length; n++)
      {
         nodes[n + 1] = new double[hiddenLayerNodes[n]];
         dimensions[n + 1] = hiddenLayerNodes[n];
      }

      weights = new double[numLayers - 1][][]; // The number of connection layers is one less
                                               // than the number of layer

      /*
       * Initializes the weights array based on the
       * dimensions of the nodes array. Weight array may
       * be jagged.
       */
      for (int n = 0; n < numLayers - 1; n++)

      {
         weights[n] = new double[dimensions[n]][dimensions[n + 1]];
      }

      this.lambda = lambda;                    // Sets the lambda value for the network's
                                               // training.
      this.outputs = outputNodes;

      theta = new double[numLayers][];         // Sets up the jaged array for theta values
      psi = new double[numLayers][];           // Sets up the jaged array for psi values
      for (int n = 0; n < numLayers; n++)
      {
         theta[n] = new double[dimensions[n]];
         psi[n] = new double[dimensions[n]];
      }

   }// Network(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda)

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
      double[] out = null;
      double[] truth = null;
      printSummary();                                                                 // Print data about the network configuration

      String timestamp = new SimpleDateFormat("yyyy-MM-dd-HH.mm.ss").format(new Date());
      System.out.println("TRAINING STARTED AT " + timestamp);                          // Print a time stamp for when the network
                                                                                       // started to train

      boolean error_threshold_met = false;                                             // boolean to check if the error threshold is met and
                                                                                       // exit loop early
      
      
      int iteration = 0;
      /*
       * Updates the weights up to a max number of
       * times. Stops after these many interations or if
       * error is low enough
       */
      while (iteration <= max_iterations && !error_threshold_met)
      {
         iteration++;
         
         for (double[] inputs : trainSet.keySet())
         {
            updateWeights(inputs, trainSet.get(inputs));                               // Updates the weights in the network based
                                                                                       // on
                                                                                       // all of the training cases
         }

         double error = 0.0;
         int cases = 0;
         /*
          * Computes total error after every epoch and
          * prints it
          */
         for (double[] inputs : trainSet.keySet())
         {
            nodes[0] = inputs;
            out = forwardPropagation();
            truth = trainSet.get(inputs);

            for (int i = 0; i < outputs; i++)
            {
               error += error(out[i], truth[i]);
            }
            cases++;
         } // for (double[] inputs : trainSet.keySet())

         error /= (((double) cases) * (double) (outputs));           // Computes the average error

         if (error < error_threshold)                                // Early stopping if error below predefined stopping point
         {
            error_threshold_met = true;
         }
      }// while (iteration <= max_iterations && !error_threshold_met)

      timestamp = new SimpleDateFormat("yyyy-MM-dd-HH.mm.ss").format(new Date());
      System.out.println("TRAINING ENDED AT " + timestamp);                                    // Print a time stamp for when the
                                                                                               // network ended training
      /*
       * Prints a completion message depending on how
       * training concluded.
       */
      if (error_threshold_met)
         System.out.println("TRAINING ENDED EARLY AFTER " + iteration + " ITERATIONS: Error below threshold");
      else
         System.out.println("TRAINING ENDED: MAXIMUM ITERATIONS REACHED: " + max_iterations + " ITERATIONS.");

      printCases(trainSet);

   }// trainNetwork(Map<double[], double[]> trainSet)

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
         double[] err = new double[outputs];
         double[] truth = trainSet.get(inputs);
         for (int i = 0; i < outputs; i++)
         {

            err[i] = error(out[i], truth[i]);
         }
         System.out.println("Inputs: " + Arrays.toString(inputs) + "\tOutput: " + Arrays.toString(out) + "\t\tExpected: "
               + Arrays.toString(trainSet.get(inputs)) + "\t\t Error: " + Arrays.toString(err));
      }

      printSummary();
   }// printCases(Map<double[], double[]> trainSet)

   

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
   }// setRandWeights(double lower, double upper)

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
      System.out.println("Input Layer: " + dimensions[0] + " nodes");
      for (int n = 1; n < numLayers - 1; n++)
      {

         System.out.println("Hidden Layer " + n + ": " + dimensions[n] + " nodes");
      }
      System.out.println("Output Layer: " + dimensions[numLayers - 1] + " nodes");
      System.out.println("Lambda: " + lambda);
      System.out.println("Max_Iterations: " + max_iterations);
   } // printSummary()

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
         for (int j = 0; j < dimensions[n]; j++)
         {

            double dotProduct = 0.0; // An accumulator for the dot product
            /*
             * Computes the dot product of the input values to
             * a node and the weights.
             */
            for (int i = 0; i < dimensions[n - 1]; i++)
            {
               dotProduct += nodes[n - 1][i] * weights[n - 1][i][j];
            }

            double activated = activation(dotProduct);   // Applies the activation function to the
                                                         // dot product
            nodes[n][j] = activated;

         } // for (int j = 0; j < dimensions[n]; j++)

      } // for (int n = 1; n < numLayers; n++)
      return nodes[numLayers - 1];

   }// forwardPropogation()

   /**
    * Updates the weights based on the given
    * activations and the expected truth values using
    * the backpropagation algorithm
    * 
    * @param activations The activations are the
    *                    input activation values
    * @param truth       the truth values for the
    *                    given activations
    */
   private void updateWeights(double[] activations, double[] truth)
   {
      nodes[0] = activations;                   // Sets the input activations

      /*
       * Does a forward propagation on the network &
       * stores theta values and activations.
       */
      for (int n = 1; n < numLayers; n++)
      {
         /*
          * Iterates across nodes in layer n and stores the
          * theta values and updates the values of the
          * nodes in this layer
          */
         for (int j = 0; j < dimensions[n]; j++)
         {
            theta[n][j] = 0.0;

            /*
             * Computes the theta values for each node in
             * layer n.
             */
            for (int k = 0; k < dimensions[n - 1]; k++)
            {
               theta[n][j] += nodes[n - 1][k] * weights[n - 1][k][j];           // Evaluates theta for each node in this layer
            }
            nodes[n][j] = activation(theta[n][j]);                              // Updates the value of the node by applying
                                                                                // activation to the theta
         }// for (int j = 0; j < dimensions[n]; j++)

      } // for (int n = 1; n < numLayers; n++)

      /*
       * Computes Psi_I and the omega values in the last
       * layer.
       */
      for (int i = 0; i < outputs; i++)
      {
         psi[numLayers - 1][i] = (truth[i] - nodes[numLayers - 1][i]) * activationDerivative(theta[numLayers - 1][i]);
      }

      /*
       * This is the backpropagation loop. It starts
       * from the last layer and works its way backward
       * through the network
       */
      for (int n = numLayers - 1; n >= 1; n--)
      {
         /*
          * Iterates through the n-1 layer and computes the
          * psi values for n-1 layer
          */
         for (int k = 0; k < dimensions[n - 1]; k++)
         {
            double omega = 0.0;
            /*
             * Iterates through the nodes in layer n and
             * computes the omegas then applies changes to the
             * weights between n-1 and n layer.
             */
            for (int j = 0; j < dimensions[n]; j++)
            {
               omega += weights[n - 1][k][j] * psi[n][j];                       // The calculation of the omega value which is
                                                                                // used to find the psi values for n-1 layer

               weights[n - 1][k][j] += lambda * nodes[n - 1][k] * psi[n][j];    // Updates the weights based on the psi values
                                                                                // from layer n and is done after omega is
                                                                                // calculated.
            }// for (int j = 0; j < dimensions[n]; j++)

            psi[n - 1][k] = omega * activationDerivative(theta[n - 1][k]);      // Using the computed omega, computes the psi
                                                                                // values in the n-1 layer.

         }// for (int k = 0; k < dimensions[n-1]; k++)

      } // for (int n = numLayers - 1; n >= 1; n--)

   }// updateWeights(double[] activations, double[] truth)
   
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
   private double activation(double x)
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
   private double activationDerivative(double x)
   {
      double activated = activation(x);
      return activated * (1.0 - activated); // Derivative of the activation function:
                                            // df/dx = f(x)(1-f(x))
   }
}// public class Network
