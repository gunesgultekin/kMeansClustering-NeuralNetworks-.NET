using Accord.Math;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using Microsoft.ML.Data;
using NeuralNetworks.Data;
using OneOf.Types;
using System;
using System.Globalization;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static Tensorflow.ApiDef.Types;


namespace NeuralNetworks.Controllers
{
    [ApiController]
    [Route("/MLPController")]
    public class MLPController : ControllerBase
    {

        private DatabaseContext _context; // ENTITY FRAMEWORK ORM DATABASE CONTEXT

        public MLPController(DatabaseContext context)
        {
             this._context = context;
        }

        // GET THE DETAILED VIEW OF ALL THE LAYERS, NEURONS AND NEURON WEIGHTS OF THE TRAINED MLP NEURAL NETWORK
        [HttpGet("GetNetworkReports")]
        public List <string> GetNetworkReports(ActivationNetwork network)
        {
            List<NetworkAttributes> networkAttributeReport = new List<NetworkAttributes>();

            List<string> results = new List<string>();

            for (int i = 0; i < network.Layers.Length - 1; ++i)
            {
                NetworkAttributes currentLayer = new NetworkAttributes(); // CREATE NEW NETWORK ATTRIBUTES INSTANCE
                currentLayer.layerID = i; // CURRENT LAYER ID = i
                currentLayer.neuronWeights = new double[network.Layers[i].Neurons.Length][]; // INITIALIZE DOUBLE ARRAY TO STORE THE NEURON WEIGHTS

               
                for (int j = 0; j < network.Layers[i].Neurons.Length; ++j)
                {
                    currentLayer.neuronWeights[j] = network.Layers[i].Neurons[j].Weights; // GET THE CURRENT LAYER'S NEURON WEIGHTS (EX: LAYER-0 NEURON 5])
                    results.Add(
                    "LAYER - " + i +"  " + "NEURON - " + " "+ j +
                    " NEURON WEIGHTS:" +"  " + ConvertArrayToString(currentLayer.neuronWeights[j]) 
                   ); // ADD CURRENT WEIGHTS TO RESULTS LIST

                }
                networkAttributeReport.Add(currentLayer); // ADD TO LIST
            }
            return results;
        }

        // SUB-FUNCTION THAT CONVERTS DOUBLE ARRAY TO STRING
        static string ConvertArrayToString(double[] array)
        {
            string result = "";

            for (int i = 0; i < array.Length; i++)
            {
                result += array[i];

                if (i < array.Length - 1)
                {
                    result += ",";
                }
            }

            return result;
        }

        // CREATE A MULTI-LAYERED NEURAL NETWORK WITH DESIRED NUMBER OF HIDDEN LAYERS, NEURONS PER LAYER AND TOTAL EPOCHS
        [HttpGet("createMLPNeuralNetwork")]
        public string createMLPNeuralNetwork(int hiddenLayerNumber, int neuronPerLayer ,int totalEpochs)
        {
           
            List<part1_train> data = _context.part1_train.ToList(); // GET TRAINING DATASET FROM DATABASE 

            double[][] trainingInputs = new double[data.Count()][]; // CREATE A 2D ARRAY TO STORE TRAINING DATA
            double[][] outputs = new double[data.Count()][]; // CREATE A 2D ARRAY TO STORE OUTPUTS (ANGLE-ACC-ARM) FROM TRAINING DATA

            // MAKE NECESSARY DATA CONVERSIONS THEN FILL THE TRAINING INPUTS
            for (int i=0; i<data.Count();++i)
            {
                trainingInputs[i] = new double[]
                {
                    double.Parse(data[i].theta1,CultureInfo.InvariantCulture),
                    double.Parse(data[i].theta2,CultureInfo.InvariantCulture),
                    double.Parse(data[i].theta3,CultureInfo.InvariantCulture),
                    double.Parse(data[i].theta4, CultureInfo.InvariantCulture),
                    double.Parse(data[i].theta5, CultureInfo.InvariantCulture),
                    double.Parse(data[i].theta6, CultureInfo.InvariantCulture),
                    double.Parse(data[i].thetad1, CultureInfo.InvariantCulture),
                    double.Parse(data[i].thetad2, CultureInfo.InvariantCulture),
                    double.Parse(data[i].thetad3, CultureInfo.InvariantCulture),
                    double.Parse(data[i].thetad4, CultureInfo.InvariantCulture),
                    double.Parse(data[i].thetad5, CultureInfo.InvariantCulture),
                    double.Parse(data[i].thetad6, CultureInfo.InvariantCulture),
                    double.Parse(data[i].tau1, CultureInfo.InvariantCulture),
                    double.Parse(data[i].tau2, CultureInfo.InvariantCulture),
                    double.Parse(data[i].tau3, CultureInfo.InvariantCulture),
                    double.Parse(data[i].tau4, CultureInfo.InvariantCulture),
                    double.Parse(data[i].tau5, CultureInfo.InvariantCulture),
                    double.Parse(data[i].dm1, CultureInfo.InvariantCulture  ),
                    double.Parse(data[i].dm2, CultureInfo.InvariantCulture),
                    double.Parse(data[i].dm3, CultureInfo.InvariantCulture),
                    double.Parse(data[i].dm4, CultureInfo.InvariantCulture),
                    double.Parse(data[i].dm5, CultureInfo.InvariantCulture),
                    double.Parse(data[i].da1, CultureInfo.InvariantCulture),
                    double.Parse(data[i].da2, CultureInfo.InvariantCulture),
                    double.Parse(data[i].da3, CultureInfo.InvariantCulture),
                    double.Parse(data[i].da4, CultureInfo.InvariantCulture),
                    double.Parse(data[i].da5, CultureInfo.InvariantCulture),
                    double.Parse(data[i].db1, CultureInfo.InvariantCulture),
                    double.Parse(data[i].db2, CultureInfo.InvariantCulture),
                    double.Parse(data[i].db3, CultureInfo.InvariantCulture),
                    double.Parse(data[i].db4, CultureInfo.InvariantCulture),
                    double.Parse(data[i].db5, CultureInfo.InvariantCulture),
                };

                // FILL OUTPUTS ARRAY WITH ANGLE-ACC-ARM VALUES
                outputs[i] = new double[]
                {
                    double.Parse(data[i].ANGLE_ACC_ARM, CultureInfo.InvariantCulture),
                };
            }

            int[] neuronCount = new int[hiddenLayerNumber+1]; // ARRAY THAT STORES N-NUMBER OF HIDDEN LAYERS AND K-NUMBER OF NEURONS PER LAYER

            for (int i=0; i<hiddenLayerNumber;++i)
            {
                neuronCount[i] = neuronPerLayer;

            }
            neuronCount[hiddenLayerNumber] = 1; // TOTAL 1 OUTPUT NEURON

            // CONSTRUCT MLP NEURAL NETWORK
            ActivationNetwork network = new ActivationNetwork(
                new BipolarSigmoidFunction(), // USE SIGMOID ACTIVATION FUNCTION
                32, // INPUT LENGTH (THERE ARE 32 NUMERICAL INPUT ATTRIBUTES)
                neuronCount // CREATE N-NUMBER OF HIDDEN LAYERS WITH K-NUMBER OF NEURONS IN EACH LAYER
                );

            // CONSTRUCT BACK PROPAGATION MECHANISM OF THE NEURAL NETWORK
            BackPropagationLearning teacher = new BackPropagationLearning( network );

            // RUN N-EPOCHS TO TRAIN NEURAL NETWORK WITH GIVEN INPUT AND OUTPUT DATA
            for (int i=0;i<totalEpochs;++i)
            {
                teacher.RunEpoch(trainingInputs, outputs); // TEACH 
            }

            // CALCULATE TRAINING PERFORMANCE SCORES

            // INITIALIZE MAE,MSE,RMSE,R^2 VALUES FOR TRAINING PERFORMANCE EVALUATION
            double trainingMAE = 0;
            double trainingMSE = 0;
            double trainingRMSE = 0;
            double trainingRsquare = 0;

            
            for (int i=0; i<trainingInputs.Count();++i)
            {
                double[] trainingPrediction = network.Compute(trainingInputs[i]); // MAKE PREDICTION FOR TRAINING DATA INPUT ROW N
                trainingMAE += Math.Abs( trainingPrediction[0] - outputs[i][0]); // CALCULATE (PREDICTED RESULT - ACTUAL DATA) AND ADD 
                trainingMSE += Math.Pow( trainingPrediction[0] - outputs[i][0] , 2); // CALCULATE ( PREDICTED RESULT - ACTUAL DATA ) ^ 2 AND ADD
            }

            trainingMAE /= trainingInputs.Count(); // DIVIDE SUM OF (PREDICTED RESULT - ACTUAL DATA) TO DATA LENGTH
            trainingMSE /= trainingInputs.Count(); // DIVIDE SUM OF (PREDICTED RESULT - ACTUAL DATA) ^ 2 TO DATA LENGTH
            trainingRMSE = Math.Sqrt(trainingMSE); // RMSE = SQRT OF MSE
            trainingRsquare = 1 - ( trainingMSE / trainingMAE ); // RSQUARE = 1 - (MSE - MAE)

            // CALCULATE TEST PERFORMANCE SCORES
            List<part1_test> testData = _context.part1_test.ToList(); // GET TEST DATA FROM DATABASE

            double[][] testInputs = new double[testData.Count()][]; // CREATE A 2D ARRAY TO STORE INPUT DATA
            double[][] testOutputs = new double[testData.Count()][]; // CREATE A 2D ARRAY TO STORE OUTPUTS(ANGLE-ACC - ARM) FROM TEST DATA

            // MAKE NECESSARY DATA CONVERTIONS
            for (int i = 0; i < testData.Count(); ++i)
            {
                testInputs[i] = new double[]
                {
                    double.Parse(testData[i].theta1,CultureInfo.InvariantCulture),
                    double.Parse(testData[i].theta2,CultureInfo.InvariantCulture),
                    double.Parse(testData[i].theta3,CultureInfo.InvariantCulture),
                    double.Parse(testData[i].theta4, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].theta5, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].theta6, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].thetad1, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].thetad2, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].thetad3, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].thetad4, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].thetad5, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].thetad6, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].tau1, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].tau2, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].tau3, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].tau4, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].tau5, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].dm1, CultureInfo.InvariantCulture  ),
                    double.Parse(testData[i].dm2, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].dm3, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].dm4, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].dm5, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].da1, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].da2, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].da3, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].da4, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].da5, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].db1, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].db2, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].db3, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].db4, CultureInfo.InvariantCulture),
                    double.Parse(testData[i].db5, CultureInfo.InvariantCulture),
                };

                // STORE OUTPUT (ANGLE-ACC-ARM) VALUES
                testOutputs[i] = new double[]
                {
                    double.Parse(testData[i].ANGLE_ACC_ARM, CultureInfo.InvariantCulture),
                };
            }

            // INITIALIZE MAE,MSE,RMSE AND TEST SQUARE
            double testMAE = 0;
            double testMSE = 0;
            double testRMSE = 0;
            double testRsquare = 0;

            
            for (int i = 0; i < testInputs.Count(); ++i)
            {
                double[] testPrediction = network.Compute(testInputs[i]); // MAKE PREDICTION FOR TEST DATA INPUT ROW N
                testMAE += Math.Abs(testPrediction[0] - testOutputs[i][0]); // CALCULATE (PREDICTED RESULT - ACTUAL DATA) AND ADD 
                testMSE += Math.Pow(testPrediction[0] - testOutputs[i][0], 2); // CALCULATE ( PREDICTED RESULT - ACTUAL DATA ) ^ 2 AND ADD
            }

            testMAE /= testInputs.Count(); // DIVIDE SUM OF (PREDICTED RESULT - ACTUAL DATA) AND ADD TO TEST DATA LENGTH
            testMSE /= testInputs.Count(); // DIVIDE SUM OF (PREDICTED RESULT - ACTUAL DATA) ^ 2 AND ADD TO TEST DATA LENGTH
            testRMSE = Math.Sqrt(testMSE); // RMSE = SQRT OF THE MSE
            testRsquare = 1 - ( testMSE / testMAE ); // R^2 = 1 - (TEST MSE / TEST MAE)

            // RETURN CALCULATED PERFORMANCE SCORES AS FORMATTED STRING
            return
                "MLP Neural Network with " + hiddenLayerNumber + " hidden layers and " + totalEpochs + " Epochs"
                + "\n\n" +
                "Train Results:\n" +
                "Training Data Count: " + trainingInputs.Count() + "\n" +
                "MAE: " + trainingMAE + "\n" +
                "MSE: " + trainingMSE + "\n" +
                "RMSE: " + trainingRMSE + "\n" +
                "R^2 (coefficient of determination): " + trainingRsquare + "\n" +
                "\n\n" +
                "Test Results: \n" +
                "Test Data Count: " + testInputs.Count() + "\n" +
                "MAE: " + testMAE + "\n" +
                "MSE: " + testMSE + "\n" +
                "RMSE: " + testRMSE + "\n" +
                "R^2 (coefficient of determination): " + testRsquare + "\n\n" +
                "NUMBER OF HIDDEN LAYERS:"+" "+ (network.Layers.Count()-1) + "\n" +
                "TOTAL NUMBER OF LAYERS:"+" "+ network.Layers.Count()+" \n" +
                "ACTIVATION FUNCTION:"+" " + "BIPOLAR SIGMOID ACTIVATION FUNCTION" +"\n"+
                "INITIAL LEARNING RATE VALUE: 0.1"+ "\n" +
                "NUMBER OF EPOCHS:"+" "+totalEpochs + "\n\n" +
                "NEURON WEIGHTS" + "\n\n\n" +
                string.Join("\n\n\n", GetNetworkReports(network));
        }
    }
}
