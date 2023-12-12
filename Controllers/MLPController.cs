using Accord.Neuro;
using Accord.Neuro.Learning;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using Microsoft.ML.Data;
using NeuralNetworks.Data;
using System.Globalization;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;


namespace NeuralNetworks.Controllers
{
    [ApiController]
    [Route("/MLPController")]
    public class MLPController : ControllerBase
    {

        private DatabaseContext _context;

        public MLPController(DatabaseContext context)
        {
             this._context = context;
        }

        [HttpGet("createMLPNeuralNetwork")]
        public string createMLPNeuralNetwork(int hiddenLayerNumber,int epochNumber)
        {
           

            List<part1_train> data = _context.part1_train.ToList();

            double[][] trainingInputs = new double[data.Count()][];
            double[][] outputs = new double[data.Count()][];

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

                outputs[i] = new double[]
                {
                    double.Parse(data[i].ANGLE_ACC_ARM, CultureInfo.InvariantCulture),


                };

            }
            
            ActivationNetwork network = new ActivationNetwork(
                new BipolarSigmoidFunction(),
                32, // input length
                hiddenLayerNumber, // hidden layer num
                1 // output length
                );

            BackPropagationLearning teacher = new BackPropagationLearning( network );

            for (int i=0;i<epochNumber;++i)
            {
                teacher.RunEpoch(trainingInputs, outputs); // TEACH MLP 

            }
           

            //double[] test = { 0.4,0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8 };
            //double[] prediction = network.Compute(test);

            // TRAINING PERFORMANCE SCORES

            double trainingMAE = 0;
            double trainingMSE = 0;
            double trainingRMSE = 0;
            double trainingRsquare = 0;


            for (int i=0; i<trainingInputs.Count();++i)
            {
                double[] trainingPrediction = network.Compute(trainingInputs[i]);
                trainingMAE += Math.Abs( trainingPrediction[0] - outputs[i][0]);
                trainingMSE += Math.Pow( trainingPrediction[0] - outputs[i][0] , 2);
            }

            trainingMAE /= trainingInputs.Count();
            trainingMSE /= trainingInputs.Count();
            trainingRMSE = Math.Sqrt(trainingMSE);
            trainingRsquare = 1 - ( trainingMSE / trainingMAE );



            // TEST PERFORMANCE SCORES
            List<part1_test> testData = _context.part1_test.ToList();

            double[][] testInputs = new double[testData.Count()][];
            double[][] testOutputs = new double[testData.Count()][];

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

                testOutputs[i] = new double[]
                {
                    double.Parse(testData[i].ANGLE_ACC_ARM, CultureInfo.InvariantCulture),


                };

            }


            double testMAE = 0;
            double testMSE = 0;
            double testRMSE = 0;
            double testRsquare = 0;

            
            for (int i = 0; i < testInputs.Count(); ++i)
            {
                double[] testPrediction = network.Compute(testInputs[i]);
                testMAE += Math.Abs(testPrediction[0] - testOutputs[i][0]);
                testMSE += Math.Pow(testPrediction[0] - testOutputs[i][0], 2);
            }

            testMAE /= testInputs.Count();
            testMSE /= testInputs.Count();
            testRMSE = Math.Sqrt(testMSE);
            testRsquare = 1 - ( testMSE / testMAE );




            return 
                "MLP Neural Network with " + hiddenLayerNumber + " hidden layers and " + epochNumber+" Epochs"
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
                "R^2 (coefficient of determination): " + testRsquare;




        }


    }
}
