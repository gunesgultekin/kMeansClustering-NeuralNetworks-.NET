using Microsoft.AspNetCore.Mvc;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.IO;
using System.Text;
using Microsoft.ML;
using NeuralNetworks.Data;
using System.Data;
using Microsoft.ML.Data;
using System.Collections;


namespace NeuralNetworks.Controllers
{
    [ApiController()]
    [Route("/kMeanController")]
    public class kMeanController : ControllerBase
    {
        [HttpGet("createKMeans")]
        public string createKMeans(int k_Value)
        {
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "part2.csv");
            string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "KMeansClusteringModel");

            var mlContext = new MLContext(seed: 0);

            IDataView dataView = mlContext.Data.LoadFromTextFile<part2Data>(_dataPath, hasHeader: true, separatorChar: ',');

            string featuresColumnName = "Features";

          

            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: k_Value));

            var model = pipeline.Fit(dataView);

            


            // CREATE A BUFFER TO STORE CENTROID COORDINATES OF THE CLUSTERS
            VBuffer<float>[] centroids = default;

            // GET CLUSTER CENTROID COODINATES OF THE TRAINED K-MEANS MODEL
            // THERE WILL BE 9 CENTROIDS FOR EACH CLUSTER DUE TO THERE ARE 9 ATTRIBUTES IN OUR DATASET (9-DIMENSIONS)

            double BCSS = 0;

            double temp = 0;

            model.LastTransformer.Model.GetClusterCentroids(ref centroids, out int k);

            for (int i = 0; i < centroids.Length - 1; ++i) // CLUSTER[0] to CLUSTER [N-1]
            {
                for (int j = 1; j < centroids.Length; ++j) // CLUSTER[1] TO CLUSTER [N]
                {
                    for (int m = 0; m < 9; ++m) // COORDINATE[0] TO COORDINATE [9] 
                    {
                        temp += getDistance( (centroids[i].GetValues()[m]) , (centroids[j].GetValues()[m])  );
                        // EX: GET THE DISTANCE BETWEEN CLUSTER-0 [X-COORD] AND CLUSTER-1 [X-COORD] 
                        // THEN CONTINUE WITH CLUSTER-0 [Y-COORD] AND CLUSTER-1 [Y-COORD]
                        // ADD CALCULATED DISTANCES TO BCSS VARIABLE


                    }
                    // AFTER ALL THE DISTANCE (TOTAL 9 CALCULATIONS) CALCULATIONS BETWEEN CLUSTER[0] AND CLUSTER[1] DONE
                    temp = Math.Sqrt( temp ); // TAKE THE SQUARE ROOT OF THE ALL SUMMED DISTANCES
                    BCSS += temp; // ADD SQUARE ROOTED FIRST ITERATION TO BCSS
                    

                }

            }
            

            var predictor = mlContext.Model.CreatePredictionEngine<part2Data, ClusterPrediction>(model);

            var preview = dataView.Preview(maxRows: 500);

            List<string> predictionResults = new List<string>();

            int counter = 1; // COUNTER FOR ROW COUNT START FROM 1

            Dictionary<uint,int> recordCounter = new Dictionary<uint, int>();

            for (uint i=0; i< k_Value;++i)
            {
                recordCounter.Add(i,0); // HASHTABLE ( KEY = CLUSTER ID , INITIAL VALUE = 0 ) 
            }
            
           

            double WCSS = 0;
            

            foreach (var row in preview.RowView)
            {
                var currentRow = row.Values.ToArray();

                var currentDataRow = new part2Data()
                {
                    a1 = float.Parse(currentRow[0].Value.ToString()),
                    a2 = float.Parse(currentRow[1].Value.ToString()),
                    a3 = float.Parse(currentRow[2].Value.ToString()),
                    a4 = float.Parse(currentRow[3].Value.ToString()),
                    a5 = float.Parse(currentRow[4].Value.ToString()),
                    a6 = float.Parse(currentRow[5].Value.ToString()),
                    a7 = float.Parse(currentRow[6].Value.ToString()),
                    a8 = float.Parse(currentRow[7].Value.ToString()),
                    a9 = float.Parse(currentRow[8].Value.ToString()),

                };


                var prediction = predictor.Predict(currentDataRow);

                recordCounter[prediction.PredictedClusterId - 1] += 1;
                

                predictionResults.Add(
                    "Record " + counter + "          " + "Cluster: " + prediction.PredictedClusterId
                    ); 

                WCSS += Math.Pow((prediction.Distances[prediction.PredictedClusterId - 1]), 2); 


                counter++;



            }

            predictionResults.Add(" ");

            predictionResults.Add("Cluster ID , Total Records");

            predictionResults.Add( string.Join("\n",recordCounter) );


           
            predictionResults.Add(" ");

            predictionResults.Add(
                "WCSS:     " +  WCSS
                );

            predictionResults.Add(" ");

            predictionResults.Add(
                "BCSS:     " + BCSS
                );

            predictionResults.Add(" ");

            predictionResults.Add(
                "Dunn Index: " + BCSS / WCSS
                );

            

            return string.Join("\n", predictionResults);


           
             
        }

        [HttpGet("getDistance")]
        public double getDistance(double coord_1, double coord_2)
        {
            return Math.Pow((coord_1 - coord_2), 2);

        }


    }


}
