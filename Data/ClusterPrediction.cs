using Microsoft.ML.Data;

namespace NeuralNetworks.Data
{
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[]? Distances;


    }
}
