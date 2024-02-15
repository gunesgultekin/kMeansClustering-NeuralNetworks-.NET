using Microsoft.ML.Data;

namespace NeuralNetworks.Data
{
    // CLASS FOR CLUSTER PREDICTIONS (K-MEANS)
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[]? Distances;
    }
}
