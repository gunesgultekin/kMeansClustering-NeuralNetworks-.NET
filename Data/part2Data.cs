using Microsoft.ML.Data;

namespace NeuralNetworks.Data
{
    // PART 2 DATA MAPPER FOR K-MEANS MODEL
    public class part2Data
    {
        [LoadColumn(0)]
        public float a1 { get; set; }

        [LoadColumn(1)]
        public float a2 { get; set; }
        [LoadColumn(2)]
        public float a3 { get; set; }
        [LoadColumn(3)]
        public float a4 { get; set; }
        [LoadColumn(4)]
        public float a5 { get; set; }
        [LoadColumn(5)]
        public float a6 { get; set; }
        [LoadColumn(6)]
        public float a7 { get; set; }
        [LoadColumn(7)]
        public float a8 { get; set; }
        [LoadColumn (8)]
        public float a9 { get; set; }
    }
}
