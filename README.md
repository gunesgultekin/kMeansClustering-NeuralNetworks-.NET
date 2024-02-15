# K-Means Clustering & Neural Networks in .NET For a specific dataset
* This project is the practical application of the K-Means clustering algorithm and multilayer artificial neural networks on .NET Core 7 that I learned in the "Machine Learning" course.
* Microsoft ML.NET library was used for the K-Means algorithm, and Accord.Neuro library was used to create the Artificial Neural network model.
# Functions:
## /createKMeans
 * Runs the algorithm according to the desired "k" (total number of clusters), calculates the predicted cluster for each data record, and overall performance scores (WCSS, BCSS, Dunn Index)
 * Example response for k = 35 (35 clusters):
   
![kMean-clusters](https://github.com/gunesgultekin/kMeansClustering-NeuralNetworks-.NET/assets/126399958/29842e09-d6e6-416f-9652-86c83fa43934)
![kMean-cluster-ids](https://github.com/gunesgultekin/kMeansClustering-NeuralNetworks-.NET/assets/126399958/2cf24d42-4b56-4a99-a22b-1c0dd8158286)
![kMean-perf-scores](https://github.com/gunesgultekin/kMeansClustering-NeuralNetworks-.NET/assets/126399958/f19a10d6-a687-4c3b-a5bb-739f163ceab9)

## /createMLPNeuralNetwork
* Creates a Neural Network model with desired number of hidden layers and neurons per hidden layer.
* Trains the model with specific data within the project for a given n number of epochs.
* Calculates performance scores ( MAE,MSE,RMSE,R-Square ) seperately for Training and Test phases
* Reports the weight values in each neuron
* Example response for a Neural Network with 3 Hidden Layers, 5 Neurons per Hidden Layer and 50 Training Epochs:
  
![nn-performance-scores](https://github.com/gunesgultekin/kMeansClustering-NeuralNetworks-.NET/assets/126399958/ef3cb4f9-5c64-496a-9c8a-874ffc95ef92)
![nn-neuron-weights](https://github.com/gunesgultekin/kMeansClustering-NeuralNetworks-.NET/assets/126399958/1ec9e742-036a-448b-a3ea-fd449265d78e)
