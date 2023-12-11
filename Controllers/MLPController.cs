using Microsoft.AspNetCore.Mvc;

using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;
using static NeuralNetworks.MLP;

namespace NeuralNetworks.Controllers
{
    [ApiController]
    [Route("/MLPController")]
    public class MLPController : ControllerBase
    {


        [HttpGet("Predict")]
        public string predict(string inputData)
        {
            List<string> list;

            string[] elements = inputData.Split(",");

            ModelInput input = new ModelInput();

            input.Theta1 = float.Parse(elements[0], CultureInfo.InvariantCulture);
            input.Theta2 = float.Parse(elements[1], CultureInfo.InvariantCulture);
            input.Theta3 = float.Parse(elements[2], CultureInfo.InvariantCulture);
            input.Theta4 = float.Parse(elements[3], CultureInfo.InvariantCulture);
            input.Theta5 = float.Parse(elements[4], CultureInfo.InvariantCulture);
            input.Theta6 = float.Parse(elements[5], CultureInfo.InvariantCulture); 
            
            input.Thetad1= float.Parse(elements[6], CultureInfo.InvariantCulture);
            input.Thetad2 = float.Parse(elements[7], CultureInfo.InvariantCulture);
            input.Thetad3 = float.Parse(elements[8], CultureInfo.InvariantCulture);
            input.Thetad4 = float.Parse(elements[9], CultureInfo.InvariantCulture);
            input.Thetad5 = float.Parse(elements[10], CultureInfo.InvariantCulture);
            input.Thetad6 = float.Parse(elements[11], CultureInfo.InvariantCulture);

            input.Tau1 = float.Parse(elements[12], CultureInfo.InvariantCulture);
            input.Tau2 = float.Parse(elements[13], CultureInfo.InvariantCulture);
            input.Tau3 = float.Parse(elements[14], CultureInfo.InvariantCulture);
            input.Tau4 = float.Parse(elements[15], CultureInfo.InvariantCulture);
            input.Tau5 = float.Parse(elements[16], CultureInfo.InvariantCulture);

            input.Dm1 = float.Parse(elements[17], CultureInfo.InvariantCulture);
            input.Dm2 = float.Parse(elements[18], CultureInfo.InvariantCulture);
            input.Dm3 = float.Parse(elements[19], CultureInfo.InvariantCulture);
            input.Dm4 = float.Parse(elements[20], CultureInfo.InvariantCulture);
            input.Dm5 = float.Parse(elements[21], CultureInfo.InvariantCulture);

            input.Da1 = float.Parse(elements[22], CultureInfo.InvariantCulture);
            input.Da2 = float.Parse(elements[23], CultureInfo.InvariantCulture);
            input.Da3 = float.Parse(elements[24], CultureInfo.InvariantCulture);
            input.Da4 = float.Parse(elements[25], CultureInfo.InvariantCulture);
            input.Da5 = float.Parse(elements[26], CultureInfo.InvariantCulture);

            input.Db1 = float.Parse(elements[27], CultureInfo.InvariantCulture);
            input.Db2 = float.Parse(elements[28], CultureInfo.InvariantCulture);
            input.Db3 = float.Parse(elements[29], CultureInfo.InvariantCulture);
            input.Db4 = float.Parse(elements[30], CultureInfo.InvariantCulture);
            input.Db5 = float.Parse(elements[31], CultureInfo.InvariantCulture);


            ModelOutput ouput =  MLP.Predict(input);

            return "Predicted Value: "+ouput.Score.ToString();
        }

       
    }
}
