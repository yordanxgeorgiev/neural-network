import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {
    final ArrayList<Neuron> inputLayer = new ArrayList<>();
    final ArrayList<Neuron> hiddenLayer = new ArrayList<>();
    final ArrayList<Neuron> outputLayer = new ArrayList<>();
    final Neuron bias = new Neuron();
    final int[] layers;
    final Random rand = new Random();
    final int randomWeightMultiplier = 1;
    final double learningRate = 0.9f;
    final double momentum = 0.7f;

    // inputs
    final double[][] inputs ={{1,1},{1,0},{0,1},{0,0}};
    // outputs
    final double[][] XOR_expectedOutputs ={{0},{1},{1},{0}};
    final double[][] AND_expectedOutputs = {{1},{0},{0},{0}};
    final double[][] OR_expectedOutputs = {{1},{1},{1},{0}};

    double[][] resultOutputs = {{-1},{-1},{-1},{-1}};
    double[] output;

    public NeuralNetwork(int input, int hidden, int output)
    {
        this.layers = new int[]{input,hidden,output};

        for(int i = 0; i < layers.length; i++)
        {
            if(i==0) // input layer
            {
                for(int j = 0; j < layers[i]; j++)
                {
                    Neuron neuron = new Neuron();
                    inputLayer.add(neuron);
                }
            }
            else if(i==1) // hidden layer
            {
                for(int j = 0 ; j < layers[i]; j++)
                {
                    Neuron neuron = new Neuron();
                    neuron.addInConnections(inputLayer);
                    neuron.addBiasConnection(bias);
                    hiddenLayer.add(neuron);
                }
            }
            else
            {
                for(int j = 0; j < layers[i]; j++)
                {
                    Neuron neuron = new Neuron();
                    neuron.addInConnections(hiddenLayer);
                    neuron.addBiasConnection(bias);
                    outputLayer.add(neuron);
                }
            }
        }
        // initialize random weights
        for(Neuron neuron : hiddenLayer)
        {
            ArrayList<Connection> connections = neuron.getInConnections();
            for(Connection con: connections)
            {
                double newWeight = getRandom();
                con.setWeight(newWeight);
            }
        }
        for(Neuron neuron : outputLayer)
        {
            ArrayList<Connection> connections = neuron.getInConnections();
            for(Connection con : connections)
            {
                double newWeight = getRandom();
                con.setWeight(newWeight);
            }
        }

        // reset id counters
        Neuron.counter = 0;
        Connection.counter = 0;
    }

    double getRandom()
    {
        return randomWeightMultiplier * (rand.nextDouble()*2-1); // from -1 to 1
    }

    public void setInput(double[] inputs)
    {
        for(int i = 0; i < inputLayer.size();i++)
        {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }

    public double[] getOutput()
    {
        double[] outputs = new double[outputLayer.size()];
        for(int i = 0; i < outputLayer.size(); i++)
        {
            outputs[i] = outputLayer.get(i).getOutput();
        }
        return outputs;
    }

    public void activate()
    {
        for(Neuron n: hiddenLayer)
        {
            n.calculateOutput();
        }
        for(Neuron n: outputLayer)
        {
            n.calculateOutput();
        }
    }

    public void backpropagation(double[] expectedOutput)
    {
        int i = 0;
        for(Neuron n: outputLayer)
        {
            ArrayList<Connection> connections = n.getInConnections();
            for(Connection con: connections)
            {
                double ak = n.getOutput();
                double ai = con.leftNeuron.getOutput();
                double desiredOutput = expectedOutput[i];

                double partialDerivative = -ak * (1-ak) * ai * (desiredOutput-ak);
                double deltaWeight = -learningRate * partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum*con.getPrevDeltaWeight());
            }
            i++;
        }

        // update weights for hidden layer
        for(Neuron n:hiddenLayer)
        {
            ArrayList<Connection> connections = n.getInConnections();
            for(Connection con : connections)
            {
                double aj = n.getOutput();
                double ai = con.leftNeuron.getOutput();
                double sumKoutputs = 0;
                int j = 0;
                for(Neuron out_neu : outputLayer)
                {
                    double wjk = out_neu.getConnection(n.id).getWeight();
                    double desiredOutput = expectedOutput[j];
                    double ak = out_neu.getOutput();
                    j++;
                    sumKoutputs += (-(desiredOutput-ak)*ak*(1-ak)*wjk);
                }

                double partialDerivative = aj*(1-aj)*ai*sumKoutputs;
                double deltaWeight = -learningRate*partialDerivative;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight + momentum*con.getPrevDeltaWeight());
            }
        }
    }
    void run(int maxSteps, double minError, String operation)
    {
        double[][] expectedOutputs = switch (operation) {
            case "XOR" -> XOR_expectedOutputs;
            case "AND" -> AND_expectedOutputs;
            case "OR" -> OR_expectedOutputs;
            default -> new double[4][1];
        };

        int i;
        // train neural network until minError reached or maxSteps exceeded
        double error = 1;

        for(i = 0; i < maxSteps && error > minError; i++)
        {
            error = 0;
            for(int p = 0; p < inputs.length; p++)
            {
                setInput(inputs[p]);
                activate();
                output = getOutput();
                resultOutputs[p] = output;

                for(int j = 0; j < expectedOutputs[p].length; j++)
                {
                    double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
                    error += err;
                }
                backpropagation(expectedOutputs[p]);
            }
        }

        printResult(operation, expectedOutputs);
        System.out.println("Sum of squared errors: " + error);
        System.out.println("Epoch No: " + i + "\n");
        if(i == maxSteps)
        {
            System.out.println("Error training");
        }
    }

    void printResult(String operation, double[][] expectedOutputs)
    {

        System.out.println("Neural network for " + operation + " operation");
        for(int p = 0; p < inputs.length; p++)
        {
            System.out.print("Inputs: ");
            for(int x = 0; x < layers[0]; x++)
            {
                System.out.print(inputs[p][x] + " ");
            }
            System.out.print("Expected: ");
            for(int x = 0; x < layers[2]; x++)
            {
                System.out.print(expectedOutputs[p][x] + " ");
            }
            System.out.print("Actual: ");
            for(int x = 0; x < layers[2]; x++)
            {
                System.out.print(resultOutputs[p][x] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void main(String[] args) {
        int maxRuns = 50000;
        double minError = 0.001;

        NeuralNetwork xor_nn = new NeuralNetwork(2,4,1);
        NeuralNetwork and_nn = new NeuralNetwork(2,4,1);
        NeuralNetwork or_nn = new NeuralNetwork(2,4,1);

        xor_nn.run(maxRuns, minError, "XOR");
        or_nn.run(maxRuns, minError, "OR");
        and_nn.run(maxRuns, minError, "AND");
    }
}