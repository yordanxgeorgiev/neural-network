import java.util.ArrayList;
import java.util.HashMap;

public class Neuron {
    static int counter = 0;
    final public int id;
    Connection biasConnection;
    final double bias = -1;
    double output;

    ArrayList<Connection> inConnections = new ArrayList<>();
    HashMap<Integer, Connection> connectionLookup = new HashMap<>();

    public Neuron()
    {
        id = counter;
        counter++;
    }

    // calculate input function Sj = Wij*Aij + w0j*bias
    public void calculateOutput()
    {
        double s = 0;

        for(Connection con : inConnections)
        {
            Neuron leftNeuron = con.getLeftNeuron();
            double weight = con.getWeight();
            double a = leftNeuron.getOutput();

            s = s+weight*a;
        }

        s = s + biasConnection.getWeight()*bias;

        output = g(s); // activation function
    }

    double g(double x)
    {
        return sigmoid(x);
    }

    double sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void addInConnections(ArrayList<Neuron> inNeurons)
    {
        for(Neuron n : inNeurons)
        {
            Connection con = new Connection(n, this);
            inConnections.add(con);
            connectionLookup.put(n.id, con);
        }
    }

    public Connection getConnection(int neuronIndex)
    {
        return connectionLookup.get(neuronIndex);
    }

    public void addBiasConnection(Neuron n)
    {
        Connection con = new Connection(n, this);
        biasConnection = con;
        inConnections.add(con);
    }

    public ArrayList<Connection> getInConnections() {
        return inConnections;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}
