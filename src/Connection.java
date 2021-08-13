public class Connection {
    final Neuron leftNeuron;
    final Neuron rightNeuron;
    double weight;
    double prevDeltaWeight;
    double deltaWeight;
    static int counter = 0;
    final public int id;

    public Connection(Neuron fromN, Neuron toN)
    {
        leftNeuron = fromN;
        rightNeuron = toN;
        id = counter;
        counter++;
    }

    public double getPrevDeltaWeight() {
        return prevDeltaWeight;
    }

    public double getWeight() {
        return weight;
    }

    public Neuron getLeftNeuron() {
        return leftNeuron;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public void setDeltaWeight(double deltaWeight) {
        this.deltaWeight = deltaWeight;
    }
}
