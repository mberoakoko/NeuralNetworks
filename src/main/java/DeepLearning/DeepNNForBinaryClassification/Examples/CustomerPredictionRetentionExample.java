package DeepLearning.DeepNNForBinaryClassification.Examples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class CustomerPredictionRetentionExample {
    //private static final Logger log = LoggerFactory.getLogger(CustomerPredictionRetentionExample.class.toGenericString());

    private static Schema generateSchema(){
        return new Schema.Builder()
                .addColumnsInteger("RowNumber", "CustomerID")
                .addColumnString("Surname")
                .addColumnsInteger("CreditScore")
                .addColumnCategorical("Geography", Arrays.asList("France", "Germany", "Spain"))
                .addColumnCategorical("Gender", Arrays.asList("Male", "Female"))
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnsInteger("NumOfProducts", "HasCrCard", "IsActiveMember")
                .addColumnDouble("EstimatedSalary")
                .addColumnsInteger("Exited")
                .build();
    }

    private static RecordReader applyTransform(RecordReader recordReader, Schema schema){
        return new TransformProcessRecordReader(
                recordReader,
                new TransformProcess.Builder(schema)
                        .removeColumns("RowNumber", "CustomerID", "Surname")
                        .categoricalToInteger("Gender")
                        .categoricalToOneHot("Geography")
                        .removeColumns("Geography[France]")
                        .build()
                );
    }

    public static RecordReader generateRecordReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(file));
        return CustomerPredictionRetentionExample.applyTransform(recordReader, CustomerPredictionRetentionExample.generateSchema());
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        final int labelIndex    = 11;
        final int batchSize     = 8;
        final int numClasses    = 2;
        final INDArray weightsArray = Nd4j.create(new double[]{0.53, 0.75});

        System.out.println(CustomerPredictionRetentionExample.class);
        Schema mySchema = generateSchema();
        Path pathToFile = Paths.get("D:\\Processing Projects\\NeuralNetworks\\src\\main\\resources\\Chapter3\\Churn_Modelling.csv");
        File csvModelingFile = new File(pathToFile.toAbsolutePath().toString());
        RecordReader fileRecordReader = CustomerPredictionRetentionExample.generateRecordReader(csvModelingFile);;

        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(fileRecordReader, batchSize)
                                                                                .classification(labelIndex, numClasses)
                                                                                .build();
        final DataNormalization normalization = new NormalizerStandardize();
        normalization.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalization);
        final DataSetIteratorSplitter dataSetIteratorSplitter = new DataSetIteratorSplitter(dataSetIterator,1250,0.8);

        //log.info("Building this stuff ------------------------------------------>");

        final MultiLayerConfiguration config =
                new NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.RELU_UNIFORM)
                        .updater(new Adam(0.015D))
                        .list()
                        .layer(new DenseLayer.Builder().nIn(11).nOut(6).activation(Activation.RELU).dropOut(0.9).build())
                        .layer(new DenseLayer.Builder().nIn(6).nOut(6).activation(Activation.RELU).dropOut(0.9).build())
                        .layer(new DenseLayer.Builder().nIn(6).nOut(4).activation(Activation.RELU).dropOut(0.9).build())
                        .layer(new OutputLayer.Builder(new LossMCXENT(weightsArray)).nIn(4).nOut(2).activation(Activation.SOFTMAX).build())
                        .build();

        final UIServer uiServer = UIServer.getInstance();
        final StatsStorage statsStorage = new InMemoryStatsStorage();

        final MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(config);
        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new ScoreIterationListener(100), new StatsListener(statsStorage));

        uiServer.attach(statsStorage);
        multiLayerNetwork.fit(dataSetIteratorSplitter.getTrainIterator(), 100);

        final Evaluation eval = multiLayerNetwork.evaluate(dataSetIteratorSplitter.getTestIterator(), Arrays.asList("0", "1"));
        System.out.println(eval.stats());

        final File file = new File("model.zip");
        ModelSerializer.writeModel(multiLayerNetwork, file, true);
        ModelSerializer.addNormalizerToModel(file, normalization);
    }
}
