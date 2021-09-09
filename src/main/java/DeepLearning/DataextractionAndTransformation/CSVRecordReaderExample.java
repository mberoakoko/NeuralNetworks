package DeepLearning.DataextractionAndTransformation;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.local.transforms.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
public class CSVRecordReaderExample {
    public static void main(String[] args) throws IOException, InterruptedException {
        final int numClasses = 2;
        final int batchSize = 8;

        Path filePathRaw = Paths.get("src/main/java/DeepLearning/DataextractionAndTransformation/titanic.csv");
        File file = new File(filePathRaw.toAbsolutePath().toString());
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(file));

        Schema schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("PClass", Arrays.asList("1", "2", "3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex", Arrays.asList("male", "female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("fare")
                .build();

        TransformProcess transformProcess  = new TransformProcess.Builder(schema)
                .removeColumns("Name", "fare")
                .categoricalToInteger("Sex")
                .categoricalToOneHot("PClass")
                .removeColumns("PClass[1]")
                .build();
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
        //DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(transformProcessRecordReader,writableConverter,8,1,7,2,-1,true);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(transformProcessRecordReader,batchSize)
                .classification(0,numClasses)
                .build();
        System.out.println("Total number of possible labels = [" + dataSetIterator.totalOutcomes()+ "]");
        DataAnalysis analysis = AnalyzeLocal.analyze(schema, recordReader);
        System.out.println(analysis);
    }
}
