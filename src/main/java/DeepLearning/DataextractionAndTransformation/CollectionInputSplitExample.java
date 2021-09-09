package DeepLearning.DataextractionAndTransformation;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CollectionInputSplitExample {
    public static void main(String[] args) {
        Path path = Paths.get("src/main/java/DeepLearning/DataextractionAndTransformation/temp");
        FileSplit split = new FileSplit(new File(path.toAbsolutePath().toString()));
        CollectionInputSplit collectionInputSplit = new CollectionInputSplit(split.locations());
        collectionInputSplit.locationsIterator().forEachRemaining(System.out::println);
    }
}
