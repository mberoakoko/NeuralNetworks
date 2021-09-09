package DeepLearning.DataextractionAndTransformation;

import org.datavec.api.split.FileSplit;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

public class FileSplitExample {

    public static void main(String[] args) {
        String [] allowedFormats = new String[]{".JPEG"};
        Path currentRelativePath = Paths.get("src/main/java/DeepLearning/DataextractionAndTransformation/temp");
        String s = currentRelativePath.toAbsolutePath().toString();;
        FileSplit splitter = new FileSplit(new File(s), allowedFormats, true);
        splitter.locationsIterator().forEachRemaining(System.out::println);
    }
}
