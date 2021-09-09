package ND4jBasics;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.ops.transforms.Transforms.*;
public class Arrays {
    public static void main(String[] args) {
        int rows = 2, columns = 2;
        INDArray zeros = Nd4j.zeros(rows, columns);
        System.out.println(zeros);
        INDArray ones = Nd4j.ones(rows , columns);
        INDArray combined = Nd4j.concat(0, zeros, ones);
        INDArray combined2 = Nd4j.concat(1, zeros, ones);
        System.out.println(ones);
        System.out.println(combined);
        System.out.println(combined2);
        double [] entries = new double[]{1, 3, 4};
        INDArray rowVector = Nd4j.create(entries);
        INDArray columnVector = rowVector.reshape(entries.length, 1);
        System.out.println("Column Vector");
        System.out.println(columnVector);
        System.out.println("Row vector");
        System.out.println(rowVector);
        INDArray matrix = Nd4j.diag(rowVector);
        System.out.println("matrix");
        System.out.println(matrix);
        System.out.println(matrix.mul(columnVector));
        System.out.println(matrix.mmul(columnVector));
        INDArray xSpace = Nd4j.linspace(0, 20, 1000);
        INDArray output = cos(xSpace);
    }
}
