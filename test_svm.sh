java -cp liblinear.jar de.bwaldvogel.liblinear.Train -s 0 -c 0.00001 train.txt svmModel
java -cp liblinear.jar de.bwaldvogel.liblinear.Predict -b 1 test.txt svmModel out

rm -rf svmModel
rm -rf out
java -cp liblinear.jar de.bwaldvogel.liblinear.Train -s 6 -t 0 -c 0.00001 train.txt svmModel
java -cp liblinear.jar de.bwaldvogel.liblinear.Predict -b 1 test.txt svmModel out

rm -rf svmModel
rm -rf out
./svm-train -s 0 -t 3 -c 0.00001 ../train.txt svmModel
./svm-predict ../test.txt svmModel out

rm -rf svmModel
rm -rf out
./svm-train -s 0 -t 2 -c 0.00001 -h 0 -g 0.0000001 -wi 0.5 ../train.txt svmModel
./svm-predict ../test.txt svmModel out
