# python PreProcessingData.py
# cd vntokenizer
# ./vnTokenizer.sh -i ../output.txt -o ../data_to_split.txt
# cd ..
# python GenerateNGram.py
python SplitData.py
# python RemoveStopWords.py
cp test.tagged jmdn-maxent2/models/sample
cp train.tagged jmdn-maxent2/models/sample
cd jmdn-maxent2/libs
java -classpath jmdn-maxent.jar:jmdn-base.jar:args4j-2.0.6.jar -Dfile.encoding=UTF8 jmdn.method.classification.maxent.Trainer -all -d ../models/sample/
cd ../..


# python RemoveStopWords.py
# cp test.tagged2 jmdn-maxent2/models/sample
# cp train.tagged2 jmdn-maxent2/models/sample
# cd jmdn-maxent2/libs
# java -classpath jmdn-maxent.jar:jmdn-base.jar:args4j-2.0.6.jar -Dfile.encoding=UTF8 jmdn.method.classification.maxent.Trainer -all -d ../models/sample/
# cd ../..

