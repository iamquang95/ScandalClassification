# python GenPureData.py 5000 14900 UnTokData1.txt
# cd vntokenizer
# ./vnTokenizer.sh -i ../UnTokData1.txt -o ../TokedData1.txt
# cd ..

python GenPureData.py 15001 20000 UnTokData2.txt
cd vntokenizer
./vnTokenizer.sh -i ../UnTokData2.txt -o ../TokedData2.txt
cd ..

python GenPureData.py 20001 25000 UnTokData3.txt
cd vntokenizer
./vnTokenizer.sh -i ../UnTokData3.txt -o ../TokedData3.txt
cd ..

python GenPureData.py 25001 30000 UnTokData4.txt
cd vntokenizer
./vnTokenizer.sh -i ../UnTokData4.txt -o ../TokedData4.txt
cd ..

