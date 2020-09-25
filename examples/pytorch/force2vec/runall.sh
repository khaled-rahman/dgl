#node-v17

module load gcc

ITERATIONS=(1)
BATCHES=(384 768 1152)
DIMS=(16 32 64 128)

blogdir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/blogcatalog.mtx"
flickrdir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/flickr.mtx"
youtubedir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/youtube.mtx"
hollywood="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/special/hollywood-2009/hollywood-2009.mtx"
amazon="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/special/com-Amazon/com-Amazon.mtx"
orkut="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/orkut.mtx"

#<< COMMENT
for iter in "${ITERATIONS[@]}"
do
	for b in "${BATCHES[@]}"
	do
		for d in "${DIMS[@]}"
		do
			echo "Param:iter="${it},"dim="${d},"batch="${b},"type=1" >> coraresults.txt
			python force2vec.py -g cora -it ${iter} -d ${d} -t 1 -b ${b} >> coraresults.txt
			echo "Param:iter="${it},"dim="${d},"batch="${b},"type=2" >> coraresults.txt
                        python force2vec.py -g cora -it ${iter} -d ${d} -t 2 -b ${b} >> coraresults.txt

			echo "Param:iter="${it},"dim="${d},"batch="${b},"type=1" >> citeseerresults.txt
                        python force2vec.py -g citeseer -it ${iter} -d ${d} -t 1 -b ${b} >> citeseerresults.txt
                        echo "Param:iter="${it},"dim="${d},"batch="${b},"type=2" >> citeseerresults.txt
                        python force2vec.py -g citeseer -it ${iter} -d ${d} -t 2 -b ${b} >> citeseerresults.txt
		
			echo "Param:iter="${it},"dim="${d},"batch="${b},"type=1" >> pubmedresults.txt
                        python force2vec.py -g pubmed -it ${iter} -d ${d} -t 1 -b ${b} >> pubmedresults.txt
                        echo "Param:iter="${it},"dim="${d},"batch="${b},"type=2" >> pubmedresults.txt
                        python force2vec.py -g pubmed -it ${iter} -d ${d} -t 2 -b ${b} >> pubmedresults.txt
		
		done
	done
done
#COMMENT

echo "blogcatalog" >> biggraphresults.txt
python force2vec.py -p ${blogdir} -it 1 -d 128 -t 1 >> biggraphresults.txt
echo "flickr" >> biggraphresults.txt
python force2vec.py -p ${flickrdir} -it 1 -d 128 -t 1 >> biggraphresults.txt
echo "amazon" >> biggraphresults.txt
python force2vec.py -p ${amazon} -it 1 -d 128 -t 1 >> biggraphresults.txt
echo "youtube" >> biggraphresults.txt
python force2vec.py -p ${youtubedir} -it 1 -d 128 -t 1 >> biggraphresults.txt
echo "hollywood" >> biggraphresults.txt
python force2vec.py -p ${hollywood} -it 1 -d 128 -t 1 >> biggraphresults.txt
echo "orkut" >> biggraphresults.txt
python force2vec.py -p ${orkut} -it 1 -d 128 -t 1 >> biggraphresults.txt

