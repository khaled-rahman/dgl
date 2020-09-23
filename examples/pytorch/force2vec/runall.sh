#node-v17

module load gcc

ITERATIONS=(5)
BATCHES=(128 256 512)
DIMS=(16 64 128)

blogdir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/blogcatalog.mtx"
flickrdir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/flickr.mtx"
youtubedir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/youtube.mtx"

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

		
			#echo "Param:iter="${it},"dim="${d},"batch="${b},"type=1" >> blogcatalogresults.txt
                        #python force2vec.py -p ${blogdir} -it ${iter} -d ${d} -t 1 -b ${b} >> blogcatalogresults.txt
                        #echo "Param:iter="${it},"dim="${d},"batch="${b},"type=2" >> blogcatalogresults.txt
                        #python force2vec.py -p ${blogdir} -it ${iter} -d ${d} -t 2 -b ${b} >> blogcatalogresults.txt

		done
	done
done
