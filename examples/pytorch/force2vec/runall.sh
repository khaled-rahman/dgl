#node-v17

module load gcc

ITERATIONS=(1)
BATCHES=(384 768 1152)
DIMS=(32 64 128 256)

blogdir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/blogcatalog.mtx"
flickrdir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/flickr.mtx"
youtubedir="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/youtube.mtx"
amazon="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/special/com-Amazon/com-Amazon.mtx"
orkut="/N/u2/m/morahma/Research/Fall2019/GraphEmbedding/HomophilyEmbedding/GraphEmbedding/force2vec/datasets/input/orkut.mtx"
flan="/N/u2/m/morahma/Research/Fall2020/GITHUBDGL/GE-SDDMM/dataset/flan.mtx"
harvard="/N/u2/m/morahma/Research/Fall2020/GITHUBDGL/GE-SDDMM/dataset/harvard.mtx"
stanford="/N/u2/m/morahma/Research/Fall2020/GITHUBDGL/GE-SDDMM/dataset/stanford.mtx"


resultfile="dgltiming.txt"

<< COMMENT
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
COMMENT

iter=10
dimf=128
typ=1
batch=1

for dim in "${DIMS[@]}"
do
echo "harvard","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${harvard} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "stanford","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${stanford} -it ${iter} -b ${batch} -d ${dim} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "cora","dim:"${dim} >> ${resultfile}
python force2vec.py -g cora -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "pubmed","dim:"${dim} >> ${resultfile}
python force2vec.py -g pubmed -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "blogcatalog","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${blogdir} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "flickr","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${flickrdir} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "amazon","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${amazon} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "youtube","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${youtubedir} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "orkut","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${orkut} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
echo "=====================================================" >> ${resultfile}
echo "flan","dim:"${dim} >> ${resultfile}
python force2vec.py -p ${flan} -it ${iter} -d ${dim} -b ${batch} -t ${typ} >> ${resultfile}
done
