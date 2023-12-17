import math
from collections import Counter
from numpy import mean, max
class Metric:
    def __init__(self):
        super(Metric, self).__init__()


    def MAP(self, results, ref, longtail):
        API_num = len(ref)
        ref_longtail = []
        for item_col in longtail:
            for item_raw in item_col:
                if item_raw > 0:
                    ref_longtail.append(item_raw)

        map=[]
        for item in ref:
            map_temp= 0
            if item>0:
                curren_i = 0
                for j in range(len(results)):
                    if item in results[j]:
                        curren_i+=1
                        map_temp += curren_i / (j+1)
                if curren_i!=0:
                    map.append(map_temp/curren_i)

        for item in ref_longtail:
            curren_i = 0
            for j in range(len(results)):
                if item in results[j]:
                    curren_i += 1
                    map.append(0.4 * curren_i/(j+1))

        return sum(map)/API_num

    def NDCG(self, results, ref, longtail):
        API_num = len(ref)
        ndcg = 0
        for i in range(len(ref)):
            ndg = 0
            common_i = 0
            longtail_i = 0
            if ref[i]>0:
                for j in range(len(results)):
                    if ref[i] in results[j]:
                        ndg += 1 / math.log2(j+2)
                        common_i += 1
            for item in longtail[i]:
                if item>0:
                    for j in range(len(results)):
                        if item > 0 and item in results[j]:
                            ndg += 0.4 / math.log2(j + 2)
                            longtail_i+=1
            idcg = 0
            if common_i==0 and longtail_i==0:
                idcg = 1
            else:
                idcg = self.IDCG(common_i, longtail_i)
            ndcg+=ndg/idcg
        return ndcg/API_num

    def IDCG(self, common_i, longtail_i):
        idcg = 0
        if common_i!=0:
            for i in range(common_i):
                idcg+= 1/math.log2(i+2)
        if longtail_i!=0:
            for i in range(common_i,longtail_i+common_i):
                idcg+= 0.4/math.log2(i+2)
        return idcg


    def bleu(self, results, ref, longtail):
        longtail_ref = []
        for item in longtail:
            for item_i in item:
                if item_i>0:
                    longtail_ref.append(item_i)
        bleu_list = []
        for result in results:
            bleu_list.append(self.one_bleu(result, ref, longtail_ref))
        return max(bleu_list), mean(bleu_list)




    def one_bleu(self, result, ref, longtail):

        sim_bleu = []
        for n in range(1,min(len(ref)+1,3)):
            temp = 0
            result_ngrams = Counter([tuple(result[i: i+n]) for i in range(len(result)+1-n)])
            ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref)+1-n)])
            temp+=sum((result_ngrams&ref_ngrams).values())
            for item in longtail:
                if item in result:
                    temp+=0.4
            max_len = 1
            if len(ref) + 1 - n > 0:
                max_len = len(ref) + 1 - n
            sim_bleu.append(temp/max_len)
        return mean(sim_bleu)




if __name__ == '__main__':
    metric = Metric()
    print(metric.bleu([[1,3,6,8,4,2],[3,19,2,4,2],[1,4,3,3,4]],[1,3,5,2,3,4,2],[[0,0],[19,0],[0,0],[0,0],[0,0],[0,0],[0,0]]))







