from pyspark import SparkConf
from pyspark.context import SparkContext
import sys
import time
import json
import math

#test_review.json model output

def find_similarity(line):
    output = {}
    if line[1] not in data["business"] or line[0] not in data["user"]:
        return None
    set1 = set(data["business"][line[1]])
    set2 = set(data["user"][line[0]])
    dot_product = len(set1.intersection(set2))
    product_vector_length = math.sqrt(len(set1)) * math.sqrt(len(set2))
    cosine_similarity = dot_product/product_vector_length
    if cosine_similarity >= 0.01:
        output["user_id"] = line[0]
        output["business_id"] = line[1]
        output["sim"] = cosine_similarity
    return(output)

if __name__ == "__main__":
    start = time.time()
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    with open(model_file, "r") as json_file:
        data = json.load(json_file)
    test_rdd = sc.textFile(test_file).map(json.loads).map(lambda x: (x['user_id'], x['business_id'])).persist()
    final = test_rdd.map(find_similarity).filter(lambda x: x!=None).collect()

    with open(output_file, 'w') as filehandle:
        for listitem in final:
            filehandle.writelines(json.dumps(listitem))
            filehandle.write('\n')

    print('Duration: ', time.time() - start)