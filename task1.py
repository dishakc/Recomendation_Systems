from pyspark import SparkConf
from pyspark.context import SparkContext
import sys
import time
import json
import itertools

#train_review.json task.res


def create_permutation(j):
    key_value = []
    h1 = (a1 * j + b1) % m
    h2 = ((a2 * j + b2) % p) % m
    for i in range(1,n+1):
        h3 = ((i * h1 + i * h2 + i * i) % p) % m
        key_value.append(h3)
    return (j, key_value)


def replace_value(line):
    key_value = []
    for i in line[1]:
        if i in user_list:
            place = user_list.index(i)
            key_value.append(place)
    return (line[0], key_value)

def create_signature(line):
    final = []
    for j in range(n):
        temp = []
        for i in line[1]:
            num = permutation[i][j]
            temp.append(num)
        mini = min(temp)
        final.append(mini)
    return (line[0], final)


def split_into_lists(each):
    nlist = each[1]
    x = [nlist[i * rows:(i + 1) * rows] for i in range((len(nlist) + rows - 1) // rows)]
    return (each[0], x)


def pairs(line):
    output = []
    user_list = line[1]
    for i in user_list:
        pos = user_list.index(i)
        v = (pos, tuple(i))
        a = (tuple(v), [line[0]])
        output.append(a)
    return output


def create_candidate_pair(line):
    output = []
    for subset in itertools.combinations(line, 2):
        output.append(subset)
    return output


if __name__ == "__main__":
    start = time.time()
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    p = 26203
    a1 = 7
    b1 = 5
    a2 = 5
    b2 = 3

    n = 50
    bands = 50
    rows = int(n / bands)


    data_rdd = sc.textFile(input_file).map(json.loads)
    train_review_rdd = data_rdd.map(lambda x: (x['business_id'], x['user_id'])).distinct().persist()

    group_rdd = train_review_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).persist()

    user_rdd = train_review_rdd.map(lambda x: x[1]).distinct()
    user_list = user_rdd.collect()
    m = len(user_list)
    user_rdd_index = user_rdd.zipWithIndex().map(lambda x: x[1])

    permutation = user_rdd_index.map(create_permutation).collectAsMap()

    replaced = group_rdd.map(replace_value)

    signature = replaced.map(create_signature)

    dividing_into_band = signature.map(split_into_lists)

    pair_rdd = dividing_into_band.flatMap(pairs).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) > 1).map(lambda x: x[1])

    candidate_pairs = pair_rdd.flatMap(create_candidate_pair).distinct().collect()
    print(len(candidate_pairs))

    compare_dict = group_rdd.collectAsMap()

    output_fl = open(output_file, "w+")
    for i in candidate_pairs:
        output = {}
        setA = set(compare_dict[i[0]])
        setB = set(compare_dict[i[1]])
        inter = len(setA.intersection(setB))
        union = len(setA.union(setB))
        similarity = inter/union
        if (similarity >= 0.05):
            output["b1"] = i[0]
            output["b2"] = i[1]
            output["sim"] = similarity
            output_fl.writelines(json.dumps(output))
            output_fl.write('\n')
    output_fl.close()
    print('Duration: ', time.time() - start)
