from pyspark import SparkConf
from pyspark.context import SparkContext
import sys
import time
import json
import math
import itertools

#train_review.json task3item.model item_based
#train_review.json task3user.model user_based

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
        place = business_dict[i]
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
    sorted_list = sorted(line)
    for subset in itertools.combinations(sorted_list, 2):
        output.append(subset)
    return output

def check_similarity(line):
    setA = set(compare_dict[line[0]])
    setB = set(compare_dict[line[1]])
    inter = len(setA.intersection(setB))
    union = len(setA.union(setB))
    similarity = inter / union
    if (similarity >= 0.01 and inter >=3):
        return line


if __name__ == "__main__":
    start = time.time()
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g")
                                  .set("spark.driver.memory", "4g"))
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    cf_type = sys.argv[3]

    output_fl = open(model_file, "w+")
    data_rdd = sc.textFile(train_file).map(json.loads).persist()

    if cf_type == "item_based":

        business_text_rdd = data_rdd.map(lambda x: ((x['business_id'], x['user_id']),x['stars'])).distinct()

        seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
        combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))

        business_text_rdd = business_text_rdd.aggregateByKey((0,0),seqOp,combOp).map(lambda x: (x[0],x[1][0]/x[1][1]))

        each_business_user_dictionary = business_text_rdd.collectAsMap()

        business_user = business_text_rdd.map(lambda x: x[0]).groupByKey().mapValues(list)

        business_user_dictionary = business_user.collectAsMap()

        business_id_list = business_user_dictionary.keys()

        for subset in itertools.combinations(business_id_list, 2):
            output = {}
            b1_users = business_user_dictionary[subset[0]]
            b2_users = business_user_dictionary[subset[1]]
            setA = set(b1_users)
            setB = set(b2_users)
            intersection = setA.intersection(setB)
            total_length = len(intersection)
            if total_length >= 3:
                star_b1 = []
                star_b2 = []
                num = 0
                den_1 =0
                den_2 = 0
                for each_user in intersection:
                    pair_key_1 = (subset[0],each_user)
                    star_b1.append(each_business_user_dictionary[pair_key_1])
                    pair_key_2 = (subset[1], each_user)
                    star_b2.append(each_business_user_dictionary[pair_key_2])
                    r1 = sum(star_b1)/total_length
                    r2 = sum(star_b2)/total_length
                for l in range(total_length):
                    num = num + ((star_b1[l]-r1)*(star_b2[l]-r2))
                    den_1 = den_1 + ((star_b1[l]-r1)**2)
                    den_2 = den_2 + ((star_b2[l] - r2) ** 2)
                den = math.sqrt(den_1)*math.sqrt(den_2)
                '''if num == 0 or den == 0:
                    sim = 0
                else:
                    sim = num/den'''

                if den!=0:
                    sim = num/den
                #elif num==0 and den_1==0 and den_2 ==0: (means all the rows are similar)
                    #sim=1.0
                else:
                    continue
                if sim>0:
                    output["b1"] = subset[0]
                    output["b2"] = subset[1]
                    output["sim"] = sim
                    output_fl.writelines(json.dumps(output)+"\n")
    elif cf_type == "user_based":
        user_text_rdd = data_rdd.map(lambda x: ((x['user_id'], x['business_id']), x['stars'])).distinct()

        seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
        combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))

        user_text_rdd = user_text_rdd.aggregateByKey((0, 0), seqOp, combOp).map(
            lambda x: (x[0], x[1][0] / x[1][1]))

        each_user_business_dictionary = user_text_rdd.collectAsMap()

        user_business = user_text_rdd.map(lambda x: x[0]).groupByKey().mapValues(list)

        user_business_dictionary = user_business.collectAsMap()

        p = 10259
        a1 = 70
        b1 = 55
        a2 = 50
        b2 = 35

        n = 60
        bands = 60
        rows = int(n / bands)

        user_text_rdd = data_rdd.map(lambda x: (x['user_id'], x['business_id'])).distinct().persist()

        user_group = user_text_rdd.groupByKey().mapValues(list)

        business = user_text_rdd.map(lambda x: x[1]).distinct().zipWithIndex()

        business_dict = business.collectAsMap()
        m = business.count()

        permutation = business.map(lambda x: x[1]).map(create_permutation).collectAsMap()

        replaced = user_group.map(replace_value)

        signature = replaced.map(create_signature)

        dividing_into_band = signature.map(split_into_lists)


        pair_rdd = dividing_into_band.flatMap(pairs)

        pair_rdd = pair_rdd.reduceByKey(lambda a, b: a + b)

        pair_rdd = pair_rdd.filter(lambda x: len(x[1]) > 1)

        pair_rdd = pair_rdd.map(lambda x: x[1])

        candidate_pairs = pair_rdd.flatMap(create_candidate_pair).distinct() #optimisze

        compare_dict = user_group.collectAsMap()

        similar = candidate_pairs.map(check_similarity).filter(lambda x: x is not None).collect()

        for subset in similar:
            output = {}
            b1_users = user_business_dictionary[subset[0]]
            b2_users = user_business_dictionary[subset[1]]
            setA = set(b1_users)
            setB = set(b2_users)
            intersection = setA.intersection(setB)
            total_length = len(intersection)
            if total_length >= 3:
                star_b1 = []
                star_b2 = []
                num = 0
                den_1 =0
                den_2 = 0
                for each_bid in intersection:
                    pair_key_1 = (subset[0],each_bid)
                    star_b1.append(each_user_business_dictionary[pair_key_1])
                    pair_key_2 = (subset[1], each_bid)
                    star_b2.append(each_user_business_dictionary[pair_key_2])
                    r1 = sum(star_b1)/total_length
                    r2 = sum(star_b2)/total_length
                for l in range(total_length):
                    num = num + ((star_b1[l]-r1)*(star_b2[l]-r2))
                    den_1 = den_1 + ((star_b1[l]-r1)**2)
                    den_2 = den_2 + ((star_b2[l] - r2) ** 2)
                den = math.sqrt(den_1)*math.sqrt(den_2)
                #if num == 0 or den == 0:
                #    sim = 0
                #else:
                #    sim = num/den
                if den!=0:
                    sim = num/den
                #elif num==0 and den_1==0 and den_2 ==0: (means all the rows are similar)
                    #sim=1.0
                else:
                    continue
                if sim>0:
                    output["u1"] = subset[0]
                    output["u2"] = subset[1]
                    output["sim"] = sim
                    output_fl.writelines(json.dumps(output)+"\n")
    else:
        print("Invalid case selection")

    output_fl.close()
    print('Duration: ', time.time() - start)