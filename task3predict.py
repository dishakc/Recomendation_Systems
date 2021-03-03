from pyspark import SparkConf
from pyspark.context import SparkContext
import sys
import time
import json


#train_review.json test_review.json task3item.model task3item.predict item_based
#train_review.json test_review.json task3user.model task3user.predict user_based

def case_1(line):
    output = {}
    tup = (line[1], line[0])
    actual_star = each_business_user_dictionary.get(tup, None)
    if actual_star is not None:
        output["user_id"] = line[0]
        output["business_id"] = line[1]
        output["stars"] = actual_star
        return output
    business_list = user_business_dictionary[line[0]]
    pair = []
    for x in business_list:
        search = (line[1],x)
        sorted_tuple = tuple(sorted(search))
        if sorted_tuple in model_dictionary.keys():
            wt = model_dictionary[sorted_tuple]
            tup = (x,line[0])
            star = each_business_user_dictionary[tup]
            wt_star = (wt,star)
            pair.append(wt_star)
    if len(pair) > 0:
        sorted_pair = sorted(pair,key=lambda x:-x[0])
        top_pairs = sorted_pair[:8]
        num = sum([x[0] * x[1] for x in top_pairs])
        den = sum([x[0] for x in top_pairs])
        predicted = num/den
        output["user_id"]=line[0]
        output["business_id"]=line[1]
        output["stars"]=predicted
    if output.keys()==None:
        output=None
    return output

def find_average(u1,u2):
    b1 = set(user_business_dictionary[u1])
    b2 = set(user_business_dictionary[u2])
    inter = b1.intersection(b2)
    stars = []
    for x in inter:
        tuple2 = (u2, x)
        star_2 = each_user_business_dictionary[tuple2]
        stars.append(star_2)
    average = sum(stars)/len(stars)
    return average

def average(u1):
    b1 = user_business_dictionary[u1]
    stars = []
    for x in b1:
        tup = (u1, x)
        star = each_user_business_dictionary[tup]
        stars.append(star)
    average = sum(stars) / len(stars)
    return average

def case_2(test):
    bid = test[0]
    uid = test[1]
    output = {}
    tup = (uid,bid)
    actual_star = each_user_business_dictionary.get(tup, None)
    if actual_star is not None:
        output["user_id"] = uid
        output["business_id"] = bid
        output["stars"] = actual_star
        return output
    user_list = business_user_dictionary.get(bid,[])
    pair = []
    average_r1= average(uid)
    for x in user_list:
        search = (uid,x)
        sorted_tuple = tuple(sorted(search))
        if search in model_dictionary.keys():
            wt = model_dictionary[sorted_tuple]
            tup = (x,bid)
            star = each_user_business_dictionary[tup]
            avg = find_average(uid,x)
            dif = star-avg
            dif_wt = (dif,wt)
            pair.append(dif_wt)
    if len(pair) >= 3:
        sorted_pair = sorted(pair,key=lambda x:-x[1])
        top_pairs = sorted_pair[:8]
        #top_pairs = pair
        num = sum([x[0] * x[1] for x in top_pairs])
        den = sum([x[1] for x in top_pairs])
        predicted = average_r1 + (num/den)
        output["user_id"]=uid
        output["business_id"]=bid
        output["stars"]=predicted
    if output.keys()==None:
        output=None
    return output

if __name__ == "__main__":
    start = time.time()
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))

    if len(sys.argv) != 6:
        print("Execution format: spark-submit task3predict.py $ASNLIB/publicdata/train_review.json"
              " $ASNLIB/publicdata/test_review.json model_file output_file cf_type")
        exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    output_file = sys.argv[4]
    cf_type = sys.argv[5]

    output_fl = open(output_file, "w+")
    train_data_rdd = sc.textFile(train_file).map(json.loads).persist()
    test_data_rdd = sc.textFile(test_file).map(json.loads).persist()
    model_rdd = sc.textFile(model_file).map(json.loads).persist()

    if cf_type == "item_based":
        business_train_rdd = train_data_rdd.map(lambda x: ((x['business_id'], x['user_id']), x['stars'])).distinct()
        test_rdd = test_data_rdd.map(lambda x: (x['user_id'], x['business_id']))
        model_dictionary = model_rdd.map(lambda x: ((x['b1'], x['b2']),x['sim'])).map(lambda x : (tuple(sorted(x[0])),x[1])).collectAsMap()

        seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
        combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))

        business_rdd = business_train_rdd.aggregateByKey((0, 0), seqOp, combOp).map(lambda x: (x[0], x[1][0] / x[1][1])).persist()
        each_business_user_dictionary = business_rdd.collectAsMap()

        user_business = business_rdd.map(lambda x: (x[0][1],x[0][0])).groupByKey().mapValues(list)
        user_business_dictionary = user_business.collectAsMap()


        weights = test_rdd.map(case_1).filter(lambda x:x!=None).collect()
        #print(weights.take(2))
        for i in weights:
            if i != {}:
                output_fl.writelines(json.dumps(i) + "\n")
    elif cf_type == "user_based":
        user_train_rdd = train_data_rdd.map(lambda x: ((x['user_id'], x['business_id']), x['stars'])).distinct()
        test_rdd = test_data_rdd.map(lambda x: (x['business_id'], x['user_id']))
        model_dictionary = model_rdd.map(lambda x: ((x['u1'], x['u2']), x['sim'])).map(lambda x : (tuple(sorted(x[0])),x[1])).collectAsMap()


        seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
        combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))

        user_rdd = user_train_rdd.aggregateByKey((0, 0), seqOp, combOp).map(lambda x: (x[0], x[1][0] / x[1][1])).persist()
        each_user_business_dictionary = user_rdd.collectAsMap()

        business_user = user_rdd.map(lambda x: (x[0][1], x[0][0])).groupByKey().mapValues(list)
        business_user_dictionary = business_user.collectAsMap()

        user_business = user_rdd.map(lambda x: (x[0][0], x[0][1])).groupByKey().mapValues(list)
        user_business_dictionary = user_business.collectAsMap()

        weights = test_rdd.map(case_2).filter(lambda x: x != None).collect()
        for i in weights:
            if i != {}:
                output_fl.writelines(json.dumps(i) + "\n")
    else:
        print("Invalid case selection")
    output_fl.close()
    print('Duration: ', time.time() - start)