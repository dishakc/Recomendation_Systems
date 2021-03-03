from pyspark import SparkConf
from pyspark.context import SparkContext
import sys
import time
import json
import collections
import math
import itertools

#train_review.json model stopwords

def filtertext(line):
    value = []
    for text in line[1]:
        listkeywords = text.split()
        for word in listkeywords:
            word = word.lower()
            word = word.replace("(","").replace("[","").replace(",","").replace(".","").replace("!","").replace("?","").replace(":","").replace(";","").replace("]","").replace(")","")
            if word not in stopwordslist and word != "":
                value.append(word)
    return (line[0],value)


def combine(line):
    output = []
    final = []
    text = line[1]
    count = collections.Counter(text)
    maxk = count.most_common(1)
    for c in count:
        f = count[c]
        tf = f/maxk[0][1]
        x=N/ni[c]
        idf = math.log2(x)
        tf_idf = tf * idf
        word = (c,tf_idf)
        output.append(word)
    output.sort(key=lambda x: -x[1])
    output = output[:200]
    for i in output:
        final.append(i[0])
    return (line[0],final)


def merge(line):
    words = []
    for ele in line[1]:
        temp = business_profile[ele]
        words.append(temp)
    result = list(itertools.chain.from_iterable(words))
    count = collections.Counter(result)
    top_words = count.most_common(200)
    final = [i[0] for i in top_words]
    return (line[0],final)


if __name__ == "__main__":
    start = time.time()
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
    train_file = sys.argv[1]
    model_file = sys.argv[2]

    output_fl = open(model_file, "w+")
    stopwords = sys.argv[3]

    stopwordslist = sc.textFile(stopwords).collect()

    writing = {}

    data_rdd = sc.textFile(train_file).map(json.loads).persist()

    business_text_rdd = data_rdd.map(lambda x: (x['business_id'], x['text'])).distinct().persist()
    user_business_rdd = data_rdd.map(lambda x: (x['user_id'], x['business_id'])).distinct().persist()
    group_rdd = business_text_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b)
    filtered_rdd = group_rdd.map(filtertext).persist()
    N = business_text_rdd.count()
    print(N)
    text = filtered_rdd.flatMapValues(lambda x: x).distinct().map(lambda x:(x[1],[x[0]]))
    ni = text.reduceByKey(lambda a, b: a + b).map(lambda x:(x[0],len(x[1]))).collectAsMap()
    business_profile = filtered_rdd.map(combine).collectAsMap()
    print(len(business_profile))
    grouped_users = user_business_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b)
    user_profile = grouped_users.map(merge).collectAsMap()
    writing["business"] = business_profile
    writing["user"] = user_profile
    output_fl.writelines(json.dumps(writing))
    output_fl.close()
    print('Duration: ', time.time() - start)
