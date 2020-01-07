from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as f


def tf_idf(df, n):
    # Extracting terms per each row/document as a list
    temp_df = df.withColumn('terms', f.split(f.lower(f.regexp_replace(df.text_entry, '[^\\w\\s-]', '')), ' '))

    # Calculating total number of words per row/document
    temp_df1 = temp_df.withColumn('total_num_words', f.size('terms'))

    # Extracting words in each documents
    temp_df2 = temp_df1.withColumn('token', f.explode('terms'))

    # Calculating tf
    temp_df3 = temp_df2.groupBy('_id', 'token', 'total_num_words').agg({'token': 'count'}).withColumnRenamed(
        'count(token)', 'occurrence').sort('_id')
    temp_df4 = temp_df3.withColumn('tf', temp_df3.occurrence)

    # Calculating df
    temp_df5 = temp_df4.groupBy('token').agg(f.countDistinct('_id')).withColumnRenamed('count(DISTINCT _id)', 'df')

    # Calculating idf
    temp_df6 = temp_df5.withColumn('idf', f.log10(n / temp_df5.df))

    # Calculating tf-idf
    joined_df = temp_df4.join(temp_df6, temp_df4.token == temp_df6.token).select(temp_df4.token, temp_df4._id,
                                                                                 temp_df4.tf, temp_df6.df, temp_df6.idf)
    result = joined_df.withColumn('tf_idf', joined_df.tf * joined_df.idf)

    return result


def search_words(query, N, tokensTfIdf, df):
    # Splitting the query into separate words
    query_splitted = query.lower().split()

    # Counting number of words in the query
    num_words_query = len(query_splitted)

    # Filtering the tokens containing the query's word
    filtered = tokensTfIdf.filter(tokensTfIdf.token.rlike('(^|\\s)(' + '|'.join(query_splitted) + ')(\\s|$)'))

    # counting the number of words that a row/document have in common with query and sum their tf-idf values
    q_sigma_count = filtered.groupby(tokensTfIdf._id).agg(f.count('tf_idf').alias('counts'),
                                                          f.sum('tf_idf').alias('sigma'))

    # Calculating the score for each document
    docs_score = q_sigma_count.select('_id',
                                      ((q_sigma_count.counts / num_words_query) * q_sigma_count.sigma).alias('score'))

    # Retrieve top N maximum scores for the query
    ranked = docs_score.orderBy('score', ascending=False).limit(N)

    # Retrieve text_entry's associated with each high score _id
    search_result = ranked.join(df, df._id == ranked._id).select(ranked._id, f.bround(ranked.score, 3),
                                                                 'text_entry').orderBy('score',
                                                                                       ascending=False).collect()

    # Printing the search result
    def print_search(q, n, output):
        print 'Query: ', q, ',', 'N: ', n
        for i in output: print tuple(i)
        print '\n'


def main(sc):
    # Reading shakespeare_full.json and create a dataframe
    sqlcontext = SQLContext(sc)
    path = '/user/root/shakespeare_full.json'
    dataframe = sqlcontext.read.json(path)

    # Count number of rows/documents
    num_documents = dataframe.count()

    # Building Index
    tokensWithTfIdf = tf_idf(dataframe, num_documents)

    # Caching the tokensWithTfIdf dataframe
    tokensWithTfIdf.cache()

    # Showing 20 entries of inverted_index
    tokensWithTfIdf.show()

    # Searching 3 sample queries
    query = ['to be or not', 'so far so', 'if you said so']

    for i in range(len(query)):
        search_words(query[i], 5, dataframe)

    print '\n'


if __name__ == '__main__':
    conf = SparkConf().setAppName('MyApp')
    sc = SparkContext(conf=conf)
    main(sc)
    sc.stop()
