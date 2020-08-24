# twitter-sentiment-using-spark

from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re

def abb_en(line):
    abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
    }
   
    abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
    return (abbrev)

def remove_features(data_str):
   
    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')    
    mention_re = re.compile(r'@|#(\w+)')  
    RT_re = re.compile(r'RT(\s+)')
    num_re = re.compile(r'(\d+)')
    data_str = str(data_str)
    data_str = RT_re.sub(' ', data_str)  
    data_str = data_str.lower()  
    data_str = url_re.sub(' ', data_str)   
    data_str = mention_re.sub(' ', data_str)  
    data_str = num_re.sub(' ', data_str)
    return data_str 
   
#Write your main function here
def polarity(value):
    if value > 0.0:
        return("Positive")
    elif value < 0.0:
        return("Negative")
    else :
        return("Neutral")

def main(sc, filename):
    data = sc.textFile(filename) #read the file
    
    cleandata = data.map(lambda line:line.split(',')).filter(lambda line:len(line) == 10).filter(lambda line:len(line[1]) > 0) #identify delimiter, length of field and remove empty lines
                         
    sentiment = (cleandata.map(lambda line:line[1]).map(lambda line:line.lower()).map(lambda line:remove_features(line)).map(lambda line:abb_en(line)).map(lambda line:TextBlob(line).sentiment.polarity).map(lambda line:polarity(line))
                         
    output = cleandata.zip(sentiment) #combine both files
    output.saveAsTextFile('try') #save the combined file
         
if __name__ == "__main__":
    
    conf = SparkConf().setMaster("local[1]").setAppName("Try")
    sc = SparkContext(conf=conf)
    filename = ("bitcoin.csv")
    main(sc, filename)
    sc.stop()
  
