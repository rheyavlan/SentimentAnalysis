import string
import os
from glob import glob

def change_data(articles_path):
  for path in glob(os.path.join(articles_path, '*.txt')):
        with open(path,"r+") as f:
            s = ""
            data = f.read()
            #print(data)
            count = 0
            for i in data:
              s+=str(i)      
              if i == ' ':      
                  count=count+1
                  if count == 25:
                         s+=str('\n')
                         count = 0
            f.close()
        with open(path,"w") as f:
            f.write(s)             
                        
change_data('news_output/DNA')

change_data('news_output/TOI')

