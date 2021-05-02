import re
from random import random, sample


class TransitionMatrix:
  """ This is the transition matrix class.

  Attributes: 
      SparseMatrix a dictionary in the form 
                 key: word_1, word_2
                 value: dictionary in form {word:integer, word:integer}
  Methods:
      next_word: add a tripple of word_1, word_2, word_3 in SparseMatrix
      next_word: given a tuple word_1,word_2 generate (next) word with probapilities according to 
      dictionary corresponding to key word_1,word_2 in SparseMatrix
  """
  def __init__(self):
    self.SparseMatrix={}
  def add_tripple(self, word1,word2,word3):
    """ for a given tripple word1,word2,word3
        check if word1,word2 is a key of the dictrionary
          if true and word_3 is a key in the corresponding entry increment value of key word_3 by one
          if true and word_3 is not a key then add word_3 as a key with value one
        if  word1,word2 is not a key, then add a key word_1,word_2 with value {word_3:1}
    """ 
    key=word1+","+word2
    if key in self.SparseMatrix:
      if word3 in self.SparseMatrix[key]:
        self.SparseMatrix[key][word3]=self.SparseMatrix[key][word3]+1
      else:
        self.SparseMatrix[key][word3]=1
    else:
      self.SparseMatrix[key]={word3:1}
  def next_word(self,word1,word2):
    """ generate next word for tuple word_1,word_2
        if word_1,word_2 is a key, then
           retrive the corresponding dictionary and based on it pick randomly a word
           see https://stackoverflow.com/questions/2570690/python-algorithm-to-randomly-select-a-key-based-on-proportionality-weight
        if word_1,word_2 is not a key, then select randomly from list [".","hence","thus","i"]
    """
    key=word1+","+word2
    if key in self.SparseMatrix:
      count=sum(self.SparseMatrix[key].values())
      rand_val = count*random()
      total = 0
      for word,idx in self.SparseMatrix[key].items():
        total += idx
        if rand_val <= total:
            return word
    else:
      return sample(["and"],1)[0]


def preprocess_sentence(sentence):
  sentence=sentence.lower()
  sentence=re.sub(r"[^\w\d.!?\s]+",'',sentence)
  sentence=re.sub('([.,!?])', r' \1 ', sentence)
  sentence = re.sub('\s{2,}', ' ', sentence)
  return sentence

def find_most_probable_path(table, start_hex, end_symbols, max_path=0, prohibited=[]):
  assigned=[start_hex]
  foundTrue=False
  prob=[{"nodes":[start_hex],"prob":1,"length":1}]
  if max_path==0:
    status=False
  else:
    status=True
  while status==True:
    chn=[]
    status=False
    for i in prob:
      if i["length"]<max_path:
          lastElement=i["nodes"][-1]
          if "," not in lastElement:
            lastElement = i["nodes"][-2].split(",")[-1] + "," + lastElement
          if lastElement in table:
            for j in table[lastElement]:
              if j not in assigned and j not in prohibited:
                temp=i.copy()
                js=temp["nodes"].copy()
                js.append(j)
                temp["nodes"]=js
                temp["prob"]=temp["prob"]*table[lastElement][j]
                temp["length"]+=1
                #print(temp)
                chn.append(temp)
                status=True
    maxv=0
    for i in chn:
      if i["prob"]>=maxv:
        maxv=i["prob"]
        added=i
    if added["nodes"][-1] in end_symbols:
      foundTrue=True
      status=False
    assigned.append(added["nodes"][-1])
    prob.append(added)
  if foundTrue==True:
    return prob[-1]["nodes"], prob
  else:
    return prob




def get_matrix(poetry_file):
  poetry_matrix=TransitionMatrix()
  poetry=[poetry_file]
  for FileName in poetry:
    print(f'Processing {FileName}')
    # Open file
    with open(FileName) as f:
      content = f.readlines()
      #remove whitespace characters like `\n` at the end of each line
      content = [x.strip() for x in content]
      content = [x.strip() for x in content if x!=""]
    # Process file
    for text in content:
      doc=preprocess_sentence(text)
      doc=doc.split()
      l=len(doc)
      for i in range(2,l):
        poetry_matrix.add_tripple(doc[i-2],doc[i-1],doc[i])
  return poetry_matrix



def generate_line(input1, input2, endword, poetry_matrix, max_path = 5, prohibited=[]):
  story=input1+" "+input2
  word1 = input1
  word2 = input2
  for i in range(15): # TODO change to while statement and count syllables..?
    new_word=poetry_matrix.next_word(word1,word2)
    story=story+" "+new_word
    if new_word==".":
      print(story)
      story=""
    word1=word2
    word2=new_word
  print(story)
  path = find_most_probable_path(poetry_matrix.SparseMatrix, input1+","+input2, endword, max_path=max_path, prohibited=prohibited)
  return story, path


print(find_most_probable_path("hex2", "hex3",5))
if __name__ == '__main__':
  """
  #Assign class TransitionMatrix to markov_matrix
  Greeks_matrix=TransitionMatrix()
  Greeks=['pg28.txt','1727-0.txt','6130-0.txt']
  for FileName in Greeks:
    print(f'Processing {FileName}')
    # Open file
    with open(FileName) as f:
      content = f.readlines()
      #remove whitespace characters like `\n` at the end of each line
      content = [x.strip() for x in content]
      content = [x.strip() for x in content if x!=""]
    # Process file
    for text in content:
      doc=preprocess_sentence(text)
      doc=doc.split()
      l=len(doc)
      for i in range(2,l):
        Greeks_matrix.add_tripple(doc[i-2],doc[i-1],doc[i])
  """
  poetry_matrix=TransitionMatrix()
  poetry=['./data/poems_selected.txt']
  for FileName in poetry:
    print(f'Processing {FileName}')
    # Open file
    with open(FileName) as f:
      content = f.readlines()
      #remove whitespace characters like `\n` at the end of each line
      content = [x.strip() for x in content]
      content = [x.strip() for x in content if x!=""]
    # Process file
    for text in content:
      doc=preprocess_sentence(text)
      doc=doc.split()
      l=len(doc)
      for i in range(2,l):
        poetry_matrix.add_tripple(doc[i-2],doc[i-1],doc[i])
  word1="it"
  word2="is"
  story=word1+" "+word2
  for i in range(50):
    new_word=poetry_matrix.next_word(word1,word2)
    story=story+" "+new_word
    if new_word==".":
      print(story)
      story=""
    word1=word2
    word2=new_word
  print(story)

  print(find_most_probable_path(poetry_matrix.SparseMatrix, "it,is", "boat",5))
