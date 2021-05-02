table={'hex1': {'hex2': 1.0},
 'hex2': {'hex4': 0.4, 'hex7': 0.2, 'hex6': 0.2, 'hex1': 0.2},
 'hex4': {'hex3': 1.0},
 'hex3': {'hex6': 0.3333333333333333, 'hex2': 0.6666666666666666},
 'hex6': {'hex1': 0.3333333333333333,
  'hex4': 0.3333333333333333,
  'hex5': 0.3333333333333333},
 'hex7': {'hex6': 1.0},
 'hex5': {'hex3': 1.0}}

def find_most_probable_path(start_hex, end_hex, max_path=0):
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
          for j in table[lastElement]:
            if j not in assigned:
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
    if added["nodes"][-1]==end_hex:
      foundTrue=True
      status=False
    assigned.append(added["nodes"][-1])
    prob.append(added)
  if foundTrue==True:
    return prob[-1]["nodes"]
  else:
    return None


print(find_most_probable_path("hex2", "hex3",5))
