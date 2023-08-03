import numpy as np
import statistics
# tplt1=np.loadtxt('prmptmnl')
# tplt2=np.loadtxt('prmpttmpl2')
# tplt3=np.loadtxt('prmpttmpl3')
# tplt4=np.loadtxt('prmpttmpl4') #92
# tplt3dic=np.loadtxt('prmpttmpl3dicvrbl')
# tplt5=np.loadtxt('prmpttmpl5') #92
# tplt6=np.loadtxt('prmpttmpl6')
# tplt7=np.loadtxt('prmpttmpl7')
# tplt8=np.loadtxt('prmpttmpl8')
# tplt9=np.loadtxt('prmpttmpl9')
# tplt10=np.loadtxt('prmpttmpl10')
# bert=np.loadtxt('bert')
tplt1=np.loadtxt('prmpttmplt1ctf')
tplt2=np.loadtxt('prmpttmplt2ctf')
tplt3=np.loadtxt('prmpttmplt3ctf')
tplt4=np.loadtxt('prmpttmplt4ctf')
tplt5=np.loadtxt('prmpttmplt5ctf')
tplt6=np.loadtxt('prmpttmplt6ctf')
tplt7=np.loadtxt('prmpttmplt7ctf')
truelabel=np.loadtxt('truectf')

for i in range(len(tplt6)) :

    tplt6[i] = 1-tplt6[i]
for i in range(len(tplt7)) :

    tplt7[i] = 1-tplt7[i]

ensemble=[tplt1,tplt2,tplt3,tplt4,tplt5,tplt6,tplt7]
# ensemble=[tplt1,tplt2,tplt3,tplt4,tplt5]
hardvt1=[]
hardvt2=[]
hardvt3=[]
hardvt4=[]
hardvt5=[]
hardvt6=[]
hardvt7=[]

hardbert=[]
result = []
def everyacc():
    for i in range(len(tplt3)):
        if tplt1[i] >= 0.5:
            hardvt1.append(1)
        else:
            hardvt1.append(0)
        if tplt2[i] >=0.5:
            hardvt2.append(1)
        else:
            hardvt2.append(0)
        if tplt3[i] >=0.5:
            hardvt3.append(1)
        else:
            hardvt3.append(0)

        if tplt4[i] >=0.5:
            hardvt4.append(1)
        else:
            hardvt4.append(0)
        if tplt5[i] >=0.5:
            hardvt5.append(1)
        else:
            hardvt5.append(0)

        if tplt6[i] >=0.5:
            hardvt6.append(1)
        else:
            hardvt6.append(0)
        if tplt7[i] >=0.5:
            hardvt7.append(1)
        else:
            hardvt7.append(0)

        # if bert[i] >= 0.5:
        #     hardbert.append(1)
        # else:
        #     hardbert.append(0)
    temp=[hardvt1,hardvt2,hardvt3,hardvt4,hardvt5,hardvt6,hardvt7]
    for i in range(7):
        tp = sum([int(i == j and i == 1) for i, j in zip(temp[i], truelabel)])
        tn = sum([int(i == j and i == 0) for i, j in zip(temp[i], truelabel)])
        fp = sum([int(i != j and i == 1) for i, j in zip(temp[i], truelabel)])
        fn = sum([int(i != j and i == 0) for i, j in zip(temp[i], truelabel)])
        if (tp + tn + fp + fn) != 0 and (tp + fp) != 0 and (tp + fn) != 0:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            print("This is {}dataset.".format(i+1))
            print("pecision:{}".format(precision))
            print("f1:{}".format(f1))
            print("recall:{}".format(recall))
            print("accuracy:{}".format(accuracy))
        else:
            print(i)
            print("false")
            print(tp)
            print(fp)
            print(tn)
            print(fn)
#hard vote
def hardvote():
    result=[]
    for i in range(len(tplt3)):
        if tplt1[i] >= 0.5:
            hardvt1.append(1)
        else:
            hardvt1.append(0)
        if tplt2[i] >=0.5:
            hardvt2.append(1)
        else:
            hardvt2.append(0)
        if tplt3[i] >=0.5:
            hardvt3.append(1)
        else:
            hardvt3.append(0)

        if tplt4[i] >=0.5:
            hardvt4.append(1)
        else:
            hardvt4.append(0)
        if tplt5[i] >=0.5:
            hardvt5.append(1)
        else:
            hardvt5.append(0)

        if tplt6[i] >=0.5:
            hardvt6.append(1)
        else:
            hardvt6.append(0)
        if tplt7[i] >=0.5:
            hardvt7.append(1)
        else:
            hardvt7.append(0)
        # if bert[i] >= 0.5:
        #     hardbert.append(1)
        # else:
        #     hardbert.append(0)
    temp=[]
    for i in range(len(hardvt3)):
        # print(i)
        temp=[hardvt1[i],hardvt2[i],hardvt3[i],hardvt4[i],hardvt5[i],hardvt6[i],hardvt7[i]]
        # s=statistics.mean(temp)
        # if s >=0.5:
        #     result.append(1)
        # else:
        #     result.append(0)
        result.append(statistics.mode(temp))
    tp = sum([int(i == j and i == 1) for i, j in zip(result, truelabel)])
    tn = sum([int(i == j and i == 0) for i, j in zip(result, truelabel)])
    fp = sum([int(i != j and i == 1) for i, j in zip(result, truelabel)])
    fn = sum([int(i != j and i == 0) for i, j in zip(result, truelabel)])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('this is hard vote')
    print("pecision:{}".format(precision))
    print("f1:{}".format(f1))
    print("recall:{}".format(recall))
    print("accuracy:{}".format(accuracy))

def softvote():
    result=[]
    for i in range(len(truelabel)):
        temp=[tplt1[i],tplt2[i],tplt3[i],tplt4[i],tplt5[i]]
        s=statistics.mean(temp)
        if s >=0.5:
            result.append(1)
        else:
            result.append(0)
    tp = sum([int(i == j and i == 1) for i, j in zip(result, truelabel)])
    tn = sum([int(i == j and i == 0) for i, j in zip(result, truelabel)])
    fp = sum([int(i != j and i == 1) for i, j in zip(result, truelabel)])
    fn = sum([int(i != j and i == 0) for i, j in zip(result, truelabel)])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('this is soft vote')
    print("pecision:{}".format(precision))
    print("f1:{}".format(f1))
    print("recall:{}".format(recall))
    print("accuracy:{}".format(accuracy))

def weightsoftvote():
    ensemblenp=np.array(ensemble)
    weightvote=[]
    for x1 in range(21):
        for y1 in range(21-x1):
            for i1 in range(21-x1-y1):
                for j1 in range(21-x1-y1-i1):
                    for k1 in range(21-x1-y1-i1-j1):
                        for l1 in range(21-x1-y1-i1-j1-k1):
                                    m1=20-i1-j1-k1-l1-x1-y1
                                    weightvote.append([i1,j1,k1,l1,m1,x1,y1])
    print(len(weightvote))
    top=0
    topweight=[]
    j3=0
    accuracy=0
    f1=0
    recall=0
    precision=0
    for i in range(len(weightvote)):
        temp=np.array(weightvote[i]).reshape(7,1).T
        resultlist=(np.squeeze((np.dot(temp,ensemblenp)/10))).tolist()
        result=[]
        for i in range(len(resultlist)):
            if resultlist[i] >=0.5:
                result.append(1)
            else:
                result.append(0)
        tp = sum([int(i == j and i == 1) for i, j in zip(result,truelabel)])
        tn = sum([int(i == j and i == 0) for i, j in zip(result,truelabel)])
        fp = sum([int(i != j and i == 1) for i, j in zip(result,truelabel)])
        fn = sum([int(i != j and i == 0) for i, j in zip(result,truelabel)])
        if (tp + tn + fp + fn) != 0 and (tp + fp) != 0 and (tp + fn) != 0:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            if accuracy>top:
                top=accuracy
                topweight=np.squeeze(temp.tolist())
                print(top)
                print(topweight)
                print("pecision:{}".format(precision))
                print("f1:{}".format(f1))
                print("recall:{}".format(recall))
                print("accuracy:{}".format(accuracy))
                acc=accuracy
                pre=precision
                rec=recall
                f1true=f1


        j3+=1
        if(j3%1000==0):
            print(j3)
    print(top)
    print(topweight)
    print('this is weight soft vote')
    print("pecision:{}".format(pre))
    print("f1:{}".format(f1true))
    print("recall:{}".format(rec))
    print("accuracy:{}".format(acc))


# softvote()

everyacc()
# hardvote()
# softvote()
weightsoftvote()