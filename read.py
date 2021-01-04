import os
import sys
import argparse
import numpy as np
from numba import cuda,jit
import time
import math
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,Birch


parser=argparse.ArgumentParser(description='T')
parser.add_argument("--input",type=str)
parser.add_argument("--headstar",type=str)
parser.add_argument("--ori",type=str)
parser.add_argument("--center",type=str)
parser.add_argument("--mode",type=int,default=0)
parser.add_argument("--headnum",type=int,default=29)
parser.add_argument("--size",type=int,default=41)
parser.add_argument("--output",type=str,default='out.log')
args=parser.parse_args()

starfile=open(args.input)
sr=starfile.readline()
colors=['red','green','blueviolet','darkgray','darkgoldenrod','darkorchid','darkturquoise','goldenrod','honeydew','indianred','mediumaquamarine','magenta']


def UPDATE_P(P,D):
#~~~~~~~~~~~~~~~~~~~~~~~UPDATE P ~~~~~~~~~~~~~~~
    if D==1:
        if (P+1)%6==0:
            D=-1
            P=P+6
        else:
            P=P+D
    else:
        if D==-1:
            if P%6==0:
                D=1
                P=P+6
            else:
                P=P+D
    return P,D
#~~~~~~~~~~~~~~~~~~~~~~~UPDATE  P ~~~~~~~~~~~~~~

if args.mode==0 or args.mode==1:
    lines=0
    while sr:
        sp=sr.split(' ')
        if len(sp)>=args.headnum:
           lines=lines+1
        sr=starfile.readline()

    starfile.close()
    print('total lines:',lines)
    starfile=open(args.input)
    cd=np.zeros(shape=(lines*2,2),dtype=np.float32)
    tk=0
    sr=starfile.readline()

    while sr:
        sp=sr.split(' ')
        if len(sp)>=args.headnum:
            l=0
            for i in range(len(sp)):
                if sp[i]!='':
                    l=l+1
                    if l==19:
                    #print(sp[i])
                        cd[tk,0]=float(sp[i])
                        cd[tk+lines,0]=cd[tk,0]+360
                    if l==20:
                        cd[tk,1]=float(sp[i])
                        cd[tk+lines,1]=cd[tk,1]
                        tk=tk+1
                #if l==29:
                    #print('end of one line',sp[i])
        sr=starfile.readline()

    starfile.close()


    ot=open(args.output,"w+")
    for wl in range(lines):
        ot.write(str(cd[wl,0]))
        ot.write(" ")
        ot.write(str(cd[wl,1]))
        ot.write("\n")
    ot.close()


    

if args.mode==1:

    km=KMeans(n_clusters=82)
    s=km.fit(cd)
    cs=s.cluster_centers_
    print(cs)
    kmout=open('kmout.log','w+')
    for wl in range(82):
        kmout.write(str(cs[wl,0]))
        kmout.write(" ")
        kmout.write(str(cs[wl,1]))
        kmout.write("\n")
    kmout.close()
    plt.scatter(cd[:,0],cd[:,1],s=0.01)
    plt.scatter(cs[:,0],cs[:,1],marker='x',color='red',s=0.5)
    plt.show()

if args.mode>=2 and args.mode <=9:
    lines=0
    while sr:
        #sp=sr.split(' ')
        #if len(sp)>=args.headnum:
        lines=lines+1
        sr=starfile.readline()
    cd=np.zeros(shape=(lines*2,2),dtype=np.float32)
    starfile.close()
    starfile=open(args.input)
    sr=starfile.readline()
    tk=0
    while sr:
        sp=sr.split('\n')
        sp=sp[0].split(' ')
        cd[tk,0]=float(sp[0])
        cd[tk+lines,0]=cd[tk,0]+360
        cd[tk,1]=float(sp[1])
        cd[tk+lines,1]=cd[tk,1]
        tk=tk+1
        sr=starfile.readline()
    starfile.close()
    #plt.scatter(cd[:,0],cd[:,1],s=0.01)
    #plt.show()

if args.mode==3:
    db=DBSCAN(eps=0.5,min_samples=10,metric='euclidean',metric_params=None,algorithm='auto')
    s=db.fit_predict(cd)
    plt.scatter(cd[:,0],cd[:,1],s=0.01,c=s)
    plt.show()

    
if args.mode==4:

    km=KMeans(n_clusters=82) 
    s=km.fit_predict(cd)
    #cs=s.cluster_centers_
    #print(cs)
    kmout=open('kmout.log','w+')
    for wl in range(82):
        kmout.write(str(s[wl,0]))
        kmout.write(" ")
        kmout.write(str(s[wl,1]))
        kmout.write("\n")
    kmout.close()
    plt.scatter(cd[:,0],cd[:,1],s=0.01,c=s)
    plt.show()

if args.mode==5:
    ag=AgglomerativeClustering(n_clusters=82)
    s=ag.fit_predict(cd)
    plt.scatter(cd[:,0],cd[:,1],s=0.01,c=s)
    plt.show()

if args.mode==6:
    bi=Birch(n_clusters=82)
    s=bi.fit_predict(cd)
    plt.scatter(cd[:,0],cd[:,1],s=0.01,c=s)
    plt.show()

if args.mode==7:
    gp=0
    groups=[]
    heads=[]
    
    while gp <=12:
        heads.append([0])
        gp=gp+1

    gp=0
    while gp <=72:
        groups.append([0])
        gp=gp+1

    for i in range (lines):
        heads[int((cd[i,0]+180)/29)%13].append((i,cd[i,0],cd[i,1]))
    
    for i in range(12):
        for hi in range(len(heads[i])-1):
            groups[i*6+int(heads[i][hi+1][2]/30)].append(heads[i][hi+1])

    ot=open('groups','w+')
    
    for gpi in range(len(groups)):
        ot.write('-------------\n')
        ot.write('-------------\n')
        ot.write('\n')
        ot.write('group :')
        ot.write(str(gpi+1))
        ot.write('\n')
        ot.write('total number :')
        ot.write(str(len(groups[gpi])))       
        ot.write('\n')

        ot.write(str(groups[gpi]))
        ot.write('\n')
        ot.write('\n')

    ot.close()
    pd=cd[:lines]
    plt.scatter(pd[:,0],pd[:,1],s=0.01)
    for gi in range(72):
        plt.scatter(groups[gi][len(groups[gi])-1][1],groups[gi][len(groups[gi])-1][2],s=5,color=colors[gi%(len(colors))])
        plt.scatter(groups[gi][1][1],groups[gi][1][2],s=5,color=colors[gi%(len(colors))])
    
    plt.show()
    
if args.mode>=8:
    #-----------------------------------------begin------------------------------------------data block------------------------------------

    gp=0
    groups=[]
    heads=[]
    
    while gp <=12:
        heads.append([0])
        gp=gp+1

    gp=0
    while gp <=72:
        groups.append([])
        gp=gp+1

    for i in range (lines):
        heads[int((cd[i,0]+180)/29)%12].append((i,cd[i,0],cd[i,1]))
    
    for i in range(12):
        for hi in range(len(heads[i])-1):
            groups[i*6+int(heads[i][hi+1][2]/30)].append(heads[i][hi+1])
  #  t=0
  #  for gi in range(len(groups)):
  #      t=t+len(groups[gi])
  #  print(t)    
  #  print (lines)
    #-----------------------------------------end------------------------------------------data block------------------------------------


    #-----------------------------------------begin----------------------------------------final block-----------------------------------
    Final=[]
    fp=0
    mf=args.size                                              # max block number
    while fp<mf:
        Final.append([])
        fp=fp+1

    #---- final block init finished --------
    U=0                                                       # used size
    P=0                                                       # data point,
    D=1                                                       # Direction 
    N=int(lines/mf)                                           # max size of each final block
    F=0                                                       # final point
    C=N                                                       # called for size
    A=len(groups[P])                                          # avaliable size 
    #--------------------------------------------------end------------------------------init----------------------------------------------
    total=0

    #-------------------------------loop main------------------------------------------------
    while F<mf:
      #  print(P,F)
        #--------------------chech for A ~ C-----------------------
        #---------------------------------------------------------
        if A<C:
            # print('\nCalled :',C,' in final [ ',F,']')

            #++++++++++++++++++++++++++++++++++
            for ti in range(U,U+A):
                Final[F].append(groups[P][ti])
            #++++++++++++++++++++++++++++++++++
            #print('DATA[',P,']:size--',len(groups[P]),'--used:',U+A,'\n--left:',len(groups[P])-U-A)
            P,D=UPDATE_P (P,D)
            total=total+A

           # print(total)
            U=0
            A=len(groups[P])   
            C=N-len(Final[F])

        #--------------------------------------------------------
        else:
            #---------------------------------------------------
            if A==C:
               # print('\nCalled :',C,' in final [ ',F,']')
                #++++++++++++++++++++++++++++++++++
                for ti in range(U,U+A):
                    Final[F].append(groups[P][ti])
                #++++++++++++++++++++++++++++++++++
               # print('DATA[',P,']:size--',len(groups[P]),'--used:',U+A,'\n--left:',len(groups[P])-U-A) 
                          
                P,D=UPDATE_P (P,D)


                total=total+A
               # print(total)
                U=0
                F=F+1
                A=len(groups[P])   
                if F<mf:
                    C=N-len(Final[F])

            #---------------------------------------------------
            else:
                #-----------------------------------------------
                if A>C:
                   # print('\nCalled :',C,' in final [ ',F,']')
                    #++++++++++++++++++++++++++++++++++
                    for ti in range(U,U+C):
                        Final[F].append(groups[P][ti])
                    #++++++++++++++++++++++++++++++++++
                   #print('DATA[',P,']:size--',len(groups[P]),'--used:',U+C,'\n--left:',len(groups[P])-U-C)

                    total=total+C
                   # print (total)
                    U=U+C
                    A=len(groups[P])-U
                    F=F+1
                    if F<mf:
                        C=N-len(Final[F])                    

    #--------------------------------------------end for loop main-------------------------------------

if args.mode==8:
    #----write file
    ot=open('finals','w+')
    
    for gpi in range(len(Final)):
        ot.write('-------------\n')
        ot.write('-------------\n')
        ot.write('\n')
        ot.write('group :')
        ot.write(str(gpi+1))
        ot.write('\n')
        ot.write('total number :')
        ot.write(str(len(Final[gpi])))       
        ot.write('\n')

        ot.write(str(Final[gpi]))
        ot.write('\n')
        ot.write('\n')
    ot.close()


    #-------------draw--

    pd=cd[:lines]
    plt.scatter(pd[:,0],pd[:,1],s=0.01)
    for gi in range(41):
        plt.scatter(Final[gi][0][1],Final[gi][0][2],s=5,color=colors[0%(len(colors))])

    #---------------

    
    plt.show()

if args.mode==9:
    hdfile=open(args.headstar)
    q=hdfile.read()
    ori=open(args.ori)
    context=ori.read()
    sp_cont=context.split('\n')
    #print(len(sp_cont))
   # --------
    for xi in range(len(Final[0])):
    #for xi in range(10):
        filename='./Group/group_'+str(xi)+'.star'
        opt=open(filename,'w+') 
        opt.write(q)
        for wi in range(41):
            index=Final[wi][xi][0]
            opt.write(sp_cont[index+33])
            opt.write('\n')
        opt.close()
    
        
 




























