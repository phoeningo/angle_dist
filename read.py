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
#for 20191122_YCY,DEFAULT OFFSET=34

parser=argparse.ArgumentParser(description='T')
parser.add_argument("--input",type=str)
parser.add_argument('--ang_rot',type=int,default=2)
parser.add_argument('--ang_tilt',type=int,default=3)
parser.add_argument('--s',type=float,default=0.01)
parser.add_argument("--headstar",type=str,default='head.star')
parser.add_argument("--ori",type=str)
parser.add_argument('--show',type=int,default=0)
parser.add_argument("--center",type=str)
parser.add_argument("--mode",type=int,default=0)
parser.add_argument("--headnum",type=int,default=29)
parser.add_argument("--size",type=int,default=41)
parser.add_argument("--output",type=str,default='out.log')
parser.add_argument('--offset',type=int,default=33)
parser.add_argument('--thre',type=int,default=3000)
parser.add_argument('--percent',type=float,default=0.5)
parser.add_argument('--Bsize',type=int,default=30)
parser.add_argument('--tmp',type=str,default='groups')
parser.add_argument('--fade',type=float,default=0.25)


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
                    if l==args.ang_rot:
                    #print(sp[i])
                        cd[tk,0]=float(sp[i])
                        cd[tk+lines,0]=cd[tk,0]+360
                    if l==args.ang_tilt:
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

if args.mode>=2 :
    '''
    lines=0
    while sr:
        sp=sr.split(' ')
        if len(sp)>=args.headnum:
            lines=lines+1
            #print(sp)
        sr=starfile.readline()
    cd=np.zeros(shape=(lines*2,2),dtype=np.float32)
    starfile.close()
    starfile=open(args.input)
    sr=starfile.readline()
    tk=0
    while sr:
        if len(sr.split(' ')>args.head_num):
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
    '''
    lines=0
    while sr:
        sp=sr.split(' ')
        #print(len(sp))
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
                    if l==args.ang_rot:
                    #print(sp[i])
                        cd[tk,0]=float(sp[i])
                        cd[tk+lines,0]=cd[tk,0]+360
                    if l==args.ang_tilt:
                        cd[tk,1]=float(sp[i])
                        cd[tk+lines,1]=cd[tk,1]
                        tk=tk+1
        sr=starfile.readline()
    starfile.close()
    #print(cd[0])
    if args.show==1:
        plt.scatter(cd[0:lines,0],cd[0:lines,1],s=args.s)
        plt.show()



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

if args.mode>=7:
    gp=0
    groups=[]
    heads=[]
    Lrange=int(360/args.Bsize)
    print(Lrange)
    Srange=Lrange/2
    Bsize=args.Bsize
    while gp <=Lrange:
        heads.append([0])
        gp=gp+1

    gp=0
    while gp <=Lrange*Srange:
        groups.append([0])
        gp=gp+1

    for i in range (lines):
        heads[int(int((cd[i,0]+180)/Bsize-1)%(Lrange+1))].append((i,cd[i,0],cd[i,1]))
    
    for i in range(Lrange):
        for hi in range(len(heads[i])-1):
            groups[int(i*Srange)+int(int(heads[i][hi+1][2]/Bsize))].append(heads[i][hi+1])


if args.mode==7:
    ot=open(args.tmp,'w+')
    
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

    if args.show==1:
        pd=cd[:lines]
        plt.scatter(pd[:,0],pd[:,1],s=0.01)
        for gi in range(Srange*Lrange):
            plt.scatter(groups[gi][len(groups[gi])-1][1],groups[gi][len(groups[gi])-1][2],s=5,color=colors[gi%(len(colors))])
            plt.scatter(groups[gi][1][1],groups[gi][1][2],s=5,color=colors[gi%(len(colors))])
    
        plt.show()
    
if args.mode>=8 and args.mode <10:
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
        plt.scatter(Final[gi][0][1],Final[gi][0][2],s=5,color=colors[int(0%(len(colors)))])

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
            opt.write(sp_cont[index+args.offset])
            opt.write('\n')
        opt.close()
    
if args.mode==11:
    print('total '+str(len(groups))+' groups')
    for gpi in range(len(groups)):
        print(len(groups[gpi]))

if args.mode==10:
    hdfile=open(args.headstar)
    q=hdfile.read()
    ori=open(args.ori)
    context=ori.read()
    sp_cont=context.split('\n')
    opt=open(args.output,'w+')
    opt.write(q)
    for gpi in range(len(groups)):
        L=len(groups[gpi])
       # print(L)
        if L>args.thre:
            for li in range(int(args.thre*args.percent)):
                index=groups[gpi][li+1][0]
                opt.write(sp_cont[index+args.offset])
                opt.write('\n')
        else:
            for li in range(L-1):
                index=groups[gpi][li+1][0]
                opt.write(sp_cont[index+args.offset])
                opt.write('\n')
    opt.close()


if args.mode==12:
    import random
    hdfile=open(args.headstar)
    q=hdfile.read()
    ori=open(args.ori)
    context=ori.read()
    sp_cont=context.split('\n')
    opt=open(args.output,'w+')
    opt.write(q)
    try:
        for gpi in range(len(groups)):
            L=len(groups[gpi])
            print(L)
            if L>args.thre:
                for li in range(int(args.thre*args.percent)):
                    index=groups[gpi][li+1][0]
                
                    opt.write(sp_cont[index+args.offset])
                    opt.write('\n')
            else:
                
                #random.shuffle(groups[gpi])
                #print(len(groups[gpi]))
                L=len(groups[gpi])
                tmp_group=groups[gpi][1:L]
                #print(tmp_group)
                random.shuffle(tmp_group)
                for li in range(int(L*args.fade)): 
                    #index=groups[gpi][li+1][0]
                    index=tmp_group[li+1][0]
                    #print(sp_cont[index+args.offset])
                    opt.write(sp_cont[index+args.offset])
                    opt.write('\n')
  
       # Required by Zhu
    except:
        
        print("current_index",index,"\ncurrent_len",len(sp_cont))
    opt.close()


         
        




























