# Import des modules nécessaires

import numpy as np
import scipy.sparse.csgraph
from heapq import *


# Graph relatif au réseau routier


G1 = (25,[  
[[1,2],[1/75,6.17]] , 
[[1,6],[1/100,9.26]] , 
[[2,3],[1/75,6.51]] ,
[[2,5],[1/85,2.57]],
[[3,4],[1/100,2.31]],
[[4,5],[1/75,5.14]],
[[4,13],[1/100,7.29]],
[[5,6],[1/75,3.26]],
[[5,12],[1/85,6.09]],
[[6,8],[1/100,4.29]],
[[6,7],[1/75,3.69]],
[[7,8],[1/85,4.63]],
[[8,9],[1/100,0.6]],
[[9,10],[1/85,3.67]],
[[9,11],[1/85,1.11]],
[[10,16],[1/50,1.2]],
[[11,12],[1/50,0.72]],
[[11,16],[1/50,3.48]],
[[12,17],[1/50,3]],
[[12,15],[1/75,2.76]],
[[13,14],[1/125,3.12]],
[[13,15],[1/85,3.36]],
[[14,17],[1/50,1.2]],
[[15,17],[1/75,0.6]],
[[16,17],[1/50,1.02]],
   ])

# Fonctions auxilliaires pour le calcul des pondérations des routes

def Troute(G,a,b):
    for i in range(len(G1[1])):
        if G1[1][i][0][0]==a :
            if G1[1][i][0][1]==b:
                return(G1[1][i][1][0],G1[1][i][1][1])
    else:
        return("les routes de sont pas reliés")

def evalTroute(G,a,b,x):
    return(Troute(G,a,b)[0]*x+Troute(G,a,b)[1])


def Eroute(G,a,b,x):
    s=0
    for i in range(1,x):
        s+=evalTroute(G,a,b,i)
    return(s)


# Calcul de la matrice d'adjacence

def produitMatrices(M1,M2):
    n=len(M1)
    P=[[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            P[i][j]=sum([M1[i][k]*M2[k][j] for k in range(n)])
    return(P)

def puissanceMatrices(M,k):
    n=len(M)
    P=[[0 for j in range(n)] for i in range(n)]
    for i in range(n) : P[i][i]=1
    for i in range(k):
        P=produitMatrices(P,M)
    return(P)
    
def matriceAdjacence(G):
    n = G[0]
    M = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
            M[i][i]=0
    for a in G[1]:
        M[a[0][0]][a[0][1]]=evalTroute(G1,a[0][0],a[0][1],0)
    return (np.array(M))


# Calcul du plus court chemin

def dijkstra (s, t, voisins):
    M = set()
    d = {s: 0}
    p = {}
    suivants = [(0, s)] # tas de couples (d[x],x)

    while suivants != []:

        dx, x = heappop(suivants)
        if x in M:
            continue

        M.add(x)

        for w, y in voisins(x):
            if y in M:
                continue
            dy = dx + w
            if y not in d or d[y] > dy:
                d[y] = dy
                heappush(suivants, (dy, y))
                p[y] = x

    path = [t]
    x = t
    while x != s:
        x = p[x]
        path.insert(0, x)

    return d[t], path 
    
def voisins (s):
    return graph[s]
   
def chgt(M):
    graph={}
    n=len(M[0])
    for i in range(n):
        graph[i]=[]
    for i in range(n):
        for j in range(n):
            
            if M[i][j]!=0:
                graph[i].append((M[i][j],j))
    return(graph)

# calcule du temps de parcours

nbv=int(input("Nombre de véhicules"))

def temps(a,b):
    global graph
    global nbvdanschaqueroute
    global itineraires
    M=matriceAdjacence(G1)
    nbvdanschaqueroute=[[(i,k),0] for i in range(len(M)) for k in range(i+1,len(M))]
    s=0
    matricesdadjacence=[M]
    itineraires={}
    for i in range(nbv):
        graph=chgt(matricesdadjacence[i])  
        it=dijkstra(a,b,voisins)[1]
        if tuple(it) in itineraires :
            itineraires[tuple(it)]+=1
        else : 
            itineraires[tuple(it)]=1
        Msuivant=M
        s2=s
        for k in range(len(it)-1):
            
            s+=M[it[k]][it[k+1]]
            Msuivant[it[k]][it[k+1]]=evalTroute(G1,it[k],it[k+1],i)
            matricesdadjacence.append(Msuivant)
            
            for route in nbvdanschaqueroute :
                if route[0]==(it[k],it[k+1]):
                    route[1]+=1


    return(s/nbv)

def nash(a,b):
    temps(a,b)
    s=0
    for route in nbvdanschaqueroute :
        s+=Eroute(G1,route[0][0],route[0][1],route[1])
    return(s)

def tempsreel(a,b):
    temps(a,b)
    listedestemps=[]
    for i in itineraires:
        s=0
        for k in range(len(i)-1):
            s+=evalTroute(G1,i[k],i[k+1],itineraires[i])
        listedestemps.append([s,itineraires[i]])
    for i in listedestemps :
        print(i[1], "vehicule(s) arrivent en", i[0], "minutes")
    sum1,sum2=0,0
    for i in listedestemps :
        sum1+=i[1]*i[0]
        sum2+=i[1]
    moyenne=sum1/sum2
    print("en moyenne un véhicule met",moyenne,"minutes à arriver")
        
    return
    
            
        


# print(scipy.sparse.csgraph.floyd_warshall(matriceAdjacence(G1)))
# print(scipy.sparse.csgraph.shortest_path(matriceAdjacence(G1)))
# print(scipy.sparse.csgraph.dijkstra(matriceAdjacence(G1)))
#nouveau commit

tempsreel(1,17)