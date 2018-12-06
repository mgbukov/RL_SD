import numpy as np 
from Q_struct_cpp import algebraic_dict
#from Q_dokmat import algebraic_dict
from copy import deepcopy

Q=algebraic_dict(8,2)
P=algebraic_dict(8,2) #Q.copy()


'''
print(Q[3,13,0] )
print(Q[3,13,-1] )
print(Q[3,13,:])
'''


Q[3,6,0]=3.0
Q[3,6,-1]=4.0
Q[3,6,:]=np.array([-1.0,2.0])
Q[4,10,:]=np.array([1.0,-2.0])
'''
print( [3,13,0] )
print(Q[3,13,-1] )
print(Q[3,13,:])
'''

print(Q)


h0='10101011'
h1='10101010'
h2='10111'
h3='111'

print(Q[len(h0),int(h0,2),:])
print()

Q[len(h0),int(h0,2),:]=np.array([0.8,0.9])
Q[len(h1),int(h1,2),:]=np.array([0.1,0.0])
Q[len(h2),int(h2,2),:]=np.array([0.0,0.7])

P[len(h1),int(h1,2),:]=np.array([0.1,0.0])
P[len(h2),int(h2,2),:]=np.array([0.0,0.7])
P[len(h3),int(h3,2),:]=np.array([0.0,0.3])

print(Q)
Q=Q*2
print(Q)

#print(Q[h0])
Q=Q+1.0
exit()
#print(Q[h0])

S=Q+P
#print(S)


S=Q*P
print(S)




