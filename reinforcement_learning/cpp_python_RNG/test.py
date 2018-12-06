from cpp_RNG import cpp_RNG
import numpy as np

seed=0

RNG = cpp_RNG()
print(RNG)
RNG.seed(seed) # set the seed
print( RNG.uniform() ) # remember this number
print( RNG.uniform(-1,1) )

RNG.uniform()
RNG.seed(seed) # reset seed 
print(RNG.uniform() ) # returns the same number as above


print(RNG.GetUint())
print(RNG.GetUint())
print(RNG.GetUint())
print(RNG.GetUint())

print()

print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))
print(RNG.randint(0,10))

print()
vec=np.array([0.1,2.3,5.5],dtype=np.double)
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))
print(RNG.choice(vec))

print()

print(RNG.choice(vec,size=3))
print(RNG.choice(vec,size=3))
print(RNG.choice(vec,size=3))
print(RNG.choice(vec,size=3))
print(RNG.choice(vec,size=3))

