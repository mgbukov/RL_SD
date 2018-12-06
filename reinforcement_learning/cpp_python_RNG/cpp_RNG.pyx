# distutils: language=c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

cimport cython
import numpy as _np
cimport numpy as _np

from libcpp.vector cimport vector


ctypedef fused int_double:
    _np.int64_t
    _np.int32_t
    _np.int_t
    cython.int
    cython.double
    


cdef extern from "RNG.h":
    cdef cppclass RNG_c:
        RNG_c() except +
 
        # Seed the random number generator 
        void seed(unsigned int) nogil
 
        # A uniform random sample from the open interval (0, 1) 
        double uniform(double, double) nogil

        vector[double] uniform(double, double, unsigned int) nogil
 
        # A uniform random sample from the set of unsigned integers 
        unsigned int GetUint() nogil
 
        # Normal (Gaussian) random sample 
        double normal(double, double) nogil

        # Integer within range
        long long randint(long long, long long) nogil

        # choose randomly a component of vector

        int choice(vector[int] vec) nogil
        double choice(vector[double] vec) nogil

        vector[int] choice(vector[int] vec,unsigned int size)
        vector[double] choice(vector[double] vec,unsigned int size)

        void choice[T](T*, T*,unsigned int vec_size) nogil
        void choice[T](T*, T*,unsigned int vec_size, unsigned int size) nogil
        
        

cdef class cpp_RNG:
    
    cdef RNG_c* rnd_gen_c # hold a C++ instance
    
    def __cinit__(self):
        self.rnd_gen_c = new RNG_c()
    
    def __dealloc__(self):
        del self.rnd_gen_c
     


    def seed(self, unsigned int u):
        self.rnd_gen_c.seed(u)
 
    # A uniform random sample from the open interval (0, 1) 
    def uniform(self,double low=0,double high=1,size=1):
        if size>1:
            return _np.array( self.rnd_gen_c.uniform(low,high,size) )
        else:
            return self.rnd_gen_c.uniform(low,high)
 
    # A uniform random sample from the set of unsigned integers 
    def GetUint(self):
        return self.rnd_gen_c.GetUint()
 
    # Normal (Gaussian) random sample 
    def normal(self, double mean=0.0, double std_dev=1.0):
        return self.rnd_gen_c.normal(mean, std_dev)

    # Integer within range
    def randint(self, long long minimum, long long maximum):
        return self.rnd_gen_c.randint(minimum,maximum)


    cpdef choice(self, int_double[:] vec, unsigned int size=1):
        vec_size = vec.shape[0]
        #_np.ndarray[int_double,ndim=1,mode="c"] out=_np.zeros((size,),dtype=int_double) # problem with dtype in np.zeros
        cdef vector[int_double] out

        out.resize(size)

        if size==0:
            return out
        elif size==1:
            self.rnd_gen_c.choice(&out[0],&vec[0],vec_size)
            return out[0]
        else:
            self.rnd_gen_c.choice(&out[0],&vec[0],vec_size,size)
            return _np.array(out)
            


 # use buffer symbol fused_type[:] an pass in reference
