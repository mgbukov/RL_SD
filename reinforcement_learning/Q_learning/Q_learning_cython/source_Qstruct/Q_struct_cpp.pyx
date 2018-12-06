# distutils: language=c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

import numpy as _np
cimport numpy as _np

from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator import preincrement as inc

#from libc.stdint cimport uint64_t


# multiplication between dict and dict
cdef void mul_map_map(unordered_map[string,vector[double]] &cpp_dict_1,
					  unordered_map[string,vector[double]] &cpp_dict_2,
					  unordered_map[string,vector[double]] &cpp_dict_out) nogil:
	cdef int i
	for ele in cpp_dict_1: # automatically does fast loop
		if cpp_dict_2.count(ele.first):
			cpp_dict_out[ele.first].resize(ele.second.size())
			for i in range(ele.second.size()):
				cpp_dict_out[ele.first][i] = cpp_dict_2[ele.first][i] * ele.second[i]

# inplace add between two dicts
cdef void iadd_map_map(unordered_map[string,vector[double]] &cpp_dict_1,
				   	   unordered_map[string,vector[double]] &cpp_dict_2) nogil:

	cdef int i
	for ele in cpp_dict_2: # automatically does fast loop
		if cpp_dict_1.count(ele.first):
			for i in range(ele.second.size()):
				cpp_dict_1[ele.first][i] += ele.second[i]
		else:
			cpp_dict_1[ele.first].resize(ele.second.size())
			for i in range(ele.second.size()):
				cpp_dict_1[ele.first][i] = ele.second[i]

# inplace add btw dict and scalar
cdef void iadd_map_scalar(unordered_map[string,vector[double]] &cpp_dict,double other) nogil:
	cdef int i
	for ele in cpp_dict: # automatically does fast loop
		for i in range(ele.second.size()):
			cpp_dict[ele.first][i] += other

# inplace multiplication between dict and scalar
cdef void imul_map_scalar(unordered_map[string,vector[double]] &cpp_dict,double other) nogil:
	cdef int i
	for ele in cpp_dict: # automatically does fast loop
		for i in range(ele.second.size()):
			cpp_dict[ele.first][i] *= other




# algebraic dictionary class
cdef class algebraic_dict:
	
	# declare all self's
	cdef unordered_map[string,vector[double]] cpp_dict # instead of string use a vec of of bools
	cdef vector[unordered_map[string,vector[double]] ] Q_struct
	cdef public unsigned int N_actions, N_time_steps, ndim
	cdef object dtype
	
	def __init__(self, int N_time_steps,int N_actions):

		self.dtype=_np.float64 # data type
		self.N_actions=N_actions # actions dimenion
		self.N_time_steps=N_time_steps
		self.ndim=3 # (time_slice, protocol-integer, actions)
		
		# assign a cpp dict for every time step
		for j in range(self.N_time_steps+1):
			self.Q_struct.push_back(self.cpp_dict)


	cdef void reset_c(self) nogil:
		cdef int j=0
		while j<self.N_time_steps+1:
			self.Q_struct[j].clear()
			inc(j)

	cpdef void reset(self):
		self.reset_c()
	
	
	cpdef copy(self):
		'''Creates copy of class object.'''

		cdef int j
		cdef algebraic_dict new=algebraic_dict(self.N_time_steps,self.N_actions)

		for j in range(self.N_time_steps):
			new.Q_struct[j] = self.Q_struct[j]

		return new	

	

	def __str__(self):
		'''Prints class object.'''
		string="\n".join(" ") 
		for j in range(self.N_time_steps+1):
			if not self.Q_struct[j].empty():
				str_list = ["{}:{}:{}".format(j, str(ele.first),list(ele.second)) for ele in self.Q_struct[j]] 
				str_list.append(" ")
				string += "\n".join(str_list)
				
		
		return string
	
	def __getitem__(self, index):
		cdef string ind
		cdef int ind_a,i,start,stop,step,time_slice
		cdef _np.ndarray[double,ndim=1] out = _np.zeros(self.N_actions,dtype=_np.float64)

		if len(index) != 3:
			raise IndexError('Q_struct object must be indexed by three integers')

		time_slice=index[0]
		if time_slice > self.N_time_steps:
			raise IndexError('First index of Q_struct cannot exceed {}'.format(self.N_time_steps))

		index=index[1:]

		if isinstance(index, tuple):

			#fetch string
			ind=index[0]

			if isinstance(index[1], slice):
				start, stop, step = index[1].indices(self.N_actions)
				if self.Q_struct[time_slice].count(ind):
					for i in range(self.N_actions):
						out[i]=self.Q_struct[time_slice][ind][i]

				return out[start:stop:step]
			else:

				ind_a = index[1] # cython does check for integer automatically
				ind_a = ind_a%self.N_actions
			
				if self.Q_struct[time_slice].count(ind):
					return self.Q_struct[time_slice][ind][ind_a]
				else:
					return 0.0			
		else:
			#fetch string
			ind=index

			if self.Q_struct[time_slice].count(ind):
				for i in range(self.N_actions):
					out[i]=self.Q_struct[time_slice][ind][i]

			return out
	
	def __setitem__(self,index,value):
		cdef string ind
		cdef int ind_a,i,j,start,stop,step,time_slice

		if len(index) != 3:
			raise IndexError('Q_struct object must be indexed by three integers')

		time_slice=index[0]
		if time_slice > self.N_time_steps:
			raise IndexError('First index of Q_struct cannot exceed {}'.format(self.N_time_steps))


		index=index[1:]

		value = _np.asarray(value)
		
		if isinstance(index, tuple):
			# fetch string
			ind=index[0]

			if self.Q_struct[time_slice].count(ind) == 0:
				self.Q_struct[time_slice][ind].resize(self.N_actions)

			if isinstance(index[1], slice):

				start, stop, step = index[1].indices(self.N_actions)
				if value.ndim == 0:
					self.set_slice_scalar(time_slice,ind,start,stop,step,value)
				else:
					self.set_slice_buffer(time_slice,ind,start,stop,step,value)
					
			else:
				ind_a=index[1]
				ind_a=ind_a%self.N_actions
				if value.ndim == 0:
					self.Q_struct[time_slice][ind][ind_a]=value
				else:
					raise ValueError("can't set single element to array_like object")

		else:
			# fetch string
			ind=index

			if self.Q_struct[time_slice].count(ind) == 0:
				self.Q_struct[time_slice][ind].resize(self.N_actions)

			if value.ndim == 0:
				self.set_scalar(time_slice,ind,value)
			else:
				self.set_buffer(time_slice,ind,value)			
	
	cdef void set_slice_scalar(self,int time_slice,string ind,int start,int stop,int step,double value) nogil:
		cdef int j=0
		while(j<stop):
			self.Q_struct[time_slice][ind][j]=value
			j += step
	
	cdef void set_slice_buffer(self,int time_slice,string ind,int start,int stop,int step,double[:] value) nogil:
		cdef int j=0
		cdef int i=0
		while(j<stop):
			self.Q_struct[time_slice][ind][j]=value[i]
			inc(i)
			j+=step

	cdef void set_scalar(self,int time_slice,string ind,double value) nogil:
		cdef int i=0
		for i in range(self.N_actions):
			self.Q_struct[time_slice][ind][i]=value

	cdef void set_buffer(self,int time_slice,string ind,double[:] value) nogil:
		cdef int i=0
		for i in range(self.N_actions):
			self.Q_struct[time_slice][ind][i]=value[i]
	
	
	cdef vector[double] evaluate_c(self, unsigned int time_slice, string ind) nogil:
		cdef unsigned int i
		cdef vector[double] Q
		
		Q.resize(self.N_actions)
		i=0
		if self.Q_struct[time_slice].count(ind):
			while i < self.N_actions:
				Q[i]=self.Q_struct[time_slice][ind][i]
				inc(i) 

		return Q

	cpdef vector[double] evaluate(self, unsigned int time_slice, string ind):
		return self.evaluate_c(time_slice,ind)


	cdef void set_value_c(self, unsigned int time_slice, string ind, unsigned int indA, double value) nogil:

		if self.Q_struct[time_slice].count(ind) == 0:
				self.Q_struct[time_slice][ind].resize(self.N_actions)

		self.Q_struct[time_slice][ind][indA]=value

	cpdef void set_value(self, unsigned int time_slice, string ind, unsigned int indA, double value):
		self.set_value_c(time_slice,ind,indA,value)




	
	# addition between two dicts
	cdef algebraic_dict algebraic_dict_addition(self,algebraic_dict other):
		cdef algebraic_dict new = self.copy()
		cdef int j
		
		for j in range(self.N_time_steps):
			iadd_map_map(new.Q_struct[j],other.Q_struct[j])
		
		return new

	cdef algebraic_dict inplace_algebraic_dict_addition(self,algebraic_dict other):
		cdef int j
		
		for j in range(self.N_time_steps):
			iadd_map_map(self.Q_struct[j],other.Q_struct[j])
		
		return self

	# addition between scalar and dict
	cdef algebraic_dict scalar_addition(self,double other):
		cdef algebraic_dict new=self.copy()
		cdef int j
		
		for j in range(self.N_time_steps):
			iadd_map_scalar(new.Q_struct[j],other)
		return new

	cdef algebraic_dict inplace_scalar_addition(self,double other):
		cdef int j
		
		for j in range(self.N_time_steps):
			iadd_map_scalar(self.Q_struct[j],other)
		return self

	# mulltiplication between two dict objects
	cdef algebraic_dict algebraic_dict_multiplication(self,algebraic_dict other):
		cdef algebraic_dict new = algebraic_dict(self.N_time_steps,self.N_actions)
		cdef int j
		
		for j in range(self.N_time_steps):
			mul_map_map(self.Q_struct[j],other.Q_struct[j],new.Q_struct[j])
		return new

	# multiplication between scalar and dict
	cdef algebraic_dict scalar_multiplication(self,double other):
		cdef algebraic_dict new = self.copy()
		cdef int j
		
		for j in range(self.N_time_steps):
			imul_map_scalar(new.Q_struct[j],other)
		return new

	cdef algebraic_dict inplace_scalar_multiplication(self,double other):
		cdef int j
		
		for j in range(self.N_time_steps):
			imul_map_scalar(self.Q_struct[j],other)

		return self

	# combined addition function
	def _addition(self,other):
		if _np.isscalar(other):
			return self.scalar_addition(other)
		elif isinstance(other,algebraic_dict):
			return self.algebraic_dict_addition(other)
		else:
			return NotImplemented

	# combined mutiplication function
	def _multiplication(self,other):
		if _np.isscalar(other):
			return self.scalar_multiplication(other)
		elif isinstance(other,algebraic_dict):
			return self.algebraic_dict_multiplication(other)
		else:
			return NotImplemented


	# combined addition function
	def _inplace_addition(self,other):
		if _np.isscalar(other):
			return self.inplace_scalar_addition(other)
		elif isinstance(other,algebraic_dict):
			return self.inplace_algebraic_dict_addition(other)
		else:
			return NotImplemented

	# combined mutiplication function
	def _inplace_multiplication(self,other):
		return self.inplace_scalar_multiplication(other)
		'''
		if _np.isscalar(other):
			return self.inplace_scalar_multiplication(other)
		elif isinstance(other,algebraic_dict):
			return NotImplemented
		else:
			return NotImplemented
		'''

	# addition method
	def __add__(x,y):
		try:
			return x._addition(y)
		except AttributeError:
			return y._addition(x)

	# multiplication method
	def __mul__(x,y):
		try:
			return x._multiplication(y)
		except AttributeError:
			return y._multiplication(x)

	# addition method
	def __iadd__(self,y):
		return self._inplace_addition(y)

	# multiplication method
	def __imul__(self,y):
		return self._inplace_multiplication(y)
