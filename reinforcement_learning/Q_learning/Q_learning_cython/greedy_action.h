//#include <cmath>
#include <algorithm>
#include <vector>


std::vector<int> greedy_indices(std::vector<double> Q) {
	std::vector<int> j_max;
	//std::vector<double>::iterator Q_max;
	double Q_max;
	unsigned int j;

	Q_max= *std::max_element( Q.begin(),Q.end() );
	

	// compute indices of all max instances in Q
	for(j=0;j<Q.size();j++){
		if(std::abs(Q[j]-Q_max)<1e-132){
			j_max.push_back(j);
		};
	};

	return j_max;
};

template<class T>
std::vector<T> evaluate_vector(std::vector<T> vec, std::vector<int> ind_vec) {
	unsigned int j;
	std::vector<T> new_vec;

	for(j=0;j<ind_vec.size();j++){
		new_vec.push_back(vec[ind_vec[j]] );
	};
	return new_vec;
};

