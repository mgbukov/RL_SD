#include <vector>

template <class T>
class environment_object {
	public:
		T initial,previous,current;
		std::vector<T> visited;
		
		environment_object();

		environment_object(int size, T initial_value) {
			initial=initial_value;
			previous=initial_value;
			current=initial_value;
			visited.resize(size);
		};

		~environment_object();

		void reset(T initial_value) {
			initial=initial_value;
			previous=initial_value;
			current=initial_value;
			visited.clear();
		};
};

