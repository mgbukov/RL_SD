#include <math.h>
#include <vector>
#include <iostream>
#include <random>

class RNG_c{

    private:
        std::random_device rd;
        std::mt19937 gen;

	public:
		RNG_c() {
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        };

		~RNG_c(){};
  
		void seed(unsigned int u) {
            gen.seed(u); 
        };


        // This is the heart of the generator.
        unsigned int GetUint() {
            return gen();
        };

        // Produce a uniform random sample from the open interval (0, 1).
        double uniform() {
            std::uniform_real_distribution<double> unif(0,1);
            return unif(gen);
        };

        double uniform(double low, double high) {
            std::uniform_real_distribution<double> unif(low, high);
            return unif(gen);
        };

        std::vector<double> uniform(double low, double high, unsigned int size) {
            unsigned int i;
            std::vector<double> vec;
            vec.resize(size);
            std::uniform_real_distribution<double> unif(low, high);

            for(i=0;i<size;i++){
                vec[i]=unif(gen);
            };
            
            return vec;
        };

        // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
        double normal() {
            // Use Box-Muller algorithm
            double u1 = uniform();
            double u2 = uniform();
            double r = sqrt( -2.0*log(u1) );
            double theta = 2.0*M_PI*u2;
            return r*sin(theta);
        };

        // Get normal (Gaussian) random sample with specified mean and standard deviation
        double normal(double mean, double standardDeviation) {
            return mean + standardDeviation*normal();
        };

        // (slightly biased) integer within range
        // https://ericlippert.com/2013/12/16/how-much-bias-is-introduced-by-the-remainder-technique/
        unsigned long randint(long min, long max) {
            std::uniform_int_distribution<> dis(min, max);
    		return dis(gen);
        };

        template <class T>
   		std::vector<T> choice(std::vector<T> vec,unsigned int size) {
            std::vector<T> chosen;
   			unsigned int u,i;

            chosen.resize(size);

            for(i=0;i<size;i++){
                u = randint(0,vec.size()-1);
                chosen[i]=vec[u];
                //std::cout << i <<' ' << vec[u] << std::endl;
            };

   			return chosen;
   		};

        
        template <class T>
        T choice(std::vector<T> vec) {
            unsigned int u = randint(0,vec.size()-1);

            // std::cout << u <<' ' << vec[u] << std::endl;
            return vec[u];
        };

        
        template <class T>
        void choice(T * chosen,T * vec,unsigned int vec_size, unsigned int size) {
            unsigned int u,i;

            for(i=0;i<size;i++){
                u = randint(0,vec_size-1);
                chosen[i]=vec[u];
                //std::cout << i <<' ' << vec[u] << std::endl;
            };

        };
        
        template <class T>
        void choice(T * chosen,T * vec, unsigned int vec_size) {
            unsigned int u = randint(0,vec_size-1);

            // std::cout << u <<' ' << vec[u] << std::endl;
            chosen[0]=vec[u];
        };

};
