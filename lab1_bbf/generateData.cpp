#include<cstdio>
#include<algorithm>
#include<ctime>
#include<cstdlib> // for atoi
#include<cstring> // for strcmp
using namespace std;

void print_usage() {
    printf("Usage: generateData [options]\n");
    printf("Options:\n");
    printf("  -T <number>   : Number of data files to generate (default: 10)\n");
    printf("  -n <number>   : Number of data points in each file (default: 100000)\n");
    printf("  -m <number>   : Number of query points in each file (default: 100)\n");
    printf("  -d <number>   : Dimensions of the data (default: 8)\n");
    printf("  -prefix <str> : Prefix for output filenames (default: 'data')\n");
    printf("  -h, --help    : Show this help message\n");
}

int main(int argc, char* argv[]){
	// Default values
	int T = 10;     // The number of cases
	int n = 100000; // The number of points
	int m = 100;    // The number of queries 
	int d = 8;      // The number of dimensions
	char prefix[100] = "data"; // Default prefix
	
	// Parse command line arguments
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
			T = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
			n = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
			m = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
			d = atoi(argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "-prefix") == 0 && i + 1 < argc) {
			strcpy(prefix, argv[i + 1]);
			i++;
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			print_usage();
			return 0;
		}
	}
	
	printf("Generating %d files with %d points, %d queries, %d dimensions\n", T, n, m, d);
	
	const int mod = 1e9;
	srand((int)time(0));
	char filename[100];
	
	for (int t = 1; t <= T; t++) {
		sprintf(filename, "./data/%s_%dD_%d.txt", prefix, d, t);
		printf("Generating file %s...\n", filename);
		
		FILE* file = fopen(filename, "w");
		if (!file) {
			printf("Error: Cannot open file %s for writing\n", filename);
			return 1;
		}
		
		fprintf(file, "%d %d %d\n", n, m, d);
		
		// Generate n data points
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= d; j++) {
				long long num = ((long long)rand() * RAND_MAX + rand()) % mod;
				if (j == d)
					fprintf(file, "%lld\n", num);
				else
					fprintf(file, "%lld ", num);
			}
		}
		
		// Generate m query points
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= d; j++) {
				long long num = ((long long)rand() * RAND_MAX + rand()) % mod;
				if (j == d)
					fprintf(file, "%lld\n", num);
				else
					fprintf(file, "%lld ", num);
			}
		}
		
		fclose(file);
	}
	
	printf("Data generation complete.\n");
	return 0;
}
