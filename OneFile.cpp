#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <climits>
#include <numeric>
#include <iomanip>
#include <complex>
#include <utility>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <queue>
#include <array>
#include <new>
#include <set>
#include <map>
#define MACROS
#ifdef  MACROS
#define int long long
#define ull unsigned long long
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define repeat(j)      for (int i = 0; i <  j; i++)
#define iter(i, a, b)  for (int i = a; i <= b; i++)
#define iterzero(i, j) for (int i = 0; i <  j; i++)
#define iterone(i, j)  for (int i=1;i<=j;i++)
#define itervec(i, j)  for (auto& i : j)
#define square(i)      ((i) * (i))
#define input(i)       std::cin >> (i)
#define outprecision(i) std::cout << std::fixed << std::setprecision(i);
#define OPTIMIZE std::cin.tie(nullptr), std::cout.tie(nullptr), ios::sync_with_stdio(false)
#endif
namespace Algorithms
{
	class Sort
	{
	private:
		int GetMax(int* arr, int n)
		{
			int max = arr[0];
			for (int i = 1; i < n; i++)
				if (arr[i] > max)
					max = arr[i];
			return max;
		}
		void Merge(int* arr, int p, int q, int r)
		{
			int n1 = q - p + 1, n2 = r - q, i, j, k;
			int* L = new int[n1], * M = new int[n2];

			for (i = 0; i < n1; i++)
				L[i] = arr[p + i];
			for (j = 0; j < n2; j++)
				M[j] = arr[q + 1 + j];

			i = 0;
			j = 0;
			k = p;
			while (i < n1 && j < n2)
			{
				if (L[i] <= M[j])
				{
					arr[k] = L[i];
					i++;
				}
				else
				{
					arr[k] = M[j];
					j++;
				}
				k++;
			}
			while (i < n1)
			{
				arr[k] = L[i];
				i++;
				k++;
			}
			while (j < n2)
			{
				arr[k] = M[j];
				j++;
				k++;
			}
			delete[] L;
			delete[] M;
		}
		void MergeRecursion(int* arr, int l, int r)
		{
			// Time & Space std::complex<int>ity: O(n*log(n)) / O(n)
			if (l < r)
			{
				int m = l + (r - l) / 2;
				MergeRecursion(arr, l, m);
				MergeRecursion(arr, m + 1, r);
				Merge(arr, l, m, r);
			}
		}
		int Partition(int* arr, int low, int high)
		{
			int pivot = arr[high];
			int i = (low - 1);
			for (int j = low; j <= high - 1; j++)
			{
				if (arr[j] < pivot)
				{
					i++;
					std::swap(arr[i], arr[j]);
				}
			}
			std::swap(arr[i + 1], arr[high]);
			return i + 1;
		}
		void QuickRecursion(int* arr, int low, int high)
		{
			if (low < high)
			{
				int p = Partition(arr, low, high);
				QuickRecursion(arr, low, p - 1);
				QuickRecursion(arr, p + 1, high);
			}
		}
		void Heapify(int* arr, int N, int i)
		{
			int largest = i, l = 2 * i + 1, r = 2 * i + 2;
			if (l < N && arr[l] > arr[largest])
				largest = l;
			if (r < N && arr[r] > arr[largest])
				largest = r;
			if (largest != i)
			{
				std::swap(arr[i], arr[largest]);
				Heapify(arr, N, largest);
			}
		}
		void SortDigit(int* arr, int n, int exp, int base)
		{
			int* output = new int[n];
			int* count = new int[base];
			for (int i = 0; i < base; i++) count[i] = 0;
			for (int i = 0; i < n; i++)
				count[(arr[i] / exp) % base]++;
			for (int i = 1; i < base; i++)
				count[i] += count[i - 1];
			for (int i = n - 1; i >= 0; i--)
			{
				output[count[(arr[i] / exp) % base] - 1] = arr[i];
				count[(arr[i] / exp) % 10]--;
			}
			for (int i = 0; i < n; i++)
				arr[i] = output[i];
			delete[] output;
			delete[] count;
		}
	public:
		void BubbleSort(int* arr, const int& len)
		{
			// Time & Space std::complex<int>ity: O(n^2) / O(1)
			bool swapped;
			for (int i = 0; i < len - 1; i++)
			{
				swapped = false;
				for (int j = 0; j < len - i - 1; j++)
				{
					if (arr[j] > arr[j + 1])
					{
						std::swap(arr[j], arr[j + 1]);
						swapped = true;
					}
				}
				if (swapped == false)
					break;
			}
		}
		void SelectionSort(int* arr, const int& len)
		{
			// Time & Space std::complex<int>ity: O(n^2) / O(1)
			int i, j, min_idx;
			for (i = 0; i < len - 1; i++)
			{
				min_idx = i;
				for (j = i + 1; j < len; j++)
					if (arr[j] < arr[min_idx])
						min_idx = j;
				if (min_idx != i)
					std::swap(arr[min_idx], arr[i]);
			}
		}
		void InsertionSort(int* arr, const int& len)
		{
			// Time & Space std::complex<int>ity: O(n^2) / O(1)
			int i, key, j;
			for (i = 1; i < len; i++)
			{
				key = arr[i];
				for (j = i - 1; j >= 0 && arr[j] > key; j--)
					arr[j + 1] = arr[j];
				arr[j + 1] = key;
			}
		}
		void MergeSort(int* arr, const int& len)
		{
			MergeRecursion(arr, 0, len - 1);
		}
		void QuickSort(int* arr, const int& len)
		{
			QuickRecursion(arr, 0, len - 1);
		}
		void HeapSort(int* arr, const int& len)
		{
			for (int i = len / 2 - 1; i >= 0; i--)
				Heapify(arr, len, i);
			for (int i = len - 1; i > 0; i--)
			{
				std::swap(arr[0], arr[i]);
				Heapify(arr, i, 0);
			}
		}
		void RadixSort(int* arr, const int& len, const int& base = 10)
		{
			int m = GetMax(arr, len);
			for (int exp = 1; m / exp > 0; exp *= base)
				SortDigit(arr, len, exp, base);
		}
		void CountingSort(int* arr, const int& len)
		{
			int max = GetMax(arr, len);
			int* count = new int[max + 1], * output = new int[len];
			for (int i = 0; i <= max; i++)
				count[i] = 0;
			for (int i = 0; i < len; i++)
				count[arr[i]]++;
			for (int i = 1; i <= max; i++)
				count[i] += count[i - 1];
			for (int i = len - 1; i >= 0; i--)
			{
				output[count[arr[i]] - 1] = arr[i];
				count[arr[i]]--;
			}
			for (int i = 0; i < len; i++)
				arr[i] = output[i];
		}
	};
	class Knapsack
	{
	public:
		int Knapsack_01(int num, int capacity, int weight[], int value[])
		{
			std::vector<std::vector<int>> dp(num + 1, std::vector<int>(capacity + 1, 0));
			for (int i = 1; i <= capacity; i++)
			{
				for (int j = 1; j <= num; j++)
				{
					if (i >= weight[j - 1])
						dp[j][i] = max(dp[j - 1][i], dp[j - 1][i - weight[j - 1]] + value[j - 1]);
					else dp[j][i] = dp[j - 1][i];
				}
			}
			return dp[num][capacity];
		}
		int Knapsack_Full(int num, int capacity, int v[], int w[])
		{
			std::vector<std::vector<int>> dp(num + 1, std::vector<int>(capacity + 1));
			for (int i = 0; i <= capacity; i++)
				dp[0][i] = i / v[0] * w[0];
			for (int i = 1; i < num; i++)
			{
				for (int j = 0; j <= capacity; j++)
				{
					int max_choice = 0, x = 1;
					while (j >= v[i] * x)
						max_choice = max(max_choice, dp[i - 1][j - x * v[i]] + x * w[i]), x++;
					dp[i][j] = max(dp[i - 1][j], max_choice);
				}
			}
			return dp[num - 1][capacity];
		}
		int Knapsack_Many(int num, int capacity, int v[], int w[], int s[])
		{
			int dp[101] = { 0 };
			for (int i = 0; i < num; i++)
			{
				if (s[i] * v[i] >= capacity)
					for (int j = v[i]; j <= capacity; j++)
						dp[j] = max(dp[j - v[i]] + w[i], dp[j]);
				else for (int j = capacity; j >= v[i]; j--)
					for (int k = s[i]; k >= 0; k--)
						if (j >= k * v[i])
							dp[j] = max(dp[j - k * v[i]] + k * w[i], dp[j]);
			}
			return dp[capacity];
		}
	};
	class Prime
	{
	public:
		std::vector<int> Euler_Sieve(const int& n)
		{
			std::vector<int> primes;
			std::vector<bool> is_prime(n);
			for (int i = 2; i < n; i++)
				is_prime[i] = true;
			for (int i = 2; i < n; i++)
			{
				if (is_prime[i])
					primes.push_back(i);
				for (int j = 0; j < primes.size() && primes[j] * i < n; j++)
				{
					is_prime[primes[j] * i] = false;
					if (!(i % primes[j]))
						break;
				}
			}
			return primes;
		}
	};
	class Fast
	{
	private:
		void FFTCalculate(std::complex<double> A[], int ord[], int n)
		{
			// Given A[] = f(x) = a_0 * x^n + a_1 * x^{n-1} + ... + a_{n-1} * x^1 + a_n, Calculate for 0 <= x < n : f(omega ^ x), omega = e ^ (2 * pi * i / n)
			const double pi = acos(-1);
			for (int i = 1; i < n; i++) // Change A sequence's arrangement to ord[i]'s
				if (ord[i] > i)
					std::swap(A[i], A[ord[i]]);
			for (int len = 2; len <= n; len <<= 1)
			{
				std::complex<double> omega(cos(2 * pi / len), sin(2 * pi / len)); // nth roots of unity
				for (int l = 0, r = len - 1; r <= n; l += len, r += len)
				{
					std::complex<double> w(1.0, 0.0); // w = omega ^ k
					for (int p = l; p < l + len / 2; p++)
					{
						std::complex<double> x = A[p] + w * A[p + len / 2], y = A[p] - w * A[p + len / 2];
						A[p] = x; A[p + len / 2] = y;
						w *= omega;
					}
				}
			}
		}
	public:
		long long pow(long long a, long long b, long long mod)
		{
			long long res = 1;
			while (b)
			{
				if (b & 1) res = (res * a) % mod;
				a = (a * a) % mod;
				b >>= 1;
			}
			return res;
		}
		long long exgcd(long long a, long long b)
		{
			if (a < b) std::swap(a, b);
			if (b == 0) return a;
			if (b == 1) return 1;
			return exgcd(b, a % b);
		}
		long long Combination(long long a, long long b, long long mod = 1e9 + 7, long long MAXN = 1e6 + 7)
		{
			if (a < b) return 0;
			std::vector<long long> fac;
			fac.push_back(1);
			for (long long i = 1; i < MAXN; i++)
				fac.push_back(fac[i - 1] * i % mod);
			return fac[a] * pow(fac[b], mod - 2, mod) % mod * pow(fac[a - b], mod - 2, mod) % mod;
		}
		std::vector<int> FFT(std::vector<int> a, std::vector<int> b)
		{
			// Calculate a * b = output where a, b, output are polynomials.
			const int MAXN = max(a.size(), b.size()) << 3 + 7;
			std::complex<double>* F = new std::complex<double>[MAXN];
			std::vector<int> res;
			int* ord = new int[MAXN];
			for (int i = 0; i < a.size(); i++) F[i].real(a[i]);
			for (int i = 0; i < b.size(); i++) F[i].imag(b[i]);
			int n = a.size() - 1, m = b.size() - 1, num = 1;
			while (num <= n + m) num <<= 1;
			int d = num >> 1, p = 0;
			ord[p++] = 0; ord[p++] = d;
			for (int w = 2; w <= num; w <<= 1)
			{
				d >>= 1;
				for (int p0 = 0; p0 < w; p0++)
					ord[p++] = ord[p0] | d;
			}
			FFTCalculate(F, ord, num);
			for (int i = 0; i < num; i++)
				F[i] *= F[i];
			FFTCalculate(F, ord, num);
			std::reverse(F + 1, F + num);
			for (int i = 0; i <= n + m; i++)
				res.push_back((int)round(F[i].imag() / 2 / num));
			return res;
		}
		int read()
		{
			int out = 0;
			char x = getchar();
			while (x != ' ' && x != '\n')
			{
				out = 10 * out + x - '0';
				x = getchar();
			}
			return out;
		}
		long long readll()
		{
			long long out = 0;
			char x = getchar();
			while (x != ' ' && x != '\n')
			{
				out = 10 * out + x - '0';
				x = getchar();
			}
			return out;
		}
	};
};
namespace Main
{
	const double e = 2.718281828459045;
	const double tau = 6.283185307179586;
	const double pi = 3.141592653589793;
	inline static int remainder(const int& a, const int& b)
	{
		return a % b;
	}
	inline static int plus(const int& a, const int& b)
	{
		return a + b;
	}
	inline static int minus(const int& a, const int& b)
	{
		return a - b;
	}
	inline static int abs_minus(const int& a, const int& b)
	{
		return abs(a - b);
	}
	inline static int times(const int& a, const int& b)
	{
		return a * b;
	}
	inline static double float_section(const float& a)
	{
		return a - floor(a);
	}
	inline static double divide(const float& a, const float& b)
	{
		return a / b;
	}
	inline static double lg(const long long& a)
	{
		return log10(a);
	}
	inline static double ln(const long long& a)
	{
		return log(a);
	}
	inline static double square_root(const int& a)
	{
		return sqrt(a);
	}
	inline static double loga_to_base(const long long& a, const int& base)
	{
		return log(a) / log(base);
	}
	inline static int max_log(const int& a, const int& base)
	{
		return floor(loga_to_base(a, base));
	}
	inline static int max_pow(const int& a, const int& base)
	{
		return pow(a, max_log(a, base));
	}
	inline static int down_divide(const float& a, const float& b)
	{
		return floor(a / b);
	}
	inline static int up_divide(const float& a, const float& b)
	{
		return ceil(a / b);
	}
	inline static int fo_up_fi_down(const float& a)
	{
		if (float_section(a) < 0.5)
			return floor(a);
		else
			return ceil(a);
	}
	inline static int fo_up_fi_down_divide(const float& a, const float& b)
	{
		return fo_up_fi_down(a / b);
	}
	inline static int factor_count(const int& a)
	{
		// including 1 and a
		int count = 0;
		for (int i = 1; i * i <= a; i++)
		{
			if (a % i == 0)
				count++;
		}
		return 2 * count - pow(floor(sqrt(a)), 2) == a;
	}
	inline static int factor_sum(const int& a)
	{
		// including 1 and a
		int sum = 0, i;
		for (i = 1; i * i < a; i++)
		{
			if (a % i == 0)
				sum += (i + a / i);
		}
		if (i * i == a) sum += i;
		return sum;
	}
	inline static int collatz(const int& a)
	{
		if (a % 2 == 0)
			return (a / 2);
		else
			return (3 * a + 1);
	}
	inline static int sum(const int& len, const int a[])
	{
		int sum = 0;
		for (int i = 0; i < len; i++)
			sum += a[i];
		return sum;
	}
	inline static int mean(const int& len, const int a[])
	{
		return sum(len, a) / len;
	}
	inline static int median(const int& len, int a[])
	{
		std::sort(a, a + len);
		if (len % 2 == 1)
			return a[(len - 1) / 2];
		else
			return (a[len / 2] + a[len / 2 - 1]) / 2;
	}
	inline static int mode(const int& len, const int a[])
	{
		int maxvalue = 0, maxcount = 0;
		for (int i = 0; i < len; i++)
		{
			int count = 0;
			for (int j = 0; j < len; j++)
			{
				if (*(a + j) == *(a + i))
					++count;
			}
			if (count > maxcount)
			{
				maxcount = count;
				maxvalue = *(a + i);
			}
			else if (maxcount == 1)
				maxvalue = -1;
		}
		return maxvalue;
	}
	inline static int gcd(const int& a, const int& b)
	{
		for (int i = min(a, b); i > 0; i--)
		{
			if (a % i == 0 and b % i == 0)
				return i;
		}
		return 1;
	}
	inline static int lcm(const int& a, const int& b)
	{
		return a * b / gcd(a, b);
	}
	inline static int arithmetic_sequence(const int& start1, const int& diff, const int& numberx)
	{
		// numberx includes start1
		return (numberx - 1) * diff + start1;
	}
	inline static int geometric_sequence(const int& start1, const int& ratio, const int& numberx)
	{
		//numberx includes start1
		return pow(ratio, numberx - 1) * start1;
	}
	inline static int fibonacci(int start1, int start2, int numberx)
	{
		int a;
		if (numberx == 1)
			return start1;
		else if (numberx == 2)
			return start2;
		for (int i = 0; i < numberx - 2; i++)
		{
			a = start1 + start2;
			start1 = start2;
			start2 = a;
		}
		return start2;
	}
	inline static long long factorial(const int& a)
	{
		long long fac = 1;
		for (int i = 1; i <= a; i++)
			fac *= i;
		return fac;
	}
	inline static long long combination(const int& a, const int& b)
	{
		long double sum = 1;
		int c = b;
		if (c > a / 2)
			c = a - b;
		for (int i = 1; i <= c; i++)
			sum = sum * (a - c + i) / i;
		return round(sum);
	}
	inline static long long catalan(const int& a)
	{
		long double count = 1;
		for (double i = 2.0; i <= a; i++)
			count *= (a + i) / i;
		if (a == 30)
			count += 1;
		return (long long)(round(count));
	}
	inline static long long has_digit_num(const int& n, const int& digit)
	{
		// Calculates how many numbers from 1 to n has the digit 'digit'.
		if (n == 0) return 0;
		int k = n / max_pow(n, 10);
		return (k > digit ? k - 1 : k) * (long long)pow(9, floor(log10(n))) + (k == digit ? -1 : has_digit_num(n % max_pow(n, 10), digit));
	}
	inline static int count_digit(long long n)
	{
		if (n == 0)
			return 1;
		int count = 0;
		while (n != 0) {
			n /= 10;
			count++;
		}
		return count;
	}
	inline static int remove_last_digit(const int& a)
	{
		return a / 10;
	}
	inline static int add_digit_at_back(const int& a, const int& digit)
	{
		return 10 * a + digit;
	}
	inline static int read_from_backwards(int a)
	{
		int new_num = 0;
		while (a != 0)
		{
			new_num = 10 * new_num + a % 10;
			a /= 10;
		}
		return new_num;
	}
	inline static int sum_of_all_digits(int a)
	{
		int sum = 0;
		while (a != 0)
		{
			sum += a % 10;
			a /= 10;
		}
		return sum;
	}
	inline static int decimal_to_base(int num, const int& base)
	{
		if (base == 10)
			return num;
		else if (base > 10)
		{
			std::cout << "Error: Can't convert to base greater than 10";
			return 0;
		}
		int converted = 0, maxpow = floor(loga_to_base(num, base));
		for (int i = maxpow; i >= 0; i--)
		{
			converted = converted * 10 + floor(num / pow(base, i));
			num -= floor(num / pow(base, i)) * pow(base, i);
		}
		return converted;
	}
	inline static int base_to_decimal(const std::string& num, const int& base)
	{
		int count = 0;
		for (int i = 0; i < num.length(); i++)
			count += pow(base, num.length() - i - 1) * ((int)num[i] - (isalpha(num[i]) ? 55 : 48));
		return count;
	}
	inline static int base_to_decimal(const int& num, const int& base)
	{
		return base_to_decimal(std::to_string(num), base);
	}
	inline static int randint(const int& startnum, const int& endnum)
	{
		// range includes startnum and endnum
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_int_distribution<int> dist(startnum, endnum);
		return dist(mt);
	}
	inline static int char_to_digit(const char& a)
	{
		return (int)a - 48;
	}
	inline static int num_in_alphabet(const char& a)
	{
		return (int)(isupper(a) ? (char)((int)a + 32) : a) - 96;
	}
	inline static int occurrence_of_digit(long long a, const int& digit)
	{
		int count = 0;
		while (a != 0)
		{
			if (a % 10 == digit)
				count++;
			a /= 10;
		}
		return count;
	}
	inline static int occurrence(const std::string& a, const std::string& b)
	{
		int occurrences = 0;
		std::string::size_type pos = 0;
		while ((pos = a.find(b, pos)) != std::string::npos) {
			++occurrences;
			pos += 1;
		}
		return occurrences;
	}
	inline static bool probability(const double& probability)
	{
		const int times = 10000;
		if (randint(1, times) <= probability * times)
			return true;
		else
			return false;
	}
	inline static bool has_digit(int a, const int& digit)
	{
		if (a == 0)
			return (digit == 0 ? true : false);
		while (a != 0)
		{
			if (a % 10 == digit)
				return true;
			a /= 10;
		}
		return false;
	}
	inline static bool is_int(const float& a)
	{
		if (a == (int)a)
			return true;
		else
			return false;
	}
	inline static bool is_digit(const int& a)
	{
		if (a >= 0 && a <= 9)
			return true;
		else
			return false;
	}
	inline static bool is_digit(const char& a)
	{
		return isdigit(a);
	}
	inline static bool is_prime(const int& a)
	{
		if (a == 0 || a == 1)
			return false;
		for (int i = 2; i * i <= a; i++)
			if (a % i == 0)
				return false;
		return true;
	}
	inline static bool is_composite(const int& a)
	{
		if (a == 0 || a == 1)
			return false;
		return !is_prime(a);
	}
	inline static bool is_perfect(const int& a)
	{
		if (a == 0 or a == 1 or (factor_sum(a) != 2 * a))
			return false;
		else
			return true;
	}
	inline static bool is_leap_year(const int& a)
	{
		if (a % 400 == 0)
			return true;
		else if (a % 100 == 0)
			return false;
		else if (a % 4 == 0)
			return true;
		else
			return false;
	}
	inline static bool is_palindrome(const int& a)
	{
		if (a == read_from_backwards(a))
			return true;
		else
			return false;
	}
	inline static bool is_palindrome(const std::string& a)
	{
		for (int i = 0; i < a.size() / 2; i++)
		{
			if (a[i] != a[a.size() - 1 - i])
				return false;
		}
		return true;
	}
	inline static bool is_square(const int& a)
	{
		int h = a & 0xf;
		if (h != 0 && h != 1 && h != 4 && h != 9)
			return false;
		return pow(floor(sqrt(a)), 2) == a;
	}
	inline static bool is_odd(const int& a)
	{
		return (a % 2 == 0) ? false : true;
	}
	inline static bool is_even(const int& a)
	{
		return (a % 2 == 0) ? true : false;
	}
	inline static bool contains(const std::string& a, const std::string& b)
	{
		return a.find(b) != std::string::npos;
	}
	inline static bool contains_either_way(const std::string& a, const std::string& b)
	{
		return contains(a, b) || contains(b, a);
	}
	inline static bool is_in_array(const int& len, const int& a, int b[])
	{
		std::sort(b, b + len);
		return std::binary_search(b, b + len, a);
	}
	inline static bool is_in_range(const int& num, const int& lower, const int& upper)
	{
		return (lower <= num && num <= upper) ? true : false;
	}
	inline static double one_one_equation(const float& a, const float& b)
	{
		// ax = b
		// x = b / a
		return b / a;
	}
	inline static std::pair<double, double> two_one_equation(const float& a, const float& b, const float& c, const float& d, const float& e, const float& f)
	{
		// ax + by = c
		// dx + ey = f
		// x = (ce-bf)/(ae-bd)
		// y = (cd-af)/(bd-ae)
		return std::make_pair((c * e - b * f) / (a * e - b * d), (c * d - a * f) / (b * d - a * e));
	}
	inline static char digit_to_char(const int& digit)
	{
		return (char)(digit + 48);
	}
	inline static char to_lower(const char& a)
	{
		if (isupper(a))
			return (char)((int)a + 32);
		else
			return a;
	}
	inline static char to_upper(const char& a)
	{
		if (islower(a))
			return (char)((int)a - 32);
		else
			return a;
	}
	inline static char random_char(bool exclude_symbols = false, bool exclude_numbers = false)
	{
		int rand_int;
		if (!exclude_symbols && !exclude_numbers)
			rand_int = randint(43, 126);
		else if (exclude_symbols && exclude_numbers)
		{
			rand_int = randint(71, 122);
			if (rand_int < 'a') rand_int -= 6;
		}
		else if (exclude_symbols && !exclude_numbers)
		{
			rand_int = randint(61, 122);
			if (rand_int < 71) rand_int -= 13;
			else if (rand_int < 97) rand_int -= 6;
		}
		else
		{
			rand_int = randint(43, 126);
			if (rand_int <= '9') rand_int -= 10;
		}
		return (char)rand_int;
	}
	inline static std::string str_first_nth(const std::string& a, const int& b)
	{
		return a.substr(0, b);
	}
	inline static std::string to_lower(const std::string& a)
	{
		std::string return_str;
		for (int i = 0; i < a.length(); i++)
			return_str += to_lower(a[i]);
		return return_str;
	}
	inline static std::string tostr(int a)
	{
		std::string res;
		while (a != 0) res += (a % 10 + '0'), a /= 10;
		std::reverse(res.begin(), res.end());
		return res;
	}
	inline static std::string to_upper(const std::string& a)
	{
		std::string return_str;
		for (int i = 0; i < a.length(); i++)
			return_str += to_upper(a[i]);
		return return_str;
	}
	inline static std::string random_string(const int& length, bool exclude_symbols = false, bool exclude_numbers = false)
	{
		std::string str;
		for (int i = 0; i < length; i++)
			str += random_char(exclude_symbols, exclude_numbers);
		return str;
	}
	inline static std::wstring to_wstring(std::string str)
	{
		std::wstring wideString(str.begin(), str.end());
		return wideString;
	}
	inline static std::vector<int> get_digits(const int& a)
	{
		std::vector<int> res; int k = a;
		while (k != 0) res.push_back(k % 10), k /= 10;
		std::reverse(res.begin(), res.end());
		return res;
	}
	inline static std::vector<int> get_factors(const int& a)
	{
		// including 1 and a
		std::vector<int> res;
		for (int i = 1; i * i <= a; i++)
		{
			if (a % i == 0)
			{
				res.push_back(i);
				if (i * i != a) res.push_back(a / i);
			}
		}
		return res;
	}
	inline static std::vector<int> get_prime_factors(const int& a)
	{
		// including 1 and a
		std::vector<int> res;
		for (int i = 1; i * i <= a; i++)
		{
			if (a % i == 0 && is_prime(i))
			{
				res.push_back(i);
				if (i * i != a && is_prime(a / i)) res.push_back(a / i);
			}
		}
		return res;
	}
	inline static void taking_notes()
	{
		std::string a;
		std::cin >> a;
		while (a != "break")
			std::cin >> a;
	}
	inline static void print_endl()
	{
		std::cout << std::endl;
	}
	inline static void print_array_with_endl(const int a[], const int len)
	{
		for (int i = 0; i < len; i++)
			std::cout << a[i] << std::endl;
	}
	inline static void print_array_with_space(const int a[], const int len)
	{
		for (int i = 0; i < len - 1; i++)
			std::cout << a[i] << " ";
		std::cout << a[len - 1];
	}
	inline static void solve(const double& number, const int& place) {
		std::cout << std::fixed;
		std::cout << std::setprecision(place);
		std::cout << number << std::endl;
	}
	inline static void guess_the_number(const int& startnum, const int& endnum)
	{
		int turns = abs(floor((log2(endnum - startnum) - 1) * 0.8)), guessnum = randint(startnum, endnum), input;
		std::string input2;
		std::cout << "Let's play a game called guess_the_number!\n";
		std::cout << "In this game, I am thinking of a number, and you have " << turns << " turns to guess it.\n";
		std::cout << "In each turn, you can guess a number, and I will say whether it is too big or too small.\n";
		std::cout << "See if you can guess it! ";
		for (int turn = 1; turn <= turns; turn++)
		{
			std::cout << "Now, please guess a number.\n";
			if (turn == turns)
			{
				std::cout << "This is your last guess.Guess wisely.\n";
				std::cin >> input;
				if (input == guessnum)
					std::cout << "Wow! Congratulations, you guessed it on your last try! You WIN!!!\n";
				else
				{
					std::cout << "Nope.This number is not correct, and you have ran out of guesses.Game Over.You lose.\n";
					std::cout << "The number is " << guessnum << ".\n";
				}
			}
			else
			{
				std::cin >> input;
				if (input < guessnum)
					std::cout << "Too small! You still have " << turns - turn << " tries.";
				else if (input > guessnum)
					std::cout << "Too big! You still have " << turns - turn << " tries.";
				else
					std::cout << "Congratulations, that is the number I'm thinking! You WIN!!!\n";
			}
		}
		std::cout << "Do you want to play again? \n";
		std::cin >> input2;
		if (input2 == "Yes" or input2 == "yes" or input2 == "YES")
			guess_the_number(startnum, endnum);
	}
}
namespace Structures
{
	struct point
	{
		int x = 0, y = 0;
		point() = default;
		point(int x, int y) : x(x), y(y) {}
		double dist(const point& other) const
		{
			return sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
		}
		double grid_dist(const point& other) const
		{
			return abs(x - other.x) + abs(y - other.y);
		}
		double chebyshev_dist(const point& other) const
		{
			return max(abs(x - other.x), abs(y - other.y));
		}
		double ptabs() const
		{
			return dist(point(0, 0));
		}
	};
	std::ostream& operator<<(std::ostream& os, const point& pt)
	{
		os << pt.x << " " << pt.y;
		return os;
	}
	std::istream& operator>>(std::istream& stream, point& pt)
	{
		stream >> pt.x >> pt.y;
		return stream;
	}
	std::ostream& operator<<(std::ostream& os, const std::pair<int, int>& p)
	{
		os << p.first << " " << p.second;
		return os;
	}
	std::istream& operator>>(std::istream& stream, std::pair<int, int>& p)
	{
		stream >> p.first >> p.second;
		return stream;
	}
	template <typename T>
	void prefix(T* begin, T* end)
	{
		for (T* i = begin + 1; i < end; i++)
			*i += *(i - 1);
	}
	template <typename T>
	void diff(T* begin, T* end)
	{
		for (T* i = end - 1; i > begin; i--)
			*i -= *(i - 1);
	}
	template <typename T>
	class SparseTable
	{
	private:
		const int n;
		int maxst;
		std::vector<T> orig;
		std::vector<std::vector<T>> st;
		T(*func)(T, T);
		bool start_from_one = true;
		void init()
		{
			for (int i = 0; i < n; i++) st[i][0] = orig[i];
			for (int p = 1, j = 1; j <= maxst; p *= 2, j++)
				for (int i = 0; i + p < n; i++)
					st[i][j] = func(st[i][j - 1], st[i + p][j - 1]);
		}
	public:
		struct Functions
		{
			T min_func(T a, T b)
			{
				return min(a, b);
			}
			T max_func(T a, T b)
			{
				return max(a, b);
			}
			T gcd_func(T a, T b)
			{
				return std::gcd(a, b);
			}
			T lcm_func(T a, T b)
			{
				return std::lcm(a, b);
			}
		};
		SparseTable(T* begin, T* end, T(*func)(T, T)) : func(func), n(end - begin)
		{
			maxst = ceil(log2(n));
			for (T* i = begin; i <= end; i++) orig.push_back(*i);
			std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
			st = _st;
			init();
		}
		SparseTable(const std::vector<T>& orig, T(*func)(T, T)) : func(func), n(orig.size()), orig(orig)
		{
			maxst = ceil(log2(n));
			std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
			st = _st;
			init();
		}
		void SetStartFromZero()
		{
			start_from_one = false;
		}
		T query(const int& l, const int& r) const
		{
			int dist = floor(log2(r - l + 1));
			if (start_from_one) return func(st[l - 1][dist], st[r - (1 << dist)][dist]);
			else return func(st[l][dist], st[r - (1 << dist) + 1][dist]);
		}
	};
	class JointSet
	{
	private:
		const int n;
		std::vector<int> fa;
	public:
		JointSet(int n) : n(n)
		{
			fa = std::vector<int>(n + 1);
			for (int i = 1; i <= n; i++) fa[i] = i;
		}
		int find(int idx)
		{
			return fa[idx] == idx ? idx : find(fa[idx]);
		}
		void merge(int x, int y)
		{
			fa[find(x)] = find(y);
		}
		bool is_connected(int x, int y)
		{
			return find(x) == find(y);
		}
	};
	class WeightedJointSet
	{
	private:
		const int n;
		std::vector<std::pair<int, int> > fa;
	public:
		WeightedJointSet(int n) : n(n)
		{
			fa = std::vector<std::pair<int, int> >(n + 1);
			for (int i = 1; i <= n; i++) fa[i].first = i, fa[i].second = 0;
		}
		int find(int idx)
		{
			return fa[idx].first == idx ? idx : find(fa[idx].first);
		}
		void merge(int x, int y, int weight)
		{
			fa[find(x)].first = find(y);
			fa[find(x)].second = weight;
		}
		bool is_connected(int x, int y)
		{
			return find(x) == find(y);
		}
	};
	template <typename T>
	class MonotonicStack // Calls a stack, but really a std::vector
	{
	private:
		bool (*func)(T, T); // From Up to Down of stack
		std::vector<T> stack;
	public:
		MonotonicStack(bool (*func)(T, T)) : func(func) {}
		void push_back(const T& val)
		{
			while (!stack.empty() && func(stack.back(), val)) stack.pop_back();
			stack.push_back(val);
		}
		void pop_back()
		{
			stack.pop_back();
		}
		T back()
		{
			return stack.back();
		}
		T operator[] (const int& idx) const
		{
			return stack[idx];
		}
		std::vector<T> GetStack()
		{
			return stack;
		}
	};
	template <typename T>
	class MonotonicDeque
	{
	private:
		int cnt = 0;
		std::deque<T> q, idx;
		const bool has_idx = false;
		bool (*func)(T, T); // From Back to Front of deque
	public:
		MonotonicDeque(bool (*func)(T, T)) : func(func) {}
		MonotonicDeque(bool (*func)(T, T), bool has_idx) : func(func), has_idx(has_idx) {}
		void push_back(const T& num)
		{
			while (!q.empty() && func(q.back(), num))
			{
				q.pop_back();
				if (has_idx) idx.pop_back();
			}
			q.push_back(num);
			if (has_idx) idx.push_back(++cnt);
		}
		void pop_back()
		{
			q.pop_back();
		}
		void push_front(const T& num)
		{
			q.push_front(num);
		}
		void pop_front()
		{
			q.pop_front();
		}
		void pop_front_until(int min_idx)
		{
			if (!has_idx) return;
			while (idx.front() <= min_idx) idx.pop_front(), q.pop_front();
		}
	};
	template <typename T>
	class TreelikeArray
	{
	private:
		const int n;
		std::vector<int> a;
	public:
		TreelikeArray(int n) : n(n)
		{
			a = std::vector<int>(n + 1);
		}
		int lowbit(int x)
		{
			return x & -x;
		}
		int getsum(int x)
		{
			int cnt = 0;
			while (x)
			{
				cnt = cnt + a[x];
				x = x - lowbit(x);
			}
			return cnt;
		}
		int getsum(int l, int r)
		{
			return getsum(r) - getsum(l - 1);
		}
		void add(int x, int k)
		{
			while (x <= n)
			{
				a[x] += k;
				x = x + lowbit(x);
			}
		}
	};
	template <typename T>
	class Tree
	{
	private:
		const int n;
		int root, tree_stmax;
		bool has_val = false, is_init_tree_st = false;
		struct node
		{
			int fa = -1, depth = -1;
			T val = 0;
			std::vector<int> to;
			node() {}
			node(int fa) : fa(fa) {}
			node(std::vector<int> to) : to(to) {}
			node(std::vector<int> to, T val) : val(val), to(to) {}
			node(int fa, std::vector<int> to) : fa(fa), to(to) {}
			node(int fa, T val, std::vector<int> to) : fa(fa), val(val), to(to) {}
		};
		std::vector<std::vector<int> > tree_st;
	public:
		std::vector<node> tree;
		Tree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) {}
		Tree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) {}
		Tree(int n, std::vector<int> to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
			init(root);
		}
		Tree(std::vector<int>* to_begin, std::vector<int>* to_end) : n(to_end - to_begin), root(1)
		{
			tree_stmax = ceil(log2(n));
			for (std::vector<int>* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
			init(root);
		}
		Tree(int n, std::vector<int> to[], int root) : n(n), root(root), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
			init(root);
		}
		Tree(std::vector<int>* to_begin, std::vector<int>* to_end, int root) : n(to_end - to_begin), root(root)
		{
			tree_stmax = ceil(log2(n));
			for (std::vector<int>* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
			init(root);
		}
		Tree(int n, std::pair<T, std::vector<int> > to[]) : has_val(true), n(n), root(1), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
			init(root);
		}
		Tree(std::pair<T, std::vector<int> >* to_begin, std::pair<T, std::vector<int> >* to_end) : has_val(true), n(to_end - to_begin), root(1)
		{
			tree_stmax = ceil(log2(n));
			tree.push_back(node());
			for (std::pair<T, std::vector<int> >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
			init(root);
		}
		Tree(int n, std::pair<T, std::vector<int> > to[], int root) : has_val(true), n(n), root(root), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
			init(root);
		}
		Tree(std::pair<T, std::vector<int> >* to_begin, std::pair<T, std::vector<int> >* to_end, int root) : has_val(true), n(to_end - to_begin), root(root)
		{
			tree_stmax = ceil(log2(n));
			tree.push_back(node());
			for (std::pair<T, std::vector<int> >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
			init(root);
		}
		void init(int now, int father = -1)
		{
			if (!has_val) tree[now].val = now;
			tree[now].depth = tree[father].depth + 1;
			tree[now].fa = father;
			for (int& i : tree[now].to)
				if (i != father)
					init(i, now);
		}
		void init_tree_st()
		{
			tree_st = std::vector<std::vector<int> >(n, std::vector<int>(tree_stmax + 1, 0));
			for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
			for (int j = 1; j <= tree_stmax; j++)
				for (int i = 1; i <= n; i++)
					tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
		}
		void insert(int l, int r, int length)
		{
			tree[l].to.push_back(r);
			tree[r].to.push_back(l);
		}
		int GetRoot()
		{
			return root;
		}
		int ChangeRoot(int new_root)
		{
			root = new_root;
			init(new_root);
		}
		int GetNodeIdx(T val)
		{
			for (int i = 0; i < tree.size(); i++)
				if (tree[i].val == val)
					return i;
		}
		void dfs(void* func(node a), int start)
		{
			func(tree[start]);
			for (int& i : tree[start].to)
				if (i != tree[start].fa)
					dfs(func, i);
		}
		int dfscnt(int func(node a), int start)
		{
			int cnt = func(tree[start]);
			for (int& i : tree[start].to)
				if (i != tree[start].fa)
					cnt += dfscnt(func, i);
			return cnt;
		}
		void bfs(void* func(node a), int start)
		{
			std::queue<int> idx_queue;
			idx_queue.push(start);
			while (!idx_queue.empty())
			{
				int idx = idx_queue.front();
				func(tree[idx]);
				for (int& i : tree[idx].to)
					if (i != tree[idx].fa)
						idx_queue.push(i);
				idx_queue.pop();
			}
		}
		int bfscnt(int func(node a), int start)
		{
			int cnt = 0;
			std::queue<int> idx_queue;
			idx_queue.push(start);
			while (!idx_queue.empty())
			{
				int idx = idx_queue.front();
				cnt += func(tree[idx]);
				for (int& i : tree[idx].to)
					if (i != tree[idx].fa)
						idx_queue.push(i);
				idx_queue.pop();
			}
			return cnt;
		}
		int lca(int l, int r)
		{
			if (!is_init_tree_st) init_tree_st(), is_init_tree_st = true;
			int _l, _r;
			if (has_val) _l = GetNodeIdx(l), _r = GetNodeIdx(r);
			else _l = l, _r = r;
			if (tree[_l].depth < tree[_r].depth) std::swap(_l, _r);
			int diff = tree[_l].depth - tree[_r].depth;
			for (int i = tree_stmax; i >= 0; i--) if ((1 << i) & diff) _l = tree_st[_l][i];
			for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
			return (_l == _r ? _l : tree[_l].fa);
		}
	};
	template <typename T>
	class WeightedTree
	{
	private:
		const int n;
		int root, tree_stmax;
		bool has_val = false, is_init_tree_st = false;
		struct node
		{
			int fa = -1, depth = -1, segment_depth = -1;
			T val = 0;
			std::vector<std::pair<int, int> > to; // std::vector<<dest_idx, length>>
			node() {}
			node(int fa) : fa(fa) {}
			node(std::vector<std::pair<int, int> > to) : to(to) {}
			node(std::vector<std::pair<int, int> > to, T val) : val(val), to(to) {}
			node(int fa, std::vector<std::pair<int, int> > to) : fa(fa), to(to) {}
			node(int fa, T val, std::vector<std::pair<int, int> > to) : fa(fa), val(val), to(to) {}
		};
		std::vector<std::vector<int> > tree_st;
	public:
		std::vector<node> tree;
		WeightedTree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) {}
		WeightedTree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) {}
		WeightedTree(int n, std::vector<std::pair<int, int> > to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
			init(root);
		}
		WeightedTree(std::vector<std::pair<int, int> >* to_begin, std::vector<std::pair<int, int> >* to_end) : n(to_end - to_begin), root(1)
		{
			tree_stmax = ceil(log2(n));
			for (std::vector<std::pair<int, int> >* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
			init(root);
		}
		WeightedTree(int n, std::vector<std::pair<int, int> > to[], int root) : n(n), root(root), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
			init(root);
		}
		WeightedTree(std::vector<std::pair<int, int> >* to_begin, std::vector<std::pair<int, int> >* to_end, int root) : n(to_end - to_begin), root(root)
		{
			tree_stmax = ceil(log2(n));
			for (std::vector<std::pair<int, int> >* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
			init(root);
		}
		WeightedTree(int n, std::pair<T, std::vector<std::pair<int, int> > > to[]) : has_val(true), n(n), root(1), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
			init(root);
		}
		WeightedTree(std::pair<T, std::vector<std::pair<int, int> > >* to_begin, std::pair<T, std::vector<std::pair<int, int> > >* to_end) : has_val(true), n(to_end - to_begin), root(1)
		{
			tree_stmax = ceil(log2(n));
			tree.push_back(node());
			for (std::pair<T, std::vector<std::pair<int, int> > >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
			init(root);
		}
		WeightedTree(int n, std::pair<T, std::vector<std::pair<int, int> > > to[], int root) : has_val(true), n(n), root(root), tree_stmax(ceil(log2(n)))
		{
			tree.push_back(node());
			for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
			init(root);
		}
		WeightedTree(std::pair<T, std::vector<std::pair<int, int> > >* to_begin, std::pair<T, std::vector<std::pair<int, int> > >* to_end, int root) : has_val(true), n(to_end - to_begin), root(root)
		{
			tree_stmax = ceil(log2(n));
			tree.push_back(node());
			for (std::pair<T, std::vector<std::pair<int, int> > >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
			init(root);
		}
		void init(int now, int father = -1, int length = -1)
		{
			if (!has_val) tree[now].val = now;
			tree[now].depth = tree[father].depth + 1;
			tree[now].segment_depth = tree[father].segment_depth + length;
			tree[now].fa = father;
			for (std::pair<int, int>& i : tree[now].to)
				if (i.first != father)
					init(i.first, now, i.second);
		}
		void init_tree_st()
		{
			tree_st = std::vector<std::vector<int> >(n, std::vector<int>(tree_stmax + 1, 0));
			for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
			for (int j = 1; j <= tree_stmax; j++)
				for (int i = 1; i <= n; i++)
					tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
		}
		void insert(int l, int r, int length)
		{
			tree[l].to.push_back(std::make_pair(r, length));
			tree[r].to.push_back(std::make_pair(l, length));
		}
		int GetRoot()
		{
			return root;
		}
		int ChangeRoot(int new_root)
		{
			root = new_root;
			init(new_root);
		}
		int GetNodeIdx(T val)
		{
			for (int i = 0; i < tree.size(); i++)
				if (tree[i].val == val)
					return i;
		}
		void dfs(void* func(node a), int start)
		{
			func(tree[start]);
			for (std::pair<int, int>& i : tree[start].to)
				if (i.first != tree[start].fa)
					dfs(func, i.first);
		}
		int dfscnt(int func(node a), int start)
		{
			int cnt = func(tree[start]);
			for (std::pair<int, int>& i : tree[start].to)
				if (i.first != tree[start].fa)
					cnt += dfscnt(func, i.first);
			return cnt;
		}
		void bfs(void* func(node a), int start)
		{
			std::queue<int> idx_queue;
			idx_queue.push(start);
			while (!idx_queue.empty())
			{
				int idx = idx_queue.front();
				func(tree[idx]);
				for (std::pair<int, int>& i : tree[idx].to)
					if (i.first != tree[idx].fa)
						idx_queue.push(i.first);
				idx_queue.pop();
			}
		}
		int bfscnt(int func(node a), int start)
		{
			int cnt = 0;
			std::queue<int> idx_queue;
			idx_queue.push(start);
			while (!idx_queue.empty())
			{
				int idx = idx_queue.front();
				cnt += func(tree[idx]);
				for (std::pair<int, int>& i : tree[idx].to)
					if (i.first != tree[idx].fa)
						idx_queue.push(i.first);
				idx_queue.pop();
			}
			return cnt;
		}
		int lca(int l, int r)
		{
			if (!is_init_tree_st) init_tree_st(), is_init_tree_st = true;
			int _l, _r;
			if (has_val) _l = GetNodeIdx(l), _r = GetNodeIdx(r);
			else _l = l, _r = r;
			if (tree[_l].depth < tree[_r].depth) std::swap(_l, _r);
			int diff = tree[_l].depth - tree[_r].depth;
			for (int i = tree_stmax; i >= 0; i--) if ((1 << i) & diff) _l = tree_st[_l][i];
			for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
			return (_l == _r ? _l : tree[_l].fa);
		}
	};
}
namespace IO
{
	template <typename T>
	T read()
	{
		T temp;
		std::cin >> temp;
		return temp;
	}
	int read()
	{
		int temp;
		std::cin >> temp;
		return temp;
	}
	template <typename T>
	void read(T& value)
	{
		std::cin >> value;
	}
	template <typename T, typename... Targs>
	void read(T& value, Targs&... Fargs)
	{
		std::cin >> value;
		read(Fargs...);
	}
	template <typename T>
	void read(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			std::cin >> *i;
	}
	template <typename Iter>
	decltype(auto) read(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			std::cin >> *i;
	}
	template <typename T>
	void print(T value)
	{
		std::cout << value;
	}
	template <typename T>
	void println(T value)
	{
		std::cout << value << std::endl;
	}
	template <typename T>
	void print_space(T value)
	{
		std::cout << value << ' ';
	}
	template <typename T, typename... Targs>
	void print(T value, Targs... Fargs)
	{
		std::cout << value;
		print(Fargs...);
	}
	template <typename T>
	void print(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			std::cout << *i;
	}
	template <typename Iter>
	decltype(auto) print(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			std::cout << *i;
	}
	template <typename T, typename... Targs>
	void print_space(T value, Targs... Fargs)
	{
		std::cout << value << ' ';
		print_space(Fargs...);
	}
	template <typename T>
	void print_space(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			std::cout << *i << ' ';
	}
	template <typename Iter>
	decltype(auto) print_space(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			std::cout << *i << ' ';
	}
	template <typename T, typename... Targs>
	void println(T value, Targs... Fargs)
	{
		std::cout << value << std::endl;
		println(Fargs...);
	}
	template <typename T>
	void println(T* begin, T* end)
	{
		for (T* i = begin; i < end; i++)
			std::cout << *i << std::endl;
	}
	template <typename Iter>
	decltype(auto) println(Iter begin, Iter end)
	{
		for (Iter i = begin; i < end; i++)
			std::cout << *i << std::endl;
	}
}
using namespace std;
using namespace Algorithms;
using namespace Main;
using namespace Structures;
using namespace IO;