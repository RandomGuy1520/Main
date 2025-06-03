//#pragma once
//
//// Standard Library
//#include <unordered_map>
//#include <algorithm>
//#include <iostream>
//#include <numeric>
//#include <climits>
//#include <iomanip>
//#include <complex>
//#include <utility>
//#include <fstream>
//#include <conio.h>
//#include <memory>
//#include <string>
//#include <vector>
//#include <chrono>
//#include <random>
//#include <thread>
//#include <cmath>
//#include <queue>
//#include <array>
//#include <new>
//#include <set>
//#include <map>
//
//// Dependencies
////#define INCLUDE_DEPENDENCIES
//
//#ifdef INCLUDE_DEPENDENCIES
//#include "GL/glew.h"
//#include "GLFW/glfw3.h"
//#endif
//
//#ifdef INCLUDE_WINDOWS
//#include <windows.h>
//#include <shellapi.h>
//#endif
//
//#define MACROS
//#ifdef  MACROS
//#ifdef WIN32
//#define getchar() _getchar_nolock()
//#else
//#ifdef WIN64
//#define getchar() _getchar_nolock()
//#else
//#define getchar() getchar_unlocked()
//#endif
//#endif
//#define ll long long
//#define ull unsigned long long
//#define desync() ios::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
//#define min(a, b) ((a) < (b) ? (a) : (b))
//#define max(a, b) ((a) > (b) ? (a) : (b))
//#define repeat(j)      for (int i = 0; i <  j; i++)
//#define iter(i, a, b)  for (int i = a; i <= b; i++)
//#define iterzero(i, j) for (int i = 0; i <  j; i++)
//#define itervec(i, j)  for (auto& i : j)
//#define square(i)      ((i) * (i))
//#define input(i)       std::cin >> (i)
//#define outprecision(i) std::cout << std::fixed << std::setprecision(i);
//#ifdef __GNUC__
//#define gcd __gcd
//#endif
//#define int long long
//#endif
//
//namespace IO
//{
//	template <typename T>
//	inline void read(T& x)
//	{
//		if (std::is_integral_v<T>)
//		{
//			int x = 0, f = 1;
//			char ch = getchar();
//			while (!isdigit(ch)) {
//				if (ch == '-')
//					f = -1;
//				ch = getchar();
//			}
//			while (isdigit(ch)) {
//				x = (x << 1) + (x << 3) + (ch ^ 48);
//				ch = getchar();
//			}
//			return x * f;
//		}
//		else std::cin >> x;
//	}
//	template <typename T, typename... Targs>
//	static inline void read(T& value, Targs&... Fargs)
//	{
//		std::cin >> value;
//		read(Fargs...);
//	}
//	template <typename T>
//	static inline void read(T* begin, T* end)
//	{
//		for (T* i = begin; i < end; i++)
//			std::cin >> *i;
//	}
//	template <typename Iter>
//	static inline decltype(auto) read(Iter begin, Iter end)
//	{
//		for (Iter i = begin; i < end; i++)
//			std::cin >> *i;
//	}
//	template <typename T>
//	static inline void print(T value)
//	{
//		std::cout << value;
//	}
//	template <typename T>
//	static void println(T value)
//	{
//		std::cout << value << std::endl;
//	}
//	template <typename T>
//	static void print_space(T value)
//	{
//		std::cout << value << ' ';
//	}
//	template <typename T, typename... Targs>
//	static void print(T value, Targs... Fargs)
//	{
//		std::cout << value;
//		print(Fargs...);
//	}
//	template <typename T>
//	static void print(T* begin, T* end)
//	{
//		for (T* i = begin; i < end; i++)
//			std::cout << *i;
//	}
//	template <typename Iter>
//	static decltype(auto) print(Iter begin, Iter end)
//	{
//		for (Iter i = begin; i < end; i++)
//			std::cout << *i;
//	}
//	template <typename T, typename... Targs>
//	static void print_space(T value, Targs... Fargs)
//	{
//		std::cout << value << ' ';
//		print_space(Fargs...);
//	}
//	template <typename T>
//	static void print_space(T* begin, T* end)
//	{
//		for (T* i = begin; i < end; i++)
//			std::cout << *i << ' ';
//	}
//	template <typename Iter>
//	static decltype(auto) print_space(Iter begin, Iter end)
//	{
//		for (Iter i = begin; i < end; i++)
//			std::cout << *i << ' ';
//	}
//	template <typename T, typename... Targs>
//	static void println(T value, Targs... Fargs)
//	{
//		std::cout << value << std::endl;
//		println(Fargs...);
//	}
//	template <typename T>
//	static void println(T* begin, T* end)
//	{
//		for (T* i = begin; i < end; i++)
//			std::cout << *i << std::endl;
//	}
//	template <typename Iter>
//	static decltype(auto) println(Iter begin, Iter end)
//	{
//		for (Iter i = begin; i < end; i++)
//			std::cout << *i << std::endl;
//	}
//}
//namespace OI
//{
//	class Sort
//	{
//	private:
//		int GetMax(int* arr, int n)
//		{
//			int max = arr[0];
//			for (int i = 1; i < n; i++)
//				if (arr[i] > max)
//					max = arr[i];
//			return max;
//		}
//		void Merge(int* arr, int p, int q, int r)
//		{
//			int n1 = q - p + 1, n2 = r - q, i, j, k;
//			int* L = new int[n1], * M = new int[n2];
//
//			for (i = 0; i < n1; i++)
//				L[i] = arr[p + i];
//			for (j = 0; j < n2; j++)
//				M[j] = arr[q + 1 + j];
//
//			i = 0;
//			j = 0;
//			k = p;
//			while (i < n1 && j < n2)
//			{
//				if (L[i] <= M[j])
//				{
//					arr[k] = L[i];
//					i++;
//				}
//				else
//				{
//					arr[k] = M[j];
//					j++;
//				}
//				k++;
//			}
//			while (i < n1)
//			{
//				arr[k] = L[i];
//				i++;
//				k++;
//			}
//			while (j < n2)
//			{
//				arr[k] = M[j];
//				j++;
//				k++;
//			}
//			delete[] L;
//			delete[] M;
//		}
//		void MergeRecursion(int* arr, int l, int r)
//		{
//			// Time & Space complexity: O(n*log(n)) / O(n)
//			if (l < r)
//			{
//				int m = l + (r - l) / 2;
//				MergeRecursion(arr, l, m);
//				MergeRecursion(arr, m + 1, r);
//				Merge(arr, l, m, r);
//			}
//		}
//		int Partition(int* arr, int low, int high)
//		{
//			int pivot = arr[high];
//			int i = (low - 1);
//			for (int j = low; j <= high - 1; j++)
//			{
//				if (arr[j] < pivot)
//				{
//					i++;
//					std::swap(arr[i], arr[j]);
//				}
//			}
//			std::swap(arr[i + 1], arr[high]);
//			return i + 1;
//		}
//		void QuickRecursion(int* arr, int low, int high)
//		{
//			if (low < high)
//			{
//				int p = Partition(arr, low, high);
//				QuickRecursion(arr, low, p - 1);
//				QuickRecursion(arr, p + 1, high);
//			}
//		}
//		void Heapify(int* arr, int N, int i)
//		{
//			int largest = i, l = 2 * i + 1, r = 2 * i + 2;
//			if (l < N && arr[l] > arr[largest])
//				largest = l;
//			if (r < N && arr[r] > arr[largest])
//				largest = r;
//			if (largest != i)
//			{
//				std::swap(arr[i], arr[largest]);
//				Heapify(arr, N, largest);
//			}
//		}
//		void SortDigit(int* arr, int n, int exp, int base)
//		{
//			int* output = new int[n];
//			int* count = new int[base];
//			for (int i = 0; i < base; i++) count[i] = 0;
//			for (int i = 0; i < n; i++)
//				count[(arr[i] / exp) % base]++;
//			for (int i = 1; i < base; i++)
//				count[i] += count[i - 1];
//			for (int i = n - 1; i >= 0; i--)
//			{
//				output[count[(arr[i] / exp) % base] - 1] = arr[i];
//				count[(arr[i] / exp) % 10]--;
//			}
//			for (int i = 0; i < n; i++)
//				arr[i] = output[i];
//			delete[] output;
//			delete[] count;
//		}
//	public:
//		void BubbleSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			// Time & Space complexity: O(n^2) / O(1)
//			bool swapped;
//			for (int i = 0; i < len - 1; i++)
//			{
//				swapped = false;
//				for (int j = 0; j < len - i - 1; j++)
//				{
//					if (arr[j] > arr[j + 1])
//					{
//						std::swap(arr[j], arr[j + 1]);
//						swapped = true;
//					}
//				}
//				if (swapped == false)
//					break;
//			}
//		}
//		void SelectionSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			// Time & Space complexity: O(n^2) / O(1)
//			int i, j, min_idx;
//			for (i = 0; i < len - 1; i++)
//			{
//				min_idx = i;
//				for (j = i + 1; j < len; j++)
//					if (arr[j] < arr[min_idx])
//						min_idx = j;
//				if (min_idx != i)
//					std::swap(arr[min_idx], arr[i]);
//			}
//		}
//		void InsertionSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			// Time & Space complexity: O(n^2) / O(1)
//			int i, key, j;
//			for (i = 1; i < len; i++)
//			{
//				key = arr[i];
//				for (j = i - 1; j >= 0 && arr[j] > key; j--)
//					arr[j + 1] = arr[j];
//				arr[j + 1] = key;
//			}
//		}
//		void MergeSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			MergeRecursion(arr, 0, len - 1);
//		}
//		void QuickSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			QuickRecursion(arr, 0, len - 1);
//		}
//		void HeapSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			for (int i = len / 2 - 1; i >= 0; i--)
//				Heapify(arr, len, i);
//			for (int i = len - 1; i > 0; i--)
//			{
//				std::swap(arr[0], arr[i]);
//				Heapify(arr, i, 0);
//			}
//		}
//		void RadixSort(int* arr, int* end, const int& base = 10)
//		{
//			int len = end - arr;
//			int m = GetMax(arr, len);
//			for (int exp = 1; m / exp > 0; exp *= base)
//				SortDigit(arr, len, exp, base);
//		}
//		void CountingSort(int* arr, int* end)
//		{
//			int len = end - arr;
//			int max = GetMax(arr, len);
//			int* count = new int[max + 1], * output = new int[len];
//			for (int i = 0; i <= max; i++)
//				count[i] = 0;
//			for (int i = 0; i < len; i++)
//				count[arr[i]]++;
//			for (int i = 1; i <= max; i++)
//				count[i] += count[i - 1];
//			for (int i = len - 1; i >= 0; i--)
//			{
//				output[count[arr[i]] - 1] = arr[i];
//				count[arr[i]]--;
//			}
//			for (int i = 0; i < len; i++)
//				arr[i] = output[i];
//		}
//		void BucketSort(int* arr, int* end, int buckets_num = NULL)
//		{
//			int len = end - arr;
//			if (buckets_num == NULL) buckets_num = len / 3 + 2;
//			int div = pow(buckets_num, floor(log2(GetMax(arr, len)) / log2(buckets_num)));
//			std::vector<std::vector<int>> b(buckets_num);
//			for (int i = 0; i < len; i++)
//			{
//				int bi = arr[i] / div - 1;
//				b[bi].push_back(arr[i]);
//			}
//			for (int i = 0; i < len; i++)
//				if (!b[i].empty())
//					std::sort(b[i].begin(), b[i].end());
//			int index = 0;
//			for (int i = 0; i < len; i++)
//				for (int j = 0; j < b[i].size(); j++)
//					arr[index++] = b[i][j];
//		}
//	};
//	class Number
//	{
//	private:
//		static unsigned long long MOD, SIZE, BASE;
//		static bool init_fac;
//		static std::vector<int> fac, inv, p, primes, is_prime;
//	public:
//		Number() {}
//		static void SetMod(int mod)
//		{
//			MOD = mod;
//		}
//		static void SetSize(int size)
//		{
//			SIZE = size;
//		}
//		static void SetBase(int base) { BASE = base; }
//		static void InitFac(int mod = MOD)
//		{
//			if (init_fac) return;
//			fac[0] = 1;
//			for (int i = 1; i < SIZE; i++) fac[i] = fac[i - 1] * i % mod;
//			inv[SIZE - 1] = qpow(fac[SIZE - 1], mod - 2);
//			for (int i = SIZE - 2; i >= 0; i--) inv[i] = inv[i + 1] * (i + 1) % mod;
//		}
//		static void InitBase(int base = BASE, int mod = MOD)
//		{
//			p[0] = 1;
//			for (int i = 1; i < SIZE; i++) p[i] = p[i - 1] * base % mod;
//		}
//		static void InitPrimes(const int& n)
//		{
//			for (int i = 2; i < n; i++)
//				is_prime[i] = true;
//			for (int i = 2; i < n; i++)
//			{
//				if (is_prime[i])
//					primes.push_back(i);
//				for (int j = 0; j < primes.size() && primes[j] * i < n; j++)
//				{
//					is_prime[primes[j] * i] = false;
//					if (!(i % primes[j]))
//						break;
//				}
//			}
//		}
//		static int qpow(int u, int v, int mod = MOD)
//		{
//			int res = 1;
//			while (v)
//			{
//				if (v & 1) res = res * u % mod;
//				u = u * u % mod;
//				v >>= 1;
//			}
//			return res;
//		}
//		static int inverse(int x, int mod = MOD)
//		{
//			return qpow(x, mod - 2, mod);
//		}
//		static int C(int u, int v, int mod = MOD)
//		{
//			InitFac(mod);
//			if (u < v) return 0;
//			if (u < mod && v < mod) return fac[u] * inverse(fac[v]) * inverse(fac[u - v]);
//			else return C(u % mod, v % mod) * C(u / mod, v / mod);
//		}
//	};
//	class Algorithms
//	{
//	private:
//		void FFTCalculate(std::complex<double> A[], int ord[], int n)
//		{
//			// Given A[] = f(x) = a_0 * x^n + a_1 * x^{n-1} + ... + a_{n-1} * x^1 + a_n, Calculate for 0 <= x < n : f(omega ^ x), omega = e ^ (2 * pi * i / n)
//			const double pi = acos(-1);
//			for (int i = 1; i < n; i++) // Change A sequence's arrangement to ord[i]'s
//				if (ord[i] > i)
//					std::swap(A[i], A[ord[i]]);
//			for (int len = 2; len <= n; len <<= 1)
//			{
//				std::complex<double> omega(cos(2 * pi / len), sin(2 * pi / len)); // nth roots of unity
//				for (int l = 0, r = len - 1; r <= n; l += len, r += len)
//				{
//					std::complex<double> w(1.0, 0.0); // w = omega ^ k
//					for (int p = l; p < l + len / 2; p++)
//					{
//						std::complex<double> x = A[p] + w * A[p + len / 2], y = A[p] - w * A[p + len / 2];
//						A[p] = x; A[p + len / 2] = y;
//						w *= omega;
//					}
//				}
//			}
//		}
//	public:
//		std::vector<int> FFT(std::vector<int> a, std::vector<int> b)
//		{
//			// Calculate a * b = output where a, b, output are polynomials.
//			const int MAXN = max(a.size(), b.size()) << 3 + 7;
//			std::complex<double>* F = new std::complex<double>[MAXN];
//			std::vector<int> res;
//			int* ord = new int[MAXN];
//			for (int i = 0; i < a.size(); i++) F[i].real(a[i]);
//			for (int i = 0; i < b.size(); i++) F[i].imag(b[i]);
//			int n = a.size() - 1, m = b.size() - 1, num = 1;
//			while (num <= n + m) num <<= 1;
//			int d = num >> 1, p = 0;
//			ord[p++] = 0; ord[p++] = d;
//			for (int w = 2; w <= num; w <<= 1)
//			{
//				d >>= 1;
//				for (int p0 = 0; p0 < w; p0++)
//					ord[p++] = ord[p0] | d;
//			}
//			FFTCalculate(F, ord, num);
//			for (int i = 0; i < num; i++)
//				F[i] *= F[i];
//			FFTCalculate(F, ord, num);
//			std::reverse(F + 1, F + num);
//			for (int i = 0; i <= n + m; i++)
//				res.push_back((int)round(F[i].imag() / 2 / num));
//			return res;
//		}
//		int Knapsack_01(int num, int capacity, int weight[], int value[])
//		{
//			std::vector<std::vector<int>> dp(num + 1, std::vector<int>(capacity + 1, 0));
//			for (int i = 1; i <= capacity; i++)
//			{
//				for (int j = 1; j <= num; j++)
//				{
//					if (i >= weight[j - 1])
//						dp[j][i] = max(dp[j - 1][i], dp[j - 1][i - weight[j - 1]] + value[j - 1]);
//					else dp[j][i] = dp[j - 1][i];
//				}
//			}
//			return dp[num][capacity];
//		}
//		int Knapsack_Full(int num, int capacity, int v[], int w[])
//		{
//			std::vector<std::vector<int>> dp(num + 1, std::vector<int>(capacity + 1));
//			for (int i = 0; i <= capacity; i++)
//				dp[0][i] = i / v[0] * w[0];
//			for (int i = 1; i < num; i++)
//			{
//				for (int j = 0; j <= capacity; j++)
//				{
//					int max_choice = 0, x = 1;
//					while (j >= v[i] * x)
//						max_choice = max(max_choice, dp[i - 1][j - x * v[i]] + x * w[i]), x++;
//					dp[i][j] = max(dp[i - 1][j], max_choice);
//				}
//			}
//			return dp[num - 1][capacity];
//		}
//		int Knapsack_Many(int num, int capacity, int v[], int w[], int s[])
//		{
//			int dp[101] = { 0 };
//			for (int i = 0; i < num; i++)
//			{
//				if (s[i] * v[i] >= capacity)
//					for (int j = v[i]; j <= capacity; j++)
//						dp[j] = max(dp[j - v[i]] + w[i], dp[j]);
//				else for (int j = capacity; j >= v[i]; j--)
//					for (int k = s[i]; k >= 0; k--)
//						if (j >= k * v[i])
//							dp[j] = max(dp[j - k * v[i]] + k * w[i], dp[j]);
//			}
//			return dp[capacity];
//		}
//	};
//	namespace Structures
//	{
//		struct point
//		{
//			int x = 0, y = 0;
//			point() = default;
//			point(int x, int y) : x(x), y(y) {}
//			double dist(const point& other) const
//			{
//				return sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
//			}
//			double grid_dist(const point& other) const
//			{
//				return abs(x - other.x) + abs(y - other.y);
//			}
//			double chebyshev_dist(const point& other) const
//			{
//				return max(abs(x - other.x), abs(y - other.y));
//			}
//			double ptabs() const
//			{
//				return dist(point(0, 0));
//			}
//		};
//		std::ostream& operator<<(std::ostream& os, const point& pt)
//		{
//			os << pt.x << " " << pt.y;
//			return os;
//		}
//		std::istream& operator>>(std::istream& stream, point& pt)
//		{
//			stream >> pt.x >> pt.y;
//			return stream;
//		}
//		std::ostream& operator<<(std::ostream& os, const std::pair<int, int>& p)
//		{
//			os << p.first << " " << p.second;
//			return os;
//		}
//		std::istream& operator>>(std::istream& stream, std::pair<int, int>& p)
//		{
//			stream >> p.first >> p.second;
//			return stream;
//		}
//		template <typename T>
//		class SparseTable
//		{
//		private:
//			const int n;
//			int maxst;
//			std::vector<T> orig;
//			std::vector<std::vector<T>> st;
//			T(*func)(T, T);
//			bool start_from_one = true;
//			void init()
//			{
//				for (int i = 0; i < n; i++) st[i][0] = orig[i];
//				for (int p = 1, j = 1; j <= maxst; p *= 2, j++)
//					for (int i = 0; i + p < n; i++)
//						st[i][j] = func(st[i][j - 1], st[i + p][j - 1]);
//			}
//		public:
//			struct Functions
//			{
//				T min_func(T a, T b)
//				{
//					return min(a, b);
//				}
//				T max_func(T a, T b)
//				{
//					return max(a, b);
//				}
//				T gcd_func(T a, T b)
//				{
//					return std::gcd(a, b);
//				}
//				T lcm_func(T a, T b)
//				{
//					return std::lcm(a, b);
//				}
//			};
//			SparseTable(T* begin, T* end, T(*func)(T, T)) : func(func), n(end - begin)
//			{
//				maxst = ceil(log2(n));
//				for (T* i = begin; i <= end; i++) orig.push_back(*i);
//				std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
//				st = _st;
//				init();
//			}
//			SparseTable(const std::vector<T>& orig, T(*func)(T, T)) : func(func), n(orig.size()), orig(orig)
//			{
//				maxst = ceil(log2(n));
//				std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
//				st = _st;
//				init();
//			}
//			void SetStartFromZero()
//			{
//				start_from_one = false;
//			}
//			T query(const int& l, const int& r) const
//			{
//				int dist = floor(log2(r - l + 1));
//				if (start_from_one) return func(st[l - 1][dist], st[r - (1 << dist)][dist]);
//				else return func(st[l][dist], st[r - (1 << dist) + 1][dist]);
//			}
//		};
//		class JointSet
//		{
//		private:
//			const int n;
//			std::vector<int> fa;
//		public:
//			JointSet(int n) : n(n)
//			{
//				fa = std::vector<int>(n + 1);
//				for (int i = 1; i <= n; i++) fa[i] = i;
//			}
//			int find(int idx)
//			{
//				return fa[idx] == idx ? idx : find(fa[idx]);
//			}
//			void merge(int x, int y)
//			{
//				fa[find(x)] = find(y);
//			}
//			bool is_connected(int x, int y)
//			{
//				return find(x) == find(y);
//			}
//		};
//		class WeightedJointSet
//		{
//		private:
//			const int n;
//			std::vector<std::pair<int, int> > fa;
//		public:
//			WeightedJointSet(int n) : n(n)
//			{
//				fa = std::vector<std::pair<int, int> >(n + 1);
//				for (int i = 1; i <= n; i++) fa[i].first = i, fa[i].second = 0;
//			}
//			int find(int idx)
//			{
//				return fa[idx].first == idx ? idx : find(fa[idx].first);
//			}
//			void merge(int x, int y, int weight)
//			{
//				fa[find(x)].first = find(y);
//				fa[find(x)].second = weight;
//			}
//			bool is_connected(int x, int y)
//			{
//				return find(x) == find(y);
//			}
//		};
//		template <typename T>
//		class MonotonicStack // Calls a stack, but really a std::vector
//		{
//		private:
//			bool (*func)(T, T); // From Up to Down of stack
//			std::vector<T> stack;
//		public:
//			MonotonicStack(bool (*func)(T, T)) : func(func) {}
//			void push_back(const T& val)
//			{
//				while (!stack.empty() && func(stack.back(), val)) stack.pop_back();
//				stack.push_back(val);
//			}
//			void pop_back()
//			{
//				stack.pop_back();
//			}
//			T back()
//			{
//				return stack.back();
//			}
//			T operator[] (const int& idx) const
//			{
//				return stack[idx];
//			}
//			std::vector<T> GetStack()
//			{
//				return stack;
//			}
//		};
//		template <typename T>
//		class MonotonicDeque
//		{
//		private:
//			int cnt = 0;
//			std::deque<T> q, idx;
//			const bool has_idx = false;
//			bool (*func)(T, T); // From Back to Front of deque
//		public:
//			MonotonicDeque(bool (*func)(T, T)) : func(func) {}
//			MonotonicDeque(bool (*func)(T, T), bool has_idx) : func(func), has_idx(has_idx) {}
//			void push_back(const T& num)
//			{
//				while (!q.empty() && func(q.back(), num))
//				{
//					q.pop_back();
//					if (has_idx) idx.pop_back();
//				}
//				q.push_back(num);
//				if (has_idx) idx.push_back(++cnt);
//			}
//			void pop_back()
//			{
//				q.pop_back();
//			}
//			void push_front(const T& num)
//			{
//				q.push_front(num);
//			}
//			void pop_front()
//			{
//				q.pop_front();
//			}
//			void pop_front_until(int min_idx)
//			{
//				if (!has_idx) return;
//				while (idx.front() <= min_idx) idx.pop_front(), q.pop_front();
//			}
//		};
//		template <typename T>
//		class TreelikeArray
//		{
//		private:
//			const int n;
//			std::vector<int> a;
//		public:
//			TreelikeArray(int n) : n(n)
//			{
//				a = std::vector<int>(n + 1);
//			}
//			int lowbit(int x)
//			{
//				return x & -x;
//			}
//			int getsum(int x)
//			{
//				int cnt = 0;
//				while (x)
//				{
//					cnt = cnt + a[x];
//					x = x - lowbit(x);
//				}
//				return cnt;
//			}
//			int getsum(int l, int r)
//			{
//				return getsum(r) - getsum(l - 1);
//			}
//			void add(int x, int k)
//			{
//				while (x <= n)
//				{
//					a[x] += k;
//					x = x + lowbit(x);
//				}
//			}
//		};
//		template <typename T>
//		class Tree
//		{
//		private:
//			const int n;
//			int root, tree_stmax;
//			bool has_val = false, is_init_tree_st = false;
//			struct node
//			{
//				int fa = -1, depth = -1;
//				T val = 0;
//				std::vector<int> to;
//				node() {}
//				node(int fa) : fa(fa) {}
//				node(std::vector<int> to) : to(to) {}
//				node(std::vector<int> to, T val) : val(val), to(to) {}
//				node(int fa, std::vector<int> to) : fa(fa), to(to) {}
//				node(int fa, T val, std::vector<int> to) : fa(fa), val(val), to(to) {}
//			};
//			std::vector<std::vector<int> > tree_st;
//		public:
//			std::vector<node> tree;
//			Tree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) {}
//			Tree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) {}
//			Tree(int n, std::vector<int> to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//				init(root);
//			}
//			Tree(std::vector<int>* to_begin, std::vector<int>* to_end) : n(to_end - to_begin), root(1)
//			{
//				tree_stmax = ceil(log2(n));
//				for (std::vector<int>* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//				init(root);
//			}
//			Tree(int n, std::vector<int> to[], int root) : n(n), root(root), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//				init(root);
//			}
//			Tree(std::vector<int>* to_begin, std::vector<int>* to_end, int root) : n(to_end - to_begin), root(root)
//			{
//				tree_stmax = ceil(log2(n));
//				for (std::vector<int>* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//				init(root);
//			}
//			Tree(int n, std::pair<T, std::vector<int> > to[]) : has_val(true), n(n), root(1), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//				init(root);
//			}
//			Tree(std::pair<T, std::vector<int> >* to_begin, std::pair<T, std::vector<int> >* to_end) : has_val(true), n(to_end - to_begin), root(1)
//			{
//				tree_stmax = ceil(log2(n));
//				tree.push_back(node());
//				for (std::pair<T, std::vector<int> >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//				init(root);
//			}
//			Tree(int n, std::pair<T, std::vector<int> > to[], int root) : has_val(true), n(n), root(root), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//				init(root);
//			}
//			Tree(std::pair<T, std::vector<int> >* to_begin, std::pair<T, std::vector<int> >* to_end, int root) : has_val(true), n(to_end - to_begin), root(root)
//			{
//				tree_stmax = ceil(log2(n));
//				tree.push_back(node());
//				for (std::pair<T, std::vector<int> >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//				init(root);
//			}
//			void init(int now, int father = -1)
//			{
//				if (!has_val) tree[now].val = now;
//				tree[now].depth = tree[father].depth + 1;
//				tree[now].fa = father;
//				for (int& i : tree[now].to)
//					if (i != father)
//						init(i, now);
//			}
//			void init_tree_st()
//			{
//				tree_st = std::vector<std::vector<int> >(n, std::vector<int>(tree_stmax + 1, 0));
//				for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
//				for (int j = 1; j <= tree_stmax; j++)
//					for (int i = 1; i <= n; i++)
//						tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
//			}
//			void insert(int l, int r, int length)
//			{
//				tree[l].to.push_back(r);
//				tree[r].to.push_back(l);
//			}
//			int GetRoot()
//			{
//				return root;
//			}
//			int ChangeRoot(int new_root)
//			{
//				root = new_root;
//				init(new_root);
//			}
//			int GetNodeIdx(T val)
//			{
//				for (int i = 0; i < tree.size(); i++)
//					if (tree[i].val == val)
//						return i;
//			}
//			void dfs(void* func(node a), int start)
//			{
//				func(tree[start]);
//				for (int& i : tree[start].to)
//					if (i != tree[start].fa)
//						dfs(func, i);
//			}
//			int dfscnt(int func(node a), int start)
//			{
//				int cnt = func(tree[start]);
//				for (int& i : tree[start].to)
//					if (i != tree[start].fa)
//						cnt += dfscnt(func, i);
//				return cnt;
//			}
//			void bfs(void* func(node a), int start)
//			{
//				std::queue<int> idx_queue;
//				idx_queue.push(start);
//				while (!idx_queue.empty())
//				{
//					int idx = idx_queue.front();
//					func(tree[idx]);
//					for (int& i : tree[idx].to)
//						if (i != tree[idx].fa)
//							idx_queue.push(i);
//					idx_queue.pop();
//				}
//			}
//			int bfscnt(int func(node a), int start)
//			{
//				int cnt = 0;
//				std::queue<int> idx_queue;
//				idx_queue.push(start);
//				while (!idx_queue.empty())
//				{
//					int idx = idx_queue.front();
//					cnt += func(tree[idx]);
//					for (int& i : tree[idx].to)
//						if (i != tree[idx].fa)
//							idx_queue.push(i);
//					idx_queue.pop();
//				}
//				return cnt;
//			}
//			int lca(int l, int r)
//			{
//				if (!is_init_tree_st) init_tree_st(), is_init_tree_st = true;
//				int _l, _r;
//				if (has_val) _l = GetNodeIdx(l), _r = GetNodeIdx(r);
//				else _l = l, _r = r;
//				if (tree[_l].depth < tree[_r].depth) std::swap(_l, _r);
//				int diff = tree[_l].depth - tree[_r].depth;
//				for (int i = tree_stmax; i >= 0; i--) if ((1ll << i) & diff) _l = tree_st[_l][i];
//				for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
//				return (_l == _r ? _l : tree[_l].fa);
//			}
//		};
//		template <typename T>
//		class WeightedTree
//		{
//		private:
//			const int n;
//			int root, tree_stmax;
//			bool has_val = false, is_init_tree_st = false;
//			struct node
//			{
//				int fa = -1, depth = -1, segment_depth = -1;
//				T val = 0;
//				std::vector<std::pair<int, int> > to; // std::vector<<dest_idx, length>>
//				node() {}
//				node(int fa) : fa(fa) {}
//				node(std::vector<std::pair<int, int> > to) : to(to) {}
//				node(std::vector<std::pair<int, int> > to, T val) : val(val), to(to) {}
//				node(int fa, std::vector<std::pair<int, int> > to) : fa(fa), to(to) {}
//				node(int fa, T val, std::vector<std::pair<int, int> > to) : fa(fa), val(val), to(to) {}
//			};
//			std::vector<std::vector<int> > tree_st;
//		public:
//			std::vector<node> tree;
//			WeightedTree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) {}
//			WeightedTree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) {}
//			WeightedTree(int n, std::vector<std::pair<int, int> > to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//				init(root);
//			}
//			WeightedTree(std::vector<std::pair<int, int> >* to_begin, std::vector<std::pair<int, int> >* to_end) : n(to_end - to_begin), root(1)
//			{
//				tree_stmax = ceil(log2(n));
//				for (std::vector<std::pair<int, int> >* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//				init(root);
//			}
//			WeightedTree(int n, std::vector<std::pair<int, int> > to[], int root) : n(n), root(root), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
//				init(root);
//			}
//			WeightedTree(std::vector<std::pair<int, int> >* to_begin, std::vector<std::pair<int, int> >* to_end, int root) : n(to_end - to_begin), root(root)
//			{
//				tree_stmax = ceil(log2(n));
//				for (std::vector<std::pair<int, int> >* i = to_begin; i <= to_end; i++) tree.push_back(node(*i));
//				init(root);
//			}
//			WeightedTree(int n, std::pair<T, std::vector<std::pair<int, int> > > to[]) : has_val(true), n(n), root(1), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//				init(root);
//			}
//			WeightedTree(std::pair<T, std::vector<std::pair<int, int> > >* to_begin, std::pair<T, std::vector<std::pair<int, int> > >* to_end) : has_val(true), n(to_end - to_begin), root(1)
//			{
//				tree_stmax = ceil(log2(n));
//				tree.push_back(node());
//				for (std::pair<T, std::vector<std::pair<int, int> > >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//				init(root);
//			}
//			WeightedTree(int n, std::pair<T, std::vector<std::pair<int, int> > > to[], int root) : has_val(true), n(n), root(root), tree_stmax(ceil(log2(n)))
//			{
//				tree.push_back(node());
//				for (int i = 1; i <= n; i++) tree.push_back(node(to[i].second, to[i].first));
//				init(root);
//			}
//			WeightedTree(std::pair<T, std::vector<std::pair<int, int> > >* to_begin, std::pair<T, std::vector<std::pair<int, int> > >* to_end, int root) : has_val(true), n(to_end - to_begin), root(root)
//			{
//				tree_stmax = ceil(log2(n));
//				tree.push_back(node());
//				for (std::pair<T, std::vector<std::pair<int, int> > >* i = to_begin; i <= to_end; i++) tree.push_back(node((*i).second, (*i).first));
//				init(root);
//			}
//			void init(int now, int father = -1, int length = -1)
//			{
//				if (!has_val) tree[now].val = now;
//				tree[now].depth = tree[father].depth + 1;
//				tree[now].segment_depth = tree[father].segment_depth + length;
//				tree[now].fa = father;
//				for (std::pair<int, int>& i : tree[now].to)
//					if (i.first != father)
//						init(i.first, now, i.second);
//			}
//			void init_tree_st()
//			{
//				tree_st = std::vector<std::vector<int> >(n, std::vector<int>(tree_stmax + 1, 0));
//				for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
//				for (int j = 1; j <= tree_stmax; j++)
//					for (int i = 1; i <= n; i++)
//						tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
//			}
//			void insert(int l, int r, int length)
//			{
//				tree[l].to.push_back(std::make_pair(r, length));
//				tree[r].to.push_back(std::make_pair(l, length));
//			}
//			int GetRoot()
//			{
//				return root;
//			}
//			int ChangeRoot(int new_root)
//			{
//				root = new_root;
//				init(new_root);
//			}
//			int GetNodeIdx(T val)
//			{
//				for (int i = 0; i < tree.size(); i++)
//					if (tree[i].val == val)
//						return i;
//			}
//			void dfs(void* func(node a), int start)
//			{
//				func(tree[start]);
//				for (std::pair<int, int>& i : tree[start].to)
//					if (i.first != tree[start].fa)
//						dfs(func, i.first);
//			}
//			int dfscnt(int func(node a), int start)
//			{
//				int cnt = func(tree[start]);
//				for (std::pair<int, int>& i : tree[start].to)
//					if (i.first != tree[start].fa)
//						cnt += dfscnt(func, i.first);
//				return cnt;
//			}
//			void bfs(void* func(node a), int start)
//			{
//				std::queue<int> idx_queue;
//				idx_queue.push(start);
//				while (!idx_queue.empty())
//				{
//					int idx = idx_queue.front();
//					func(tree[idx]);
//					for (std::pair<int, int>& i : tree[idx].to)
//						if (i.first != tree[idx].fa)
//							idx_queue.push(i.first);
//					idx_queue.pop();
//				}
//			}
//			int bfscnt(int func(node a), int start)
//			{
//				int cnt = 0;
//				std::queue<int> idx_queue;
//				idx_queue.push(start);
//				while (!idx_queue.empty())
//				{
//					int idx = idx_queue.front();
//					cnt += func(tree[idx]);
//					for (std::pair<int, int>& i : tree[idx].to)
//						if (i.first != tree[idx].fa)
//							idx_queue.push(i.first);
//					idx_queue.pop();
//				}
//				return cnt;
//			}
//			int lca(int l, int r)
//			{
//				if (!is_init_tree_st) init_tree_st(), is_init_tree_st = true;
//				int _l, _r;
//				if (has_val) _l = GetNodeIdx(l), _r = GetNodeIdx(r);
//				else _l = l, _r = r;
//				if (tree[_l].depth < tree[_r].depth) std::swap(_l, _r);
//				int diff = tree[_l].depth - tree[_r].depth;
//				for (int i = tree_stmax; i >= 0; i--) if ((1 << i) & diff) _l = tree_st[_l][i];
//				for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
//				return (_l == _r ? _l : tree[_l].fa);
//			}
//		};
//	}
//}
