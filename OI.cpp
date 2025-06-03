#pragma once
#include "Everything.h"
#ifdef __GNUC__
#define gcd __gcd
#endif

namespace OI
{
	class Sort
	{
	private:
		template <class T>
		int GetMax(T arr, int n)
		{
			auto max = arr[0];
			for (int i = 1; i < n; i++)
				if (arr[i] > max)
					max = arr[i];
			return max;
		}
		template <class T>
		void Merge(T arr, int p, int q, int r)
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
		template <class T>
		void MergeRecursion(T arr, int l, int r)
		{
			// Time & Space complexity: O(n*log(n)) / O(n)
			if (l < r)
			{
				int m = l + (r - l) / 2;
				MergeRecursion(arr, l, m);
				MergeRecursion(arr, m + 1, r);
				Merge(arr, l, m, r);
			}
		}
		template <class T>
		int Partition(T arr, int low, int high)
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
		template <class T>
		void QuickRecursion(T arr, int low, int high)
		{
			if (low < high)
			{
				int p = Partition(arr, low, high);
				QuickRecursion(arr, low, p - 1);
				QuickRecursion(arr, p + 1, high);
			}
		}
		template <class T>
		void Heapify(T arr, int N, int i)
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
		template <class T>
		void SortDigit(T arr, int n, int exp, int base)
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
		template <class T>
		void BubbleSort(T arr, T end)
		{
			int len = end - arr;
			// Time & Space complexity: O(n^2) / O(1)
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
		template <class T>
		void SelectionSort(T arr, T end)
		{
			int len = end - arr;
			// Time & Space complexity: O(n^2) / O(1)
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
		template <class T>
		void InsertionSort(T arr, T end)
		{
			int len = end - arr;
			// Time & Space complexity: O(n^2) / O(1)
			int i, key, j;
			for (i = 1; i < len; i++)
			{
				key = arr[i];
				for (j = i - 1; j >= 0 && arr[j] > key; j--)
					arr[j + 1] = arr[j];
				arr[j + 1] = key;
			}
		}
		template <class T>
		void MergeSort(T arr, T end)
		{
			int len = end - arr;
			MergeRecursion(arr, 0, len - 1);
		}
		template <class T>
		void QuickSort(T arr, T end)
		{
			int len = end - arr;
			QuickRecursion(arr, 0, len - 1);
		}
		template <class T>
		void HeapSort(T arr, T end)
		{
			int len = end - arr;
			for (int i = len / 2 - 1; i >= 0; i--)
				Heapify(arr, len, i);
			for (int i = len - 1; i > 0; i--)
			{
				std::swap(arr[0], arr[i]);
				Heapify(arr, i, 0);
			}
		}
		template <class T>
		void RadixSort(T arr, T end, const int& base = 10)
		{
			int len = end - arr;
			int m = GetMax(arr, len);
			for (int exp = 1; m / exp > 0; exp *= base)
				SortDigit(arr, len, exp, base);
		}
		template <class T>
		void CountingSort(T arr, T end)
		{
			int len = end - arr;
			int max = GetMax(arr, len);
			std::vector<int> count(max + 1), output(len);
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
		template <class T>
		void BucketSort(T arr, T end, double (*compress_func) (int) = sqrt)
		{
			int len = end - arr;
			int div = (int)compress_func(GetMax(arr, len));
			std::vector<std::vector<int>> b(div + 1);
			for (int i = 0; i < len; i++)
				b[(int)compress_func(arr[i])].push_back(arr[i]);
			for (int i = 0; i <= div; i++)
				std::sort(b[i].begin(), b[i].end());
			int idx = 0;
			for (int i = 0; i <= div; i++)
				for (int&j : b[i])
					arr[idx++] = j;
		}
	};
#define int long long
	class Number
	{
	private:
		unsigned long long MOD, SIZE, BASE = 0;
		bool init_fac = false;
		std::vector<int> fac, inv, p, primes, is_prime;
	public:
		Number(int size, int mod) : SIZE(size), MOD(mod) {}
		void SetMod(int mod)
		{
			MOD = mod;
		}
		void SetSize(int size)
		{
			SIZE = size;
		}
		void SetBase(int base)
		{
			BASE = base;
		}
		void InitFac(int mod = 0)
		{
			if (init_fac) return;
			init_fac = true;
			if (mod == 0) mod = MOD;
			fac.push_back(1);
			for (int i = 1; i < SIZE; i++) fac.push_back(fac.back() * i % mod);
			inv.resize(SIZE);
			inv[SIZE - 1] = qpow(fac[SIZE - 1], mod - 2);
			for (int i = SIZE - 2; i >= 0; i--) inv[i] = inv[i + 1] * (i + 1) % mod;
		}
		void InitBase(int base = 0, unsigned long long mod = 0)
		{
			if (base == 0) base = BASE;
			if (mod == 0) mod = MOD;
			p[0] = 1;
			for (int i = 1; i < SIZE; i++) p[i] = p[i - 1] * base % mod;
		}
		void InitPrimes(int n)
		{
			is_prime.resize(n + 1);
			for (int i = 2; i <= n; i++)
				is_prime[i] = true;
			for (int i = 2; i <= n; i++)
			{
				if (is_prime[i])
					primes.push_back(i);
				for (int j = 0; j < primes.size() && primes[j] * i <= n; j++)
				{
					is_prime[primes[j] * i] = false;
					if (!(i % primes[j]))
						break;
				}
			}
		}
		bool IsPrime(int n)
		{
			return is_prime[n];
		}
		int GetNthPrime(int n)
		{
			return primes[n - 1];
		}
		int qpow(int u, int v, unsigned long long mod = 0)
		{
			if (mod == 0) mod = MOD;
			int res = 1;
			while (v)
			{
				if (v & 1) res = res * u % mod;
				u = u * u % mod;
				v >>= 1;
			}
			return res;
		}
		int inverse(int x, unsigned long long mod = 0)
		{
			if (mod == 0) mod = MOD;
			return qpow(x, mod - 2, mod);
		}
		int C(int u, int v, unsigned long long mod = 0)
		{
			if (mod == 0) mod = MOD;
			InitFac(mod);
			if (u < v) return 0;
			if (u < mod && v < mod)	return fac[u] * inverse(fac[v]) % mod * inverse(fac[u - v]) % mod;
			else return C(u % mod, v % mod) * C(u / mod, v / mod);
		}
	};
#undef int
	class Algorithms
	{
	private:
		static void FFTCalculate(std::complex<double> A[], int ord[], int n)
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
		static std::vector<int> FFT(const std::vector<int>& a, const std::vector<int>& b)
		{
			// Calculate a * b = output where a, b, output are polynomials.
			const int MAXN = max(a.size(), b.size()) << 1 + 7;
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
		static int Knapsack_01(int num, int capacity, int weight[], int value[])
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
		static int Knapsack_Full(int num, int capacity, int v[], int w[])
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
		static int Knapsack_Many(int num, int capacity, int v[], int w[], int s[])
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
	namespace Structures
	{
		static struct point
		{
			int x = 0, y = 0;
			point() = default;
			point(int x, int y) : x(x), y(y) {}
			double dist(const point& other)
			{
				return sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
			}
			double grid_dist(const point& other)
			{
				return abs(x - other.x) + abs(y - other.y);
			}
			double chebyshev_dist(const point& other)
			{
				return max(abs(x - other.x), abs(y - other.y));
			}
			double absolute()
			{
				return dist(point(0, 0));
			}
		};
		static std::ostream& operator<<(std::ostream& os, const point& pt)
		{
			os << pt.x << " " << pt.y;
			return os;
		}
		static std::istream& operator>>(std::istream& stream, point& pt)
		{
			stream >> pt.x >> pt.y;
			return stream;
		}
		static std::ostream& operator<<(std::ostream& os, const std::pair<int, int>& p)
		{
			os << p.first << " " << p.second;
			return os;
		}
		static std::istream& operator>>(std::istream& stream, std::pair<int, int>& p)
		{
			stream >> p.first >> p.second;
			return stream;
		}
		class SegmentTree
		{
		private:
			int n;
			std::vector<int> a, lazy;
		public:
			template <class T>
			SegmentTree(T begin, T end)
			{
				n = end - begin;
				a.resize(2 * n);
				lazy.resize(2 * n);
				Build(begin);
			}
			void PushDown(int idx, int l, int r)
			{
				if (lazy[idx])
				{
					lazy[idx * 2] += lazy[idx];
					lazy[idx * 2 + 1] += lazy[idx];
					int mid = l + r >> 1;
					a[idx * 2] += lazy[idx] * (mid - l + 1);
					a[idx * 2 + 1] += lazy[idx] * (r - mid);
					lazy[idx] = 0;
				}
			}
			void PushUp(int idx)
			{
				a[idx] = a[idx * 2] + a[idx * 2 + 1];
			}
			void Build(int* init, int idx = 1, int l = 1, int r = 0)
			{
				if (r == 0) r = n;
				if (l == r) { a[idx] = init[l - 1]; return; }
				int mid = l + r >> 1;
				Build(init, 2 * idx, l, mid);
				Build(init, 2 * idx + 1, mid + 1, r);
				PushUp(idx);
			}
			void ChangeVal(int lidx, int ridx, int v, int idx = 1, int l = 1, int r = 0)
			{
				if (r == 0) r = n;
				if (lidx <= l && r <= ridx) { a[idx] += (r - l + 1) * v, lazy[idx] += v; return; }
				PushDown(idx, l, r);
				int mid = l + r >> 1;
				if (mid >= lidx) ChangeVal(lidx, ridx, v, 2 * idx, l, mid);
				if (mid < ridx) ChangeVal(lidx, ridx, v, 2 * idx + 1, mid + 1, r);
				PushUp(idx);
			}
			int Query(int lidx, int ridx, int idx = 1, int l = 1, int r = 0)
			{
				if (r == 0) r = n;
				if (l >= lidx && ridx >= r) return a[idx];
				if (lidx > r || ridx < l) return 0;
				PushDown(idx, l, r);
				int mid = l + r >> 1, ret = 0;
				ret += Query(lidx, ridx, 2 * idx, l, mid);
				ret += Query(lidx, ridx, 2 * idx + 1, mid + 1, r);
				return ret;
			}
		};
		template <typename T>
		class SparseTable
		{
		private:
			int n, maxst;
			std::vector<T> orig;
			std::vector<std::vector<T>> st;
			T(*func)(T, T);
			bool start_from_one;
			void init()
			{
				for (int i = 0; i < n; i++) st[i][0] = orig[i];
				for (int p = 1, j = 1; j <= maxst; p *= 2, j++)
					for (int i = 0; i + p < n; i++)
						st[i][j] = func(st[i][j - 1], st[i + p][j - 1]);
			}
		public:
			SparseTable(T* begin, T* end, T(*_func)(T, T)) : n(end - begin)
			{
				start_from_one = true;
				func = _func;
				maxst = ceil(log2(n));
				for (T* i = begin; i <= end; i++) orig.push_back(*i);
				std::vector<std::vector<T>> _st(n, std::vector<T>(maxst + 1, 0));
				st = _st;
				init();
			}
			SparseTable(const std::vector<T>& orig, T(*_func)(T, T)) : n(orig.size()), orig(orig)
			{
				start_from_one = true;
				func = _func;
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
				return fa[idx] == idx ? idx : fa[idx] = find(fa[idx]);
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
				a.resize(n + 1);
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
					x -= lowbit(x);
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
			int root, tree_stmax, dfn_cnt = 0ll, euler_cnt = 0ll;
			bool has_val = false, is_init_tree_st = false;
			struct node
			{
				int fa = -1, depth = -1, dfn = -1, euler = -1;
				T val = 0;
				std::vector<int> to;
				node() {}
				node(int fa) : fa(fa) {}
				node(std::vector<int> to) : to(to) {}
				node(std::vector<int> to, T val) : val(val), to(to) {}
				node(int fa, std::vector<int> to) : fa(fa), to(to) {}
				node(int fa, T val, std::vector<int> to) : fa(fa), val(val), to(to) {}
			};
			std::vector<int> euler = { 0 }, ant = { 0 };
			std::vector<std::vector<int> > tree_st;
		public:
			std::vector<node> tree;
			Tree(int n) : n(n), root(1), tree_stmax(ceil(log2(n))) { tree.resize(n + 1); }
			Tree(int n, int root) : n(n), root(root), tree_stmax(ceil(log2(n))) { tree.resize(n + 1); }
			Tree(int n, std::vector<int> to[]) : n(n), root(1), tree_stmax(ceil(log2(n)))
			{
				tree.push_back(node());
				for (int i = 1; i <= n; i++) tree.push_back(node(to[i]));
				init(root);
			}
			Tree(std::vector<int>* to_begin, std::vector<int>* to_end) : n(to_end - to_begin), root(1)
			{
				tree_stmax = ceil(log2(n));
				tree.push_back(node());
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
				tree.push_back(node());
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
			int GetN() { return n; }
			void init(int now, int father = 0)
			{
				if (!has_val) tree[now].val = now;
				tree[now].depth = tree[father].depth + 1;
				tree[now].fa = father;
				tree[now].dfn = ++dfn_cnt;
				ant.push_back(now);
				tree[now].euler = ++euler_cnt;
				euler.push_back(now);
				for (int& i : tree[now].to)
					if (i != father)
						init(i, now), euler.push_back(now), ++euler_cnt;
			}
			void init_tree_st()
			{
				tree_st = std::vector<std::vector<int> >(n + 1, std::vector<int>(tree_stmax + 1, 0));
				for (int i = 1; i <= n; i++) tree_st[i][0] = tree[i].fa;
				for (int j = 1; j <= tree_stmax; j++)
					for (int i = 1; i <= n; i++)
						tree_st[i][j] = tree_st[tree_st[i][j - 1]][j - 1];
			}
			void insert(int l, int r)
			{
				tree[l].to.push_back(r);
				tree[r].to.push_back(l);
			}
			int GetRoot()
			{
				return root;
			}
			void ChangeRoot(int new_root)
			{
				root = new_root;
				dfn_cnt = 0, euler_cnt = 0;
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
				for (int i = tree_stmax; i >= 0; i--) if ((1ll << i) & diff) _l = tree_st[_l][i];
				for (int i = tree_stmax; i >= 0; i--) if (tree_st[_l][i] != tree_st[_r][i]) _l = tree_st[_l][i], _r = tree_st[_r][i];
				return (_l == _r ? _l : tree[_l].fa);
			}
		};
		static std::istream& operator>>(std::istream& in, Tree<int>& tree)
		{
			for (int i = 1; i < tree.GetN(); i++)
			{
				int u, v; std::cin >> u >> v;
				tree.insert(u, v);
			}
			return in;
		}
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
}
